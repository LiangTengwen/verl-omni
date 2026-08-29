#!/usr/bin/env bash
# Qwen3.5-35B-A3B (MoE) Thinker-only GSPO Megatron full-parameter training, omni V1 trainer.
# Modalities: text + image (+ video) -> text.
# Hardware: Ascend NPU (910B/910C, 64GB)。卡数由 NUM_NPUS 决定，默认自动探测。
#
# Training backend: Megatron (model_type=language_model + strategy=megatron).
#   On NPU this dispatches to verl upstream's MindspeedEngineWithLMHead, which
#   replaces CUDA ops with CANN ops via mindspeed.megatron_adaptor.repatch().
#   Weights are converted HF -> Megatron through Megatron-Bridge
#   (vanilla_mbridge=False), which provides the multimodal provider that keeps
#   the vision tower in the graph (frozen) alongside the LLM.
#
# Rollout backend: vLLM-Omni (qwen3_5_moe thinker-only pipeline), reused from
#   the FSDP recipe.  Registering qwen35moe in verl_omni/pipelines/__init__.py
#   is what makes pipeline_name=qwen3_5_moe resolve.
#
# Config routing: uses --config-name=omni_megatron_trainer.yaml, whose defaults
#   are `defaults: ppo_megatron_trainer` + `model_type: language_model` +
#   `strategy: megatron` + `trainer.v1.trainer_mode: sync`.  The sync trainer
#   forces checkpoint_engine.backend=naive, so weights flow Megatron (via
#   get_per_tensor_param) -> vLLM-Omni rollout in place, NOT via an nccl engine.
#
# Qwen3.5 GDN (linear attention) note (from upstream verl run_qwen3_5_35b_megatron.sh):
#   GDN does NOT support packed sequences (THD) in Megatron-LM, therefore:
#     - actor_rollout_ref.model.use_remove_padding=False
#     - actor_rollout_ref.actor.megatron.use_remove_padding=False
#     - actor_rollout_ref.actor.use_dynamic_bsz=False
set -x

# =============================================================================
# §1. Ascend / vLLM 运行时（沿用 FSDP 脚本的已验证值，均为后端无关项）
# =============================================================================
export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0            # 配合 rollout adapter 的 weight_nz_mode=0
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export USE_OPTIMIZED_MODEL=0
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export NPU_GRAPH_CAPTURE_DISABLE=1
export ASCEND_SECURITY_CHECK_DISABLED=1
export WANDB_MODE=${WANDB_MODE:-offline}

export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy

export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1

unset PYTORCH_NPU_ALLOC_CONF

# =============================================================================
# §2. HCCL / 通信超时
# =============================================================================
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV

SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME}

# =============================================================================
# §3. Ray 运行时
# =============================================================================
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"

set -eo pipefail

VERL_OMNI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# =============================================================================
# §4. 路径与卡数
# =============================================================================
MODEL_PATH=${MODEL_PATH:-"/opt/huawei/dataset/downloaded_models/Qwen3.5-35B-A3B"}
TRAIN_FILE=${TRAIN_FILE:-"/opt/huawei/dataset/ccb_data/data/omni/geo3k/train.parquet"}
VAL_FILE=${VAL_FILE:-"/opt/huawei/dataset/ccb_data/data/omni/geo3k/test.parquet"}

_DETECTED_NPUS=$(ls -d /dev/davinci[0-9]* 2>/dev/null | wc -l | tr -d ' ')
NUM_NPUS=${NUM_NPUS:-${_DETECTED_NPUS:-8}}
NNODES=${NNODES:-${VC_WORKER_NUM:-${MA_NUM_HOSTS:-1}}}

if [ "${NUM_NPUS}" -le 0 ] 2>/dev/null; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} is not a positive integer." >&2
    exit 1
fi

# ---- 数据文件防护 ----
for _v in MODEL_PATH TRAIN_FILE VAL_FILE; do
    _p="${!_v}"
    if [ -z "${_p}" ]; then
        echo "FATAL: ${_v} is empty. Set it explicitly." >&2
        exit 1
    fi
    if [ ! -e "${_p}" ]; then
        echo "FATAL: ${_v} does not exist: ${_p}" >&2
        exit 1
    fi
done

# =============================================================================
# §5. Megatron 并行度
# =============================================================================
# 训练侧拓扑约束：world = TP × PP × CP × DP，且 MoE 的 EP 必须整除 DP。
# 默认按单机 8 卡给一套安全、总能通过的配置；多机/大卡请按需覆盖。
TP=${TP:-2}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}

_WORLD_SIZE=$(( NUM_NPUS * NNODES ))
_TP_PP_CP=$(( TP * PP * CP ))
if [ $(( _WORLD_SIZE % _TP_PP_CP )) -ne 0 ]; then
    echo "FATAL: world=${_WORLD_SIZE} 不被 TP×PP×CP=${_TP_PP_CP} 整除" >&2
    exit 1
fi
_DP=$(( _WORLD_SIZE / _TP_PP_CP ))
if [ $(( _DP % EP )) -ne 0 ]; then
    echo "FATAL: EP=${EP} 必须整除 DP=${_DP}" >&2
    exit 1
fi
echo "Megatron topology: world=${_WORLD_SIZE} TP=${TP} PP=${PP} CP=${CP} DP=${_DP} EP=${EP} ETP=${ETP}"

# rollout(vLLM-Omni) 张量并行，与训练 TP 解耦（沿用 FSDP 脚本）
ROLLOUT_TP=${ROLLOUT_TP:-4}
if [ $(( NUM_NPUS % ROLLOUT_TP )) -ne 0 ]; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} 不被 ROLLOUT_TP=${ROLLOUT_TP} 整除" >&2
    exit 1
fi
ROLLOUT_NUM_WORKERS=$(( NUM_NPUS / ROLLOUT_TP ))

# =============================================================================
# §6. 训练超参
# =============================================================================
# Megatron 用 bf16 计算 + 自带 fp32 master/optimizer state（distributed optimizer），
# 不存在 FSDP 那条「显式 bf16 顶掉 fp32 强制」的坑，因此这里保持 bfloat16。
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-bfloat16}

# 显存优化：param/grad/optimizer 全下放（cpu），35B MoE 专家权重在 64GB 卡上基本是必需。
ALL_OFFLOAD=${ALL_OFFLOAD:-True}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-5}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
_SEQ_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$(( _SEQ_LEN + 128 ))}

RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
PROJECT_NAME=${PROJECT_NAME:-qwen35_moe_gspo_megatron}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gspo_qwen35_moe_megatron_npu}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${VERL_OMNI_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${VERL_OMNI_ROOT}/tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${TENSORBOARD_DIR}" "${DEFAULT_LOCAL_DIR}"

# =============================================================================
# §7. 启动
# =============================================================================
python3 -m verl_omni.trainer.main_omni \
    --config-name=omni_megatron_trainer \
    trainer.device=npu \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.image_key=images \
    data.shuffle=true \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=64 \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=False \
    actor_rollout_ref.model.use_remove_padding=False \
    \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.dtype=${ACTOR_MODEL_DTYPE} \
    actor_rollout_ref.actor.checkpoint.strict=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True \
    \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_NUM_WORKERS} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_5_moe \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    \
    algorithm.adv_estimator=grpo \
    algorithm.trainer_type=policy_gradient \
    algorithm.sample_source=online \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.n_gpus_per_node=${NUM_NPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    "$@" \
    2>&1 | tee run_qwen35_moe_gspo_megatron_npu.log