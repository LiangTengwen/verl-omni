#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO training on NPU using Megatron-Bridge (new bridge).
#
# This script uses the new Megatron-Bridge (vanilla_mbridge=False) which natively
# supports Qwen3-Omni architecture detection.  No hf_config swapping or thinker
# weight extraction is needed -- the bridge handles everything automatically.
#
# Key differences from the old mindspeed_megatron recipe:
#   1. strategy=megatron (not mindspeed_megatron) -- uses new Megatron-Bridge path
#   2. vanilla_mbridge=False (not True) -- uses new Megatron-Bridge
#   3. actor.megatron.* config namespace (not actor.mindspeed.*)
#   4. model_engine=megatron (not mindspeed)
#   5. model_type=omni_model triggers OmniMindspeedMegatronBridgeEngine
#
# Hardware: Atlas 800T A3 (Ascend 910C NPU)
# Software: mindspeed_llm + megatron-core (CANN) + Megatron-Bridge + vLLM-Omni (NPU)
set -x

# --------------- Environment Variables ---------------
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export VERL_USE_EXTERNAL_MODULES=verl_omni

# --------------- Node Configuration ---------------
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}

# --------------- Model Configuration ---------------
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/math/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/math/test.parquet"}

# --------------- Data Length Configuration ---------------
max_prompt_length=512
max_response_length=1024

# --------------- Training Batch Configuration ---------------
train_prompt_bsz=8
train_prompt_mini_bsz=8
n_resp_per_prompt=2
micro_batch_size=1

# --------------- Algorithm Configuration ---------------
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

# --------------- Megatron Parallelism Configuration ---------------
train_tp=4
train_pp=2

# --------------- vLLM-Omni Generation Configuration ---------------
gen_tp=4
gen_dp=1
gpu_memory_utilization=0.5
max_model_len=$((max_prompt_length + max_response_length))

# --------------- Data Configuration ---------------
DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='left'
)

# --------------- Model Configuration ---------------
MODEL_CONFIG=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.model_type=omni_model
    actor_rollout_ref.model.use_remove_padding=True
)

# --------------- Actor Configuration (Megatron-Bridge) ---------------
ACTOR_CONFIG=(
    # Core Runtime Settings
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=False
    # Loss Function Configuration
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=0
    # PPO Training Parameters
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_model_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    # Optimizer Settings
    actor_rollout_ref.actor.optim.lr=1e-6
    # Megatron-Bridge Strategy (new bridge, vanilla_mbridge=False)
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_mbridge=True
    # Megatron Parallelism
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    # Memory Optimization
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    # NPU-specific overrides (MindSpeed CANN operator replacement)
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
)

# --------------- Reference Model Configuration (Megatron-Bridge) ---------------
REF_CONFIG=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_model_len}
    # Megatron-Bridge Strategy
    actor_rollout_ref.ref.strategy=megatron
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    # Memory Optimization
    actor_rollout_ref.ref.megatron.param_offload=True
    # Bridge Configuration
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
)

# --------------- vLLM-Omni Stage Config (NPU Thinker-only) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/qwen3_omni_thinker_only_npu.yaml"

# --------------- Rollout Configuration (vLLM-Omni) ---------------
ROLLOUT_CONFIG=(
    # Use vLLM-Omni as the rollout backend (standard for Qwen3-Omni).
    actor_rollout_ref.rollout.name=vllm_omni
    actor_rollout_ref.rollout.mode=async
    # vLLM-Omni pipeline registration (qwen3_omni_moe is the registered pipeline name).
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe"
    # Thinker-only stage config (single stage, all GPUs for TP).
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}"
    # Autoregressive output mode for text-token rollout.
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar
    # Sampling parameters.
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    # Log probability computation.
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_model_len}
    # Memory and parallelism configuration.
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    actor_rollout_ref.rollout.enforce_eager=False
    # Validation configuration.
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
)

# --------------- Reward Configuration ---------------
REWARD_CONFIG=(
    reward.reward_manager.source=register
    reward.reward_manager.name=naive
)

# --------------- Algorithm Configuration ---------------
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

# --------------- Trainer Configuration ---------------
TRAINER_CONFIG=(
    trainer.logger='["console","tensorboard"]'
    trainer.project_name='qwen3_omni_thinker_rl'
    trainer.experiment_name='gspo_megatron_bridge_npu'
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    trainer.total_epochs=1
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.total_training_steps=1
)

# --------------- Launch Training ---------------
python3 -m verl_omni.trainer.main_omni \
    --config-path="${SCRIPT_DIR:-.}/config" \
    --config-name='qwen3_omni_thinker_gspo' \
    model_engine=megatron \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${REWARD_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "$@"