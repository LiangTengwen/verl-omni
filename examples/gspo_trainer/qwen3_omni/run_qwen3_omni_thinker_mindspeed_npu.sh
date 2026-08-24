#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO training on NPU using MindSpeed Megatron backend.
#
# This script demonstrates how to run Qwen3-Omni Thinker RL training on
# Huawei Ascend NPU with the MindSpeed Megatron training backend.
#
# Key differences from the GPU Megatron recipe (PR #399):
#   1. model_type=omni_model → triggers OmniMindSpeedMegatronEngine
#   2. strategy=mindspeed_megatron → uses MindSpeed (CANN) instead of Megatron-Core (CUDA)
#   3. vanilla_mbridge=True → uses old mbridge (NPU requirement)
#   4. rollout.name=vllm_omni → uses vLLM-Omni (standard rollout backend for Qwen3-Omni)
#   5. model_engine=mindspeed → enables MindSpeed engine config
#
# Hardware: Atlas 800T A3 (Ascend 910C NPU)
# Software: mindspeed_llm + megatron-core (CANN) + vLLM-Omni (NPU)
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

# --------------- SGLang Generation Configuration ---------------
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

# --------------- Actor Configuration (MindSpeed Megatron) ---------------
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
    # MindSpeed Megatron Parallelism Strategy
    actor_rollout_ref.actor.strategy=mindspeed_megatron
    actor_rollout_ref.actor.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.mindspeed.pipeline_model_parallel_size=${train_pp}
    # Memory Optimization
    actor_rollout_ref.actor.mindspeed.param_offload=True
    actor_rollout_ref.actor.mindspeed.optimizer_offload=True
    actor_rollout_ref.actor.mindspeed.grad_offload=True
    # Model Weights Management (old mbridge for NPU)
    actor_rollout_ref.actor.mindspeed.use_mbridge=True
    actor_rollout_ref.actor.mindspeed.vanilla_mbridge=True
    # Transformer Architecture Optimizations (Qwen3-MoE spec)
    actor_rollout_ref.actor.mindspeed.mcore_kwargs.spec='[mindspeed_llm.tasks.models.spec.qwen3_spec, layer_spec]'
    actor_rollout_ref.actor.mindspeed.mcore_kwargs.seq_length=${max_model_len}
    actor_rollout_ref.actor.mindspeed.mcore_kwargs.micro_batch_size=${micro_batch_size}
    +actor_rollout_ref.actor.mindspeed.mcore_kwargs.num_query_groups=8
    +actor_rollout_ref.actor.mindspeed.mcore_kwargs.recompute_method=uniform
    +actor_rollout_ref.actor.mindspeed.mcore_kwargs.recompute_granularity=full
    +actor_rollout_ref.actor.mindspeed.mcore_kwargs.recompute_num_layers=1
    +actor_rollout_ref.actor.mindspeed.mcore_kwargs.overlap_grad_reduce=True
    +actor_rollout_ref.actor.mindspeed.mcore_kwargs.overlap_param_gather=True
)

# --------------- Reference Model Configuration (MindSpeed Megatron) ---------------
REF_CONFIG=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_model_len}
    # MindSpeed Megatron Parallelism Strategy
    actor_rollout_ref.ref.strategy=mindspeed_megatron
    actor_rollout_ref.ref.mindspeed.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.mindspeed.pipeline_model_parallel_size=${train_pp}
    # Memory Optimization
    actor_rollout_ref.ref.mindspeed.param_offload=True
    # Model Weights Management
    actor_rollout_ref.ref.mindspeed.use_mbridge=True
    actor_rollout_ref.ref.mindspeed.vanilla_mbridge=True
)

# --------------- vLLM-Omni Stage Config (NPU Thinker-only) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/qwen3_omni_thinker_only_npu.yaml"

# --------------- Rollout Configuration (vLLM-Omni) ---------------
ROLLOUT_CONFIG=(
    # Use vLLM-Omni as the rollout backend (standard for Qwen3-Omni on NPU).
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
    trainer.experiment_name='gspo_mindspeed_npu'
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
    model_engine=mindspeed \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${REWARD_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    "$@"