#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO full-parameter AVQA training with the omni V1 trainer.
# Modalities: text + image + audio -> text.
# Hardware: Atlas 800T A3 (16 × Ascend 910C 64GB).
set -x

export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export USE_OPTIMIZED_MODEL=0
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export NPU_GRAPH_CAPTURE_DISABLE=1
export WANDB_MODE=${WANDB_MODE:-offline}

export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export ASCEND_SECURITY_CHECK_DISABLED=1

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"

# after the Ascend env scripts, which are not -e clean
set -eo pipefail

MODEL_PATH=${MODEL_PATH:-"/opt/huawei/dataset/hfy_data/models/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"/opt/huawei/dataset/hfy_data/dataset/avqa_r1_6k/train.parquet"}
VAL_FILE=${VAL_FILE:-"/opt/huawei/dataset/hfy_data/dataset/avqa_r1_6k/validation.parquet"}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-16}
NNODES=${VC_WORKER_NUM:-1}
ROLLOUT_TP=${ROLLOUT_TP:-4}

python3 -m verl_omni.trainer.main_omni \
    --config-name='omni_npu_megatron_trainer' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=64 \
    data.max_prompt_length=3072 \
    data.max_response_length=2048 \
    data.shuffle=true \
    data.seed=42 \
    data.val_max_samples=-1 \
    data.validation_shuffle=false \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=64 \
    data.truncation=error \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=QwenOmniRLHFDataset \
    ++data.mm_processor_kwargs.sampling_rate=16000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=1024 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=3100 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6144 \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen3_omni_thinker_only_npu.yaml" \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6144 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path=/opt/huawei/dataset/hfy_data/package_install/verl-omni/verl_omni/utils/reward_score/choice_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=true \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qwen3_omni_avqa \
    trainer.experiment_name=gspo_avqa_npu \
    trainer.n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@" \
    2>&1 | tee run_qwen3omni_npu_avqa_v1.log