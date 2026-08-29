#!/usr/bin/env bash
# Qwen3.5-35B-A3B (MoE) Thinker-only GSPO full-parameter training, omni V1 trainer.
# Modalities: text + image (+ video) -> text.
# Hardware: Ascend NPU (910B/910C, 64GB)。卡数由 NUM_NPUS 决定，默认自动探测。
#
# Prerequisite (verified separately, see mdfiles/§3.2): the installed vLLM-Omni
# must carry the Qwen3.5-MoE pipeline, i.e.
#   vllm_omni/model_executor/models/qwen3_5_moe/pipeline.py
#   "qwen3_5_moe_thinker_only" registered in vllm_omni/config/pipeline_registry.py
#
# -----------------------------------------------------------------------------
# 本轮修订依据：logs/1.txt、logs/2.txt 两次实跑，以及远端已跑通的参考脚本
#   MB_NPU/utils/run_qwen35_35b_grpo_910b_main_omni_remote.sh（下称「参考脚本」）。
# 参考脚本用 vllm_ar/vllm-ascend，本脚本用 vllm_omni —— rollout 后端**不改**，
# 只移植与后端无关的显存/精度/环境/编排结论。
#
#   1. gpu_memory_utilization 0.3 -> 0.85    修 logs/2.txt 的 KV cache ValueError（§4.1）
#   2. actor model_dtype bfloat16 -> fp32    修「能跑但不学」的静默 bug（§4.2）
#   3. NUM_NPUS 自动探测 + 前置校验          修 logs/1.txt 的 aclInit 107001（§4.3）
#   4. 补 Ray/HCCL 环境变量与超时            参考脚本 §7，多机与大模型加载必备
#   5. unset PYTORCH_NPU_ALLOC_CONF          对齐参考脚本，高显存占用下减少碎片
#   6. trainer.resume_mode=disable           默认 auto 会静默续训，基线对比必须显式关掉
# -----------------------------------------------------------------------------
set -x

# =============================================================================
# §1. Ascend / vLLM 运行时
# =============================================================================
export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0            # 配合 additional_config.weight_nz_mode=0
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export USE_OPTIMIZED_MODEL=0
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export NPU_GRAPH_CAPTURE_DISABLE=1
export ASCEND_SECURITY_CHECK_DISABLED=1
export WANDB_MODE=${WANDB_MODE:-offline}

# verl DataProto 序列化：参考脚本已验证值，numpy 路径在 NPU 上更稳
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy

# NPU 运行时调优（取自参考脚本 §3，均为已验证值）
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1

# PYTORCH_NPU_ALLOC_CONF：**故意 unset**，与参考脚本一致。
# 原值 max_split_size_mb:128 会把分配器切得很碎，而 35B MoE 的专家权重是大块张量；
# 在 gpu_memory_utilization=0.85 这种高占用下，小 split 反而更容易触发碎片型 OOM。
# 若确认需要限制 split，再显式 export，不要保留旧的 128。
unset PYTORCH_NPU_ALLOC_CONF

# =============================================================================
# §2. HCCL / 通信超时（多机必备，取自参考脚本已验证值）
# =============================================================================
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV

# 网卡名：多机 HCCL/GLOO 通信用，不可为空否则可能选错网卡
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME}

# =============================================================================
# §3. Ray 运行时（取自参考脚本 §7）
# =============================================================================
# RAY_DEDUP_LOGS=0：logs/1.txt 与 logs/2.txt 都被 Ray 折叠成
# "[repeated 15x across cluster]"，导致无法区分是哪几个 rank 出错 —— 定位 107001
# 时正是靠 RankID 才认出「只有 8..15 失败」。排障期间必须关掉去重。
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# 35B MoE 权重加载慢，默认超时不够，会表现为 worker 注册失败/GCS 超时
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0

# 让 verl/vLLM 自己管设备可见性，不要由 Ray 改写 ASCEND_RT_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
source "${ASCEND_HOME_PATH}/set_env.sh"
source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"

# Ascend 的 env 脚本不是 -e clean 的，放在它们之后
set -eo pipefail

VERL_OMNI_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

# =============================================================================
# §4. 路径与规模参数
# =============================================================================
MODEL_PATH=${MODEL_PATH:-"/opt/huawei/dataset/downloaded_models/Qwen3.5-35B-A3B"}
TRAIN_FILE=${TRAIN_FILE:-"/opt/huawei/dataset/ccb_data/data/omni/geo3k/train.parquet"}
VAL_FILE=${VAL_FILE:-"/opt/huawei/dataset/ccb_data/data/omni/geo3k/test.parquet"}

ROLLOUT_TP=${ROLLOUT_TP:-4}

# ---- §4.3 卡数：logs/1.txt 的 aclInit 107001 就死在这里 ----
# 旧默认 NUM_NPUS=16 跑在 8 卡节点上 -> Ray 照样调度 16 个 worker（--resources 里的
# NPU 只是个计数器，Ray **不校验物理卡是否存在**）-> rank 8..15 拿到不存在的
# device id -> aclInit error 107001 "Invalid device ID"。
# 这里改成默认按 /dev/davinci* 探测，并在启动前把三条硬约束全部校验掉。
_DETECTED_NPUS=$(ls -d /dev/davinci[0-9]* 2>/dev/null | wc -l | tr -d ' ')
NUM_NPUS=${NUM_NPUS:-${_DETECTED_NPUS:-8}}
NNODES=${NNODES:-${VC_WORKER_NUM:-${MA_NUM_HOSTS:-1}}}

if [ "${NUM_NPUS}" -le 0 ] 2>/dev/null; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} is not a positive integer."
    echo "       /dev/davinci* probe found '${_DETECTED_NPUS}'. Set NUM_NPUS explicitly."
    exit 1
fi
if [ "${_DETECTED_NPUS}" -gt 0 ] && [ "${NUM_NPUS}" -gt "${_DETECTED_NPUS}" ]; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} exceeds ${_DETECTED_NPUS} devices visible on this node."
    echo "       This is exactly the logs/1.txt failure (aclInit 107001, ranks ${_DETECTED_NPUS}..$((NUM_NPUS-1)))."
    echo "       Use NUM_NPUS=${_DETECTED_NPUS} on a single node, or raise NNODES for multi-node."
    exit 1
fi
if [ $((NUM_NPUS % ROLLOUT_TP)) -ne 0 ]; then
    echo "FATAL: NUM_NPUS=${NUM_NPUS} is not divisible by ROLLOUT_TP=${ROLLOUT_TP};"
    echo "       rollout.agent.num_workers (replica count) would be wrong."
    exit 1
fi
ROLLOUT_NUM_WORKERS=$((NUM_NPUS / ROLLOUT_TP))
echo "Cluster: NUM_NPUS=${NUM_NPUS} (detected ${_DETECTED_NPUS}) NNODES=${NNODES}" \
     "ROLLOUT_TP=${ROLLOUT_TP} replicas=${ROLLOUT_NUM_WORKERS} IF=${SOCKET_IFNAME}"

# Ray 侧资源账本校验：脚本连的是已存在的集群（logs/1.txt 是
# "Connecting to existing Ray cluster"），若集群是按 16 卡起的，光改 NUM_NPUS 没用。
if command -v ray >/dev/null 2>&1; then
    _ray_npu=$(ray status 2>/dev/null | grep -oE '[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ NPU' | head -n1 | sed 's#.*/##; s# NPU##' || true)
    if [ -n "${_ray_npu:-}" ]; then
        _ray_npu_int=$(printf '%.0f' "${_ray_npu}")
        _want=$((NUM_NPUS * NNODES))
        echo "Ray cluster reports ${_ray_npu_int} NPU, this job wants ${_want}."
        if [ "${_ray_npu_int}" -lt "${_want}" ]; then
            echo "FATAL: Ray cluster only has ${_ray_npu_int} NPU but the job needs ${_want}."
            echo "       Restart Ray with --resources='{\"NPU\": ${NUM_NPUS}}' per node."
            exit 1
        fi
    fi
fi

# ---- 数据文件防护 ----
# verl 的 copy_to_local 对空字符串会抛 IndexError: string index out of range
# （fs.py:238 的 src[-1] 越界），堆栈很深不好读，这里提前拦掉。
for _v in MODEL_PATH TRAIN_FILE VAL_FILE; do
    _p="${!_v}"
    if [ -z "${_p}" ]; then
        echo "FATAL: ${_v} is empty. Set it explicitly."
        exit 1
    fi
    if [ ! -e "${_p}" ]; then
        echo "FATAL: ${_v} does not exist: ${_p}"
        exit 1
    fi
done

# =============================================================================
# §4.1 显存预算 —— logs/2.txt 的直接死因
# =============================================================================
# gpu_memory_utilization 是 **vLLM 引擎可用显存的总占比，模型权重算在里面**，
# 不是「扣掉权重之后留给 KV cache 的那一份」。
#
# 实测数（logs/2.txt:1324-1325，单卡 total = 61.27 GB）：
#   权重 35.11B x 2B(bf16) / TP=4       = 17.56 GB/卡    <- 硬地板
#   旧值 0.30 -> 引擎预算 0.30 x 61.27  = 18.38 GB
#   剩余     18.38 - 17.56              =  0.82 GB       -> 扣掉 activation 只剩 0.04 GiB
#   而单条 5200-token 请求就需要           0.09 GiB       -> ValueError，4 个 replica 全挂
#
# 0.85 是参考脚本在**同模型 / 同 TP=4 / 同 910B 64GB** 上跑通的值（参考脚本:219）。
#
# 若 OOM，往【下】退：0.85 -> 0.80 -> 0.70 -> 0.60。
# 地板是 17.56/61.27 ≈ 0.29，低于约 0.35 连权重都放不下。
# 注意：方案文档 §5.4 的「gpu_memory_utilization 0.3->0.25」阶梯方向是**反的**，
#       对本报错照做只会死得更早（连权重都加载不了）。
# 真要省显存，正确的杠杆是 ROLLOUT_TP 4->8（权重降到 8.78 GB/卡）。
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}

# =============================================================================
# §4.2 精度开关 —— 改错会「能跑但不学」，是最隐蔽的一个
# =============================================================================
# verl/verl/workers/engine/fsdp/transformer_impl.py:239-245
#     torch_dtype = self.engine_config.model_dtype
#     if torch_dtype is None:
#         # if it is training, we force torch_dtype to fp32
#         torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16
#
# 显式传 bfloat16 会**顶掉**这段 fp32 强制，等于拿掉 optimizer 的 fp32 master weight。
# bf16 只有 8 bit 尾数（相对精度 ~2^-8 ≈ 0.4%），而 lr=1e-6 的单步更新量远小于该量级，
# 绝大部分更新会被直接舍入掉 —— 训练照跑、loss 正常、reward 就是不涨。
#
# 前向/反向仍是 bf16：由 MixedPrecisionPolicy(param_dtype=bfloat16) 单独负责
# （同文件 :378 与 :436），与本开关无关，改 fp32 **不会**把计算退成 fp32。
#
# 参考脚本实测（同模型）：bfloat16 的 reward 在 0.25~0.45 震荡不涨；
# fp32 从 0.28 涨到 0.83（step 101）。
#
# fp32 master weight 会增加 host 内存（optimizer state 在 CPU 上），
# 若 host 侧吃紧，退让顺序见 §4.4 PPO_MICRO_BATCH_SIZE_PER_GPU 处的说明，
# **不要**拿这个开关换内存 —— 换来的是「能跑但不学」。
ACTOR_MODEL_DTYPE=${ACTOR_MODEL_DTYPE:-fp32}

# ref 是 forward_only、不持 optimizer state，且本配置 use_kl_loss=false
# （ref logprob 不进 loss），bf16 在数值上无害且省显存，故保持 bfloat16。
REF_MODEL_DTYPE=${REF_MODEL_DTYPE:-bfloat16}

# =============================================================================
# §4.4 训练超参
# =============================================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}

# PPO_MINI_BATCH_SIZE：参考脚本用 64（= TRAIN_BATCH_SIZE，即完全 on-policy，
# 每个 rollout batch 只做 1 次 optimizer step）。本脚本用 4，也就是每批做 16 次更新。
# GSPO 的 clip_ratio 只有 3e-4/4e-4，off-policy 漂移会被 clip 大量截断，
# 若观察到 clip fraction 接近 1，理论上该调大这个值 ——
# **但在 16 卡上调不动**：它会同比放大每卡序列数，而下面 PPO_MICRO_BATCH_SIZE_PER_GPU
# 的注释解释了为什么这一版 verl 在每卡 >4 条时两头都是 OOM。真要调，先加卡。
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}

ROLLOUT_N=${ROLLOUT_N:-8}

# -----------------------------------------------------------------------------
# PPO_MICRO_BATCH_SIZE_PER_GPU —— 必须让「每卡 micro-batch 数 == 1」，否则必 OOM
# -----------------------------------------------------------------------------
# 这不是调优参数，是这一版 verl 的硬约束。原因在
# verl/workers/engine/fsdp/transformer_impl.py:672-698 `_gradient_sync_context`：
# 除最后一个 micro-batch 外，它对 FSDP2 调 set_requires_gradient_sync(False)，
# 其 docstring 自己承认代价是 "temporarily retaining unsharded gradients until
# the final backward"。落到 torch 里是 _fsdp_param_group.py:389-394 —— reshard 之后
# 直接 return，**不做 reduce-scatter**；再经 _fsdp_param.py:639-645
# to_accumulated_grad_if_needed()，因为 reduce_dtype 被硬编码成 fp32
# （transformer_impl.py:379）而 grad 是 bf16，dtype 不等 => 升位到 fp32 并保留。
#
# 于是每多一个 micro-batch，就要在卡上囤住**全部已反传层的未分片 fp32 梯度**：
#     单层 expert 参数 805,306,368 × 4 B = 3.0 GiB
#     × 40 层                            ≈ 120 GiB      <- 单卡只有 61.27 GiB
#     logs/3.txt 死在 57.98 GiB          ≈ 第 19 层
# 注意 CPUOffloadPolicy 救不了：它只搬 reduce-scatter **之后**的分片梯度，
# 这条路径压根没走到 reduce-scatter。
#
# micro-batch 数 = 每卡 mini-batch 序列数 / PPO_MICRO_BATCH_SIZE_PER_GPU，其中
#     每卡 mini-batch 序列数 = PPO_MINI_BATCH_SIZE × rollout.n / world_size
# rollout.n 那一乘是 trainer 内部做的，见 verl/trainer/ppo/v1/trainer_base.py:1674-1675，
# 脚本里看不到，logs/3.txt 就是栽在这里：4 × 8 / 16 = 2 条 => 2 个 micro-batch。
_WORLD_SIZE=$(( NUM_NPUS * NNODES ))
_SEQS_PER_RANK=$(( PPO_MINI_BATCH_SIZE * ROLLOUT_N / _WORLD_SIZE ))
if [ "${_SEQS_PER_RANK}" -lt 1 ]; then
    echo "FATAL: PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE} × rollout.n=${ROLLOUT_N}" \
         "= $(( PPO_MINI_BATCH_SIZE * ROLLOUT_N )) < world_size=${_WORLD_SIZE};" >&2
    echo "       每卡分不到一条序列。请调大 PPO_MINI_BATCH_SIZE。" >&2
    exit 1
fi
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-${_SEQS_PER_RANK}}
_NUM_MICRO_BATCHES=$(( _SEQS_PER_RANK / PPO_MICRO_BATCH_SIZE_PER_GPU ))
echo "Actor batching: ${_SEQS_PER_RANK} seq/rank / micro_bsz=${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
     "=> ${_NUM_MICRO_BATCHES} micro-batch(es)"
if [ "${_NUM_MICRO_BATCHES}" -gt 1 ]; then
    echo "WARNING: micro-batch 数 > 1，会触发未分片 fp32 梯度囤积（约 3.0 GiB/层 × 40 层）。" >&2
    echo "         除非你清楚自己在做什么，否则 OOM 几乎是必然的。" >&2
fi
if [ "${_SEQS_PER_RANK}" -gt 4 ]; then
    echo "WARNING: 每卡 ${_SEQS_PER_RANK} 条序列。这一版 verl 在 ${_WORLD_SIZE} 卡下**没有安全解**：" >&2
    echo "         micro_bsz 设小 -> 多 micro-batch -> 未分片梯度囤积 OOM；" >&2
    echo "         micro_bsz 设大 -> 单次前反传激活/logits 撑爆 -> 同样 OOM。" >&2
    echo "         唯一出路是加卡（抬 NNODES）或减小 PPO_MINI_BATCH_SIZE / ROLLOUT_N。" >&2
    echo "         注意参考脚本 run_qwen35_35b_grpo_910b_main_omni_remote.sh 的 mini=64" \
         "不能照搬：它跑在 64 卡且是另一版 verl-omni。" >&2
fi
# 如果 ${_SEQS_PER_RANK} 条序列的激活放不下，正确的退让顺序是：
#   1) 缩序列长度（见下面 §4.5）—— 首选，因为 logits 栈跟长度是线性的；
#   2) PPO_MINI_BATCH_SIZE 减半（4->2），每卡降到 1 条 —— 代价是 optimizer step 数翻倍，
#      GSPO 的 clip 3e-4/4e-4 对 off-policy 漂移很敏感，改完盯 clip fraction；
#   3) 最后才是动 micro_bsz —— 因为动它就会重新引入上面那条囤积路径。

# =============================================================================
# §4.5 序列长度 —— 单次前反传显存的主导项
# =============================================================================
# 本配置 use_remove_padding=False（原因见启动命令上方的说明），所以一条序列在
# 前反传里**按 padding 后的满长度算**：
#
#     L = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH
#
# 吃显存最狠的是全词表 logits，vocab = 248144，比参考脚本 Qwen3-Omni 的 ~152k 大 1.6 倍：
#
#     单份 logits = seq/卡 × L × 248144 × 2 B (bf16)
#       L=5120（改前）: 2 × 5120 × 248144 × 2 = 4.73 GiB   ×3 份 ≈ 14.2 GiB
#       L=3072（改后）: 2 × 3072 × 248144 × 2 = 2.84 GiB   ×3 份 ≈  8.5 GiB
#     （3 份 ≈ logits + log_softmax + 反传梯度；entropy_coeff=0 且 calculate_entropy
#       解析为 False，所以 entropy 那份不产生 —— 已在 logs/3.txt:46 确认。）
#     rollout 侧 KV cache 需求同步降到 0.60×。
#
# 对比参考脚本 examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v2.sh
# （同 16 卡、同 mini=4、同 n=8、同样 ppo_micro_batch_size_per_gpu=1，能跑）：
# 它靠 use_dynamic_bsz=true + use_remove_padding=true，把每卡 2 条**真实 token**
# 打包进一个 ≤6144 token 的 micro-batch；长度写的虽然也是 3072/2048，但 AVQA 的实际
# 序列远短于此，所以从没触发过 §4.4 那条囤积路径。本脚本两个开关都是 False
# （eager attention 不能配 rmpad），只能靠**真的把长度改小**拿到同样的效果。
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}      # <- 原 3072
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}  # <- 原 2048
_SEQ_LEN=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))

# rollout 侧跟着一起缩，否则 KV cache 仍按 5200 预留、白占显存
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$(( _SEQ_LEN + 128 ))}   # <- 原 5200
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-$(( MAX_PROMPT_LENGTH + 32 ))}  # <- 原 3100

# token 预算 = 每卡序列数 × L。当前 use_dynamic_bsz=False，这个值不生效（见启动命令
# 上方说明），但保持它跟长度一致：万一哪天把 dynamic_bsz 打开，这个值恰好等于
# 「每卡全部序列装进 1 个 micro-batch」，不会退回多 micro-batch 的囤积路径。
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-$(( _SEQ_LEN * _SEQS_PER_RANK ))}

echo "Seq len: prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} L=${_SEQ_LEN}" \
     "| rollout max_model_len=${ROLLOUT_MAX_MODEL_LEN} | token budget/gpu=${MAX_TOKEN_LEN_PER_GPU}"

# ⚠️ 调小 MAX_PROMPT_LENGTH 有一个静默副作用：data.filter_overlong_prompts=true
#    会把 prompt 超长的样本**直接丢掉**（不是截断，truncation=error）。
#    3072 -> 2048 意味着数据集可能变小。跑起来后在日志里搜 "filter" / 数据集条数，
#    确认丢掉的比例可接受；若丢太多，优先把 MAX_RESPONSE_LENGTH 再压一档，
#    而不是把 prompt 加回去。

RESUME_MODE=${RESUME_MODE:-disable}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}

PROJECT_NAME=${PROJECT_NAME:-qwen35_moe_gspo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gspo_qwen35_moe_npu}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${VERL_OMNI_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# _TensorboardAdapter 默认写相对路径 tensorboard_log/<proj>/<exp>（tracking.py:299），
# cwd 一变或容器一销毁就找不到，显式指向持久化目录。
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${VERL_OMNI_ROOT}/tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${TENSORBOARD_DIR}" "${DEFAULT_LOCAL_DIR}"

# =============================================================================
# §5. 启动
# =============================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 严禁在下面这段反斜杠续行里用 `#` 注释掉某一行参数。
#
# bash 在词法阶段就把 `#` 当注释开始，一直吃到行尾 —— 包括那一行末尾的续行符 `\`。
# 续行链一断，整条命令就在那里结束，**该行之后的所有参数以及末尾的管道全部丢失**，
# 剩下的行还会被当成独立命令执行。参考脚本 logs/omni8.txt 就是这么炸的：注释掉
# reward.reward_manager 两行后，trainer.logger 回退成默认的 ["console","wandb"]，
# 报出一个跟 reward manager 毫无关系的 wandb API key 错误。
#
# 要临时去掉某个参数：直接删行。
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# 下面这条命令里有两个**看着生效、实际被吞掉**的参数，别按字面理解：
#
# 1) actor.fsdp_config.param_offload / optimizer_offload
#    只要 offload_policy=true（或 forward_only，ref worker 就是），
#    transformer_impl.py:440-444 会无条件把 _is_offload_param、_is_offload_optimizer
#    强制改回 False，改用 FSDP2 原生的 CPUOffloadPolicy(pin_memory=True)。
#    两行留着只是为了「万一把 offload_policy 关掉时还有兜底」，当前配置下是死参数。
#    重要推论：CPUOffloadPolicy 只搬 reduce-scatter **之后**的分片梯度，
#    因此它对 §4.4 说的那条未分片梯度囤积路径完全无效 —— 别指望它救 OOM。
#
# 2) actor.ppo_max_token_len_per_gpu
#    仅在 use_dynamic_bsz=True 时用于切分；这里是 False，该值不起作用。
#    真正决定单次前反传规模的是 ppo_micro_batch_size_per_gpu。
python3 -m verl_omni.trainer.main_omni \
    trainer.device=npu \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.image_key=images \
    data.shuffle=true \
    data.seed=42 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=64 \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.trust_remote_code=False \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    \
    actor_rollout_ref.actor.trainer_type=policy_gradient \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
    actor_rollout_ref.actor.entropy_from_logits_chunk_size=1024 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${ACTOR_MODEL_DTYPE} \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','extra']" \
    \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=${REF_MODEL_DTYPE} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
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
    trainer.v1.trainer_mode=omni_sync \
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
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@" \
    2>&1 | tee run_qwen35_moe_gspo_npu1.log