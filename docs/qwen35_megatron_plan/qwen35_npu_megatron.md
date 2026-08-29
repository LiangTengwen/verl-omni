## 任务说明

本任务的目标是：在 verl-omni 中完成 **NPU + Qwen3.5-35B-A3B + Megatron** 训练后端的适配开发，使该模型能够在昇腾 NPU 上以 Megatron 后端完成**多模态**训练。

### 背景与上游
- verl-omni 的上游框架为 verl-0.9.0，本任务在其提供的训练框架（Ray 编排、PPO 训练循环、Engine 注册机制等）之上，扩展 Megatron 后端能力。
- NPU 不支持 NVIDIA 原生 Megatron-Core，需评估并借助 MindSpeed / MindSpeed-Bridge 完成适配（如 `vanilla_mbridge=False` 的 Megatron-Bridge，以支持 HuggingFace 格式权重转换）。

### 参考文档
[code_walkthrough_verl_omni_qwen35.md](code_walkthrough_verl_omni_qwen35.md) 已完整走读 Qwen3.5-35B-A3B 在 NPU 上 **FSDP** 后端的训练流程。本次 Megatron 适配可复用其中的适配器注册、rollout pipeline 配置、设备补丁等既有逻辑，核心是将训练后端由 FSDP/FSDP2 替换为 Megatron（两者主要差异是视觉塔的处理方式，见第 3 点）。

### 主要工作

> 与 FSDP 走读不同：Megatron 路径复用 verl 上游的 `language_model` + `megatron` 引擎（NPU 上为 `MindspeedEngineWithLMHead`），经 Megatron-Bridge（`vanilla_mbridge=False`）提供的**多模态 provider** 构建「视觉塔 + LLM」完整模型；**不经过** verl-omni 的 `OmniFSDPEngine` / `OmniModelBase.configure_model`，verl-omni 侧的 `Qwen35MoeThinkerAdapter` 仅负责 tokenizer/processor 构建。

1. **训练后端接入与分发**：确认 `omni_megatron_trainer.yaml` 的 `model_type: language_model` + `strategy: megatron` 正确路由到 verl 上游已注册的 NPU Megatron 引擎（`MindspeedEngineWithLMHead`，注册键 `language_model`+`megatron`+`npu`），而非新增 `omni_model`+`megatron` 后端。
2. **Megatron-Bridge 权重转换（核心）**：`vanilla_mbridge=False` 走 `verl/models/mcore/bridge` 的 `AutoBridge.from_hf_pretrained` + `to_megatron_provider`，完成 HF → Megatron 权重映射（TP 分片 gather、`gate_up_proj` 融合、MoE expert 布局等）；需确认/补齐 Megatron-Bridge 对 `Qwen3_5MoeForConditionalGeneration`（MoE + linear/full attention 混合层）**多模态 provider（视觉塔 + LLM）**的支持。
3. **视觉塔处理（冻结而非剥离）**：本任务为**多模态**训练，视觉塔保留在 Megatron 图中作为编码器，但参数被冻结（`freeze_vision_tower=True`），前向仍接收 `multi_modal_inputs`（`pixel_values`、`image_grid_thw`）。与 FSDP 的差异仅在于「视觉塔是否训练」：FSDP 下 `get_strip_modules()` 返回空、视觉塔参与训练；Megatron 下视觉塔冻结、仅训练语言部分。
4. **NPU 算子适配（MindSpeed）**：`MindspeedEngineWithLMHead._init_device_mesh` 调用 `mindspeed.megatron_adaptor.repatch()` 将 CUDA 算子替换为 CANN 算子；如需要 CANN 优化 layer spec，通过 `override_mcore_model_config` / `mcore_kwargs.spec` 指定。
5. **rollout 与权重同步**：复用 `Qwen35MoeRolloutAdapter`（`pipeline_name=qwen3_5_moe`，NPU 上 `weight_nz_mode=0`）+ vLLM-Omni 异步 rollout。同步场景权重同步走 `naive` checkpoint backend（同步 trainer 强制 `checkpoint_engine.backend=naive`）：Megatron 引擎经 `get_per_tensor_param()` 导出 HF 格式权重、vLLM-Omni rollout 就地更新，**不经过** `nccl` checkpoint engine（NCCL 仅用于 separate-async 的 actor/rollout 分离式部署）。
6. **启动脚本与参数**：完善 Megatron NPU 启动脚本（如 `run_qwen35_moe_thinker_gspo_megatron_npu.sh`），设置 `use_mbridge`、`vanilla_mbridge`、并行度（TP/PP/EP/CP）、offload（param/grad/optimizer）、`pipeline_name` 等参数。