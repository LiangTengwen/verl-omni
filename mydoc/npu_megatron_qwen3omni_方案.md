# NPU + Megatron + Qwen3-Omni 训练方案

> 日期：2026-08-22
> 目标读者：对 verl-omni 和 Megatron 不太熟悉的开发者
> 阅读前提：了解基本的 RLHF 训练概念即可

---

## 目录

1. [背景：为什么需要这个方案](#1-背景为什么需要这个方案)
2. [核心概念通俗讲解](#2-核心概念通俗讲解)
3. [PR #399 是怎么在 GPU 上跑通的](#3-pr-399-是怎么在-gpu-上跑通的)
4. [为什么 NPU 不能直接复用 PR #399](#4-为什么-npu-不能直接复用-pr-399)
5. [我们的解决方案：OmniMindSpeedMegatronEngine](#5-我们的解决方案omnimindspeedmegatronengine)
6. [代码实现详解](#6-代码实现详解)
7. [文件清单与修改说明](#7-文件清单与修改说明)
8. [常见问题与调试指南](#8-常见问题与调试指南)
9. [附录：相关术语表](#9-附录相关术语表)

---

## 1. 背景：为什么需要这个方案

### 1.1 verl-omni 已有的训练路径矩阵

在 verl-omni 中，Qwen3-Omni 模型在不同平台和训练后端上的支持情况如下：

| 模态 | 平台 | 训练后端 | Rollout 模式 | 状态 |
|------|------|---------|-------------|------|
| Image→Text | GPU | Megatron 全量 | 全异步 | ✅ 已打通 (PR #399) |
| Image→Text | NPU | FSDP 全参 | 同步 | ✅ 已打通 |
| **Image→Text** | **NPU** | **Megatron 全量** | **异步** | **❌ 未实现 ← 我们要做这个** |
| Audio→Text | NPU | Megatron 全量 | 异步 | ❌ 未实现 |

### 1.2 为什么要做这个方案

- **NPU 上训练大模型**：如果要在华为昇腾 NPU 上训练 Qwen3-Omni，FSDP 有其局限性（显存利用率低、通信开销大），Megatron 的 TP+PP 并行能显著提升效率
- **Megatron 的优势**：Tensor Parallel + Pipeline Parallel 可以把大模型切分到多个 NPU 上，支持更大的模型和更长的序列
- **PR #399 已有 GPU 方案**：PR #399 已经在 GPU 上实现了 Megatron 训练，我们希望把同样的能力迁移到 NPU

---

## 2. 核心概念通俗讲解

### 2.1 什么是 Megatron？

Megatron 是 NVIDIA 开发的一种**模型并行训练框架**。它把一个大模型切成多块，分散到多个 GPU 上训练。

想象一下：你有一个超大蛋糕（大模型），一个人吃不完（单卡显存不够）。Megatron 帮你把蛋糕切成几块（Tensor Parallel），每块由不同的人拿着，大家一起吃。

```
单卡训练：           Megatron TP=4：
┌─────────┐         ┌──┬──┬──┬──┐
│ 整个模型 │         │片1│片2│片3│片4│
│ 放一张卡 │         │卡1│卡2│卡3│卡4│
└─────────┘         └──┴──┴──┴──┘
```

### 2.2 什么是 MindSpeed？

MindSpeed 是华为昇腾 NPU 上的 Megatron 兼容库。它做了两件事：

1. **接口兼容**：提供和 NVIDIA Megatron-Core 一样的 API，上层代码不用改
2. **算子替换**：把 CUDA 算子换成 CANN 算子（NPU 的运算库）

```
GPU 上：  Megatron-Core (CUDA 算子) → NVIDIA GPU
NPU 上：  MindSpeed (CANN 算子)     → 昇腾 NPU
```

### 2.3 什么是 mbridge？

mbridge（Megatron Bridge）是 verl 中用来在 **HF 格式**（HuggingFace 格式）和 **Megatron 格式**之间转换的桥梁。

```
HF 格式（人类可读）：        Megatron 格式（训练用）：
model.layers.0.self_attn    decoder.layers.0.self_attention
    .qkv_proj.weight             .linear_qkv.weight

mbridge 负责：
从 HF checkpoint → 转换成 Megatron 模型（加载）
从 Megatron 模型 → 转换回 HF 格式（导出，给 Rollout 用）
```

verl 有两个版本的 mbridge：
- **老版 mbridge**（`vanilla_mbridge=True`）：支持 NPU，但比较死板，只能识别标准架构
- **新版 mbridge**（`vanilla_mbridge=False`）：更智能，能自动识别模型架构，但 NPU 不支持

### 2.4 Qwen3-Omni 的特殊结构

Qwen3-Omni 是一个**多模态模型**，包含多个子模块：

```
Qwen3-Omni（完整模型）：
├── thinker（语言模型，类似 Qwen3-MoE）
│   ├── embed_tokens
│   ├── layers（28 层 Transformer）
│   │   ├── self_attn
│   │   └── mlp（MoE 专家混合）
│   ├── norm
│   └── lm_head
├── talker（语音生成）
├── visual（图像编码器）
├── audio_tower（音频编码器）
└── code2wav（语音解码器）
```

**RL 训练只训练 thinker 部分**，其他部分在推理时使用但不训练。

### 2.5 类比：训练系统就像一家餐厅

为了让你更直观地理解整个系统，我们用餐厅来类比：

```
训练系统 = 餐厅
─────────────────────────────────────────────────
菜单（config.json）         = 告诉厨师做什么菜
厨师（Megatron Engine）     = 实际做菜的人
菜谱（mbridge）             = 把食材（HF 格式）变成
                             半成品（Megatron 格式）
炉灶（CUDA / CANN）         = 实际烹饪的硬件
外卖员（Rollout Engine）    = 把菜送到顾客手中
```

我们的目标：**在 NPU 厨房（CANN）上，让厨师（MindSpeed）做出 thinker 这道菜。**

---

## 3. PR #399 是怎么在 GPU 上跑通的

### 3.1 核心思路：伪装成 Qwen3-MoE

PR #399 用了一个非常巧妙的**伪装技巧**。

在 verl 的配置中，`model_type` 字段决定了用哪个引擎类来训练：

```yaml
model_type: language_model  # ← 关键
```

`language_model` 会触发上游 verl 的 `MegatronEngineWithLMHead` 引擎。这个引擎只会做一件事：**把模型当作普通语言模型来训练**。

但问题来了：Qwen3-Omni 不是普通语言模型，它的 `config.json` 长这样：

```json
{
  "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
  "model_type": "qwen3_omni_moe",
  "thinker_config": {
    "num_hidden_layers": 28,
    "hidden_size": 2560,
    ...
  }
}
```

老版 mbridge 看到 `Qwen3OmniMoeForConditionalGeneration` 会直接报错，因为它不认识这个架构。

**PR #399 的解决方案**：使用**新版 mbridge**（`vanilla_mbridge=False`），新版 mbridge 会自动识别 `config.json` 中的 `thinker_config`，从中提取 Qwen3-MoE 所需的架构参数。

### 3.2 数据流图解

```
PR #399 的完整数据流：

1. 配置加载
   model_type: language_model
   → 触发 MegatronEngineWithLMHead（上游 verl 的引擎）
   
2. 创建 mbridge
   vanilla_mbridge: false
   → 使用新版 mbridge
   → AutoBridge 自动识别 Qwen3OmniMoeForConditionalGeneration
   → 从 thinker_config 提取架构参数
   → 创建 Megatron 配置

3. 构建模型
   GPTModel(...)  # 标准的 Megatron GPT 模型
   → 其实就是 thinker 部分

4. 加载权重
   bridge.load_weights(module, checkpoint_path)
   → 加载所有权重（包括 thinker. 前缀的）
   → 新版 mbridge 自动处理前缀映射

5. 训练循环
   forward → loss → backward → optimizer.step
   → 只训练 thinker 参数

6. 导出权重（给 Rollout 用）
   bridge.export_hf_weights(module)
   → 输出 HF 格式权重
   → 格式：model.layers.0.self_attn.qkv_proj.weight
   → 通过 NCCL 广播给 vLLM-Omni
```

### 3.3 为什么 PR #399 能跑通？

一句话总结：**新版 mbridge 足够智能，能自动识别 Qwen3-Omni 的 thinker 子配置，不需要 verl 告诉它怎么处理。**

---

## 4. 为什么 NPU 不能直接复用 PR #399

### 4.1 核心差异

| 环节 | PR #399 (GPU) | NPU 目标 |
|------|--------------|---------|
| **mbridge 版本** | 新版 (`vanilla_mbridge=False`) | 老版 (`vanilla_mbridge=True`) |
| **底层算子** | CUDA | CANN |
| **Rollout 引擎** | vLLM-Omni | vLLM-Omni (含 Ascend 后端) |
| **引擎类** | 上游 `MegatronEngineWithLMHead` | 需要自定义 |

### 4.2 为什么 NPU 只能用老版 mbridge？

新版 mbridge 依赖 NVIDIA 的 Megatron-Core 的某些特性，这些特性在 NPU 的 MindSpeed 上没有实现。所以 NPU 只能使用老版 mbridge。

### 4.3 老版 mbridge 有什么限制？

老版 mbridge 通过 `AutoBridge.from_config()` 创建，它在内部检查 `hf_config.architectures`：

```python
# 老版 mbridge 内部逻辑（简化）
if hf_config.architectures[0] == "Qwen3MoeForCausalLM":
    # 用 Qwen3-MoE 的转换器
    converter = Qwen3MoeConverter()
elif hf_config.architectures[0] == "LlamaForCausalLM":
    converter = LlamaConverter()
else:
    raise ValueError(f"Unknown architecture: {hf_config.architectures[0]}")
```

Qwen3-Omni 的 `architectures` 是 `["Qwen3OmniMoeForConditionalGeneration"]`，老版 mbridge 不认识，直接报错。

### 4.4 类比

```
PR #399（GPU）：
你把一个外国人（Qwen3-Omni）带到酒店前台（mbridge）。
前台用的是智能翻译机（新版 mbridge），
自动识别外国人的护照，顺利办理入住。

NPU 目标：
同一个外国人来到同一个酒店，
但前台今天只有老式翻译机（老版 mbridge），
老式翻译机不认识这本护照，直接拒绝。

我们要做的：
给外国人办一张假身份证（伪装成 Qwen3-MoE），
老式翻译机一看，哦，自己人，放行。
```

---

## 5. 我们的解决方案：OmniMindSpeedMegatronEngine

### 5.1 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│ 启动脚本                                                          │
│ model_type: omni_model                                            │
│ strategy: mindspeed_megatron                                      │
│ device: npu                                                       │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ EngineRegistry 查找引擎                                            │
│ → 匹配 (omni_model, mindspeed_megatron, npu)                      │
│ → 找到 OmniMindSpeedMegatronEngine                                │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│ OmniMindSpeedMegatronEngine 初始化                                 │
│                                                                  │
│  第1步：_build_tf_config()                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. 保存原始 hf_config（完整 Omni 配置）                      │ │
│  │ 2. 提取 thinker_config                                        │ │
│  │ 3. 设置 architectures = ["Qwen3MoeForCausalLM"]              │ │
│  │ 4. 临时替换 hf_config = thinker_config                        │ │
│  │ 5. 调用父类创建 mbridge（老版 mbridge 接受伪装后的配置）       │ │
│  │ 6. 恢复原始 hf_config                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  第2步：_init_device_mesh()                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. 临时替换 hf_config = thinker_config                        │ │
│  │ 2. 调用 apply_patch()（MindSpeed 替换 CUDA 算子为 CANN）     │ │
│  │ 3. 初始化 NPU 设备网格（TP=4, PP=2）                         │ │
│  │ 4. 恢复原始 hf_config                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  第3步：_build_megatron_module()                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. 遍历 checkpoint 所有 .safetensors 文件                     │ │
│  │ 2. 只保留 thinker. 开头的权重                                 │ │
│  │ 3. 去掉 thinker. 前缀                                         │ │
│  │ 4. 写入临时目录                                               │ │
│  │ 5. 复制并修改 config.json                                     │ │
│  │ 6. 设置 local_path = 临时目录                                 │ │
│  │ 7. 调用父类构建 Megatron 模型 + 加载权重                      │ │
│  │ 8. 恢复 local_path，清理临时目录                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  第4步：训练循环                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 1. Rollout 阶段：vLLM-Omni 生成样本（加载完整 Omni 模型）    │ │
│  │ 2. 训练阶段：Thinker 前向 → 算 loss → 反向 → 更新参数       │ │
│  │ 3. 权重同步：导出 thinker 权重 → NCCL → vLLM-Omni 更新     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 三个关键处理

这个引擎主要做了三件事，每一件都对应一个方法重写：

#### 关键处理 1：伪装架构（`_build_tf_config`）

**问题**：老版 mbridge 不认识 `Qwen3OmniMoeForConditionalGeneration`

**解法**：临时把 `hf_config` 换成 `thinker_config`，把架构名改成 `Qwen3MoeForCausalLM`

```python
def _build_tf_config(self):
    # 保存原始配置
    original_hf_config = self.model_config.hf_config
    
    # 提取 thinker_config（包含 thinker 的所有架构参数）
    thinker_config = original_hf_config.thinker_config
    
    # 伪装成 Qwen3-MoE
    thinker_config.architectures = ["Qwen3MoeForCausalLM"]
    self.model_config.hf_config = thinker_config
    
    try:
        super()._build_tf_config()  # mbridge 创建成功
    finally:
        # 恢复原始配置（bridge 已经创建好了，不受影响）
        self.model_config.hf_config = original_hf_config
```

**为什么 thinker_config 可以伪装成 Qwen3-MoE？**

因为 thinker 的底层结构就是 Qwen3-MoE：

```
thinker_config 包含的字段：         Qwen3-MoE 需要的字段：
num_hidden_layers: 28            ← num_hidden_layers: 28     ✓
hidden_size: 2560                ← hidden_size: 2560         ✓
num_attention_heads: 20          ← num_attention_heads: 20   ✓
num_experts: 60                  ← num_experts: 60           ✓
...                              ← ...                        ✓
```

**每一层的数学结构完全相同**（一样的 MoE、一样的 Attention、一样的 LayerNorm），所以可以安全伪装。

#### 关键处理 2：设置 MindSpeed 算子（`_init_device_mesh`）

**问题**：MindSpeed 的 `apply_patch()` 需要从 `hf_config` 读取模型架构参数，但原始 `hf_config` 是 Omni 的完整配置，这些参数在 `thinker_config` 里。

**解法**：在调用 `apply_patch()` 之前，临时把 `hf_config` 换成 `thinker_config`。

```python
def _init_device_mesh(self):
    original_hf_config = self.model_config.hf_config
    thinker_config = original_hf_config.thinker_config
    
    # 临时替换
    thinker_config.architectures = ["Qwen3MoeForCausalLM"]
    self.model_config.hf_config = thinker_config
    
    try:
        apply_patch(self.model_config, ...)  # MindSpeed 算子替换
        super()._init_device_mesh()          # 初始化设备网格
    finally:
        self.model_config.hf_config = original_hf_config
```

**`apply_patch` 做了什么？**

它把 Megatron-Core 中默认的 CUDA 算子替换为 CANN 算子：

```
原来：FlashAttentionCUDA → 替换后：FlashAttentionNPU
原来：SwiGLUCUDA        → 替换后：SwiGLUCANN
原来：RMSNormCUDA       → 替换后：RMSNormCANN
```

#### 关键处理 3：提取 Thinker 权重（`_build_megatron_module`）

**问题**：Omni checkpoint 中权重名带 `thinker.` 前缀，但老版 mbridge 期望不带前缀的权重名。

**解法**：提取 thinker 权重，去掉前缀，放到临时目录让 mbridge 加载。

```
原始 checkpoint 中的权重：
  thinker.embed_tokens.weight                    → 保留，去掉前缀
  thinker.layers.0.self_attn.qkv_proj.weight     → 保留，去掉前缀
  thinker.layers.0.mlp.gate.weight               → 保留，去掉前缀
  thinker.model.norm.weight                      → 保留，去掉前缀
  thinker.lm_head.weight                         → 保留，去掉前缀
  talker.embed_tokens.weight                     → 丢弃
  talker.layers.0.xxx                            → 丢弃
  visual.encoder.xxx                             → 丢弃
  audio_tower.xxx                                → 丢弃
  code2wav.xxx                                   → 丢弃

处理后（临时目录）：
  embed_tokens.weight                            ✓
  layers.0.self_attn.qkv_proj.weight                ✓
  layers.0.mlp.gate.weight                       ✓
  model.norm.weight                              ✓
  lm_head.weight                                 ✓
```

### 5.3 完整数据流

用一张图总结整个流程：

```
    ┌──────────────┐
    │ 启动脚本     │
    │ model_type:  │
    │ omni_model   │──────→ EngineRegistry 查找
    └──────────────┘              │
                                  │ 找到
                                  ▼
    ┌───────────────────────────────────────────────────────┐
    │ OmniMindSpeedMegatronEngine                            │
    │                                                        │
    │  ① _build_tf_config()                                   │
    │     ┌─────────────────────────────────────────────────┐│
    │     │ hf_config → thinker_config (伪装成 Qwen3-MoE)   ││
    │     │ → 创建老版 mbridge                               ││
    │     └─────────────────────────────────────────────────┘│
    │                         │                              │
    │  ② _init_device_mesh()                                 │
    │     ┌─────────────────────────────────────────────────┐│
    │     │ apply_patch() → 替换 CANN 算子                   ││
    │     │ → 初始化 NPU 设备网格                             ││
    │     └─────────────────────────────────────────────────┘│
    │                         │                              │
    │  ③ _build_megatron_module()                            │
    │     ┌─────────────────────────────────────────────────┐│
    │     │ 提取 thinker 权重 → 去掉前缀 → 临时目录        ││
    │     │ → mbridge 加载权重 → 构建 Megatron GPTModel     ││
    │     └─────────────────────────────────────────────────┘│
    │                         │                              │
    │  ④ 训练循环                                            │
    │     ┌─────────────────────────────────────────────────┐│
    │     │ forward → loss → backward → optimizer.step      ││
    │     │ (底层用 CANN 算子，不是 CUDA)                    ││
    │     └─────────────────────────────────────────────────┘│
    │                         │                              │
    │  ⑤ 权重导出                                            │
    │     ┌─────────────────────────────────────────────────┐│
    │     │ bridge.export_weights()                          ││
    │     │ → 输出 thinker 权重（HF 格式）                    ││
    │     │ → NCCL 发送给 vLLM-Omni                        ││
    │     └─────────────────────────────────────────────────┘│
    └───────────────────────────────────────────────────────┘
```

---

## 6. 代码实现详解

### 6.1 文件结构

```
verl-omni/verl_omni/workers/engine/
  ├── __init__.py                     ← 修改：添加导入
  ├── fsdp/                           ← 已有的 FSDP 引擎
  └── mindspeed/                      ← 新建：NPU 引擎
      ├── __init__.py                 ← 新建：导出
      └── omni_impl.py               ← 新建：核心代码
```

### 6.2 核心引擎代码逐段讲解

#### 6.2.1 引擎注册（第 1 段）

```python
@EngineRegistry.register(
    model_type="omni_model",          # 当 model_type=omni_model 时触发
    backend=["mindspeed_megatron"],    # 当 strategy=mindspeed_megatron 时触发
    device=["npu"]                     # 当 device=npu 时触发
)
class OmniMindSpeedMegatronEngine(MindSpeedMegatronEngineWithLMHead):
    """MindSpeed Megatron engine for Qwen3-Omni Thinker model on NPU."""
```

**类比**：这就像在酒店前台登记入住信息。当客人（训练配置）说"我是 Omni 模型，用 MindSpeed 后端，在 NPU 上跑"，前台就知道要叫这个专门的管家（引擎）来服务。

#### 6.2.2 伪装架构（`_build_tf_config`）

```python
def _build_tf_config(self):
    # 1. 保存原始配置
    original_hf_config = self.model_config.hf_config
    
    # 2. 提取 thinker_config
    #    thinker_config 包含所有 Qwen3-MoE 需要的架构参数
    #    （层数、隐藏层大小、注意力头数、专家数等）
    thinker_config = getattr(original_hf_config, "thinker_config", None)
    if thinker_config is None:
        raise ValueError("没有找到 thinker_config，这不是 Qwen3-Omni 模型")
    
    # 3. 伪装：让 thinker_config 看起来像 Qwen3-MoE
    thinker_config.architectures = ["Qwen3MoeForCausalLM"]
    self.model_config.hf_config = thinker_config
    
    try:
        # 4. 调用父类创建 mbridge
        #    父类会调用 AutoBridge.from_config(self.model_config.hf_config)
        #    此时 hf_config 已经是伪装后的 thinker_config
        #    老版 mbridge 看到 architectures=["Qwen3MoeForCausalLM"]
        #    会认为这是标准 Qwen3-MoE，顺利创建
        super()._build_tf_config()
    finally:
        # 5. 恢复原始配置
        #    mbridge 已经创建好了，它内部保存了架构信息
        #    此时恢复 hf_config 不影响 bridge 的使用
        self.model_config.hf_config = original_hf_config
```

**为什么这样做是安全的？**

```
时机：在 bridge 创建期间 → 需要伪装
状态：bridge 创建完成后 → 可以恢复

因为 bridge 在创建时已经读取了 thinker_config 的所有参数，
并保存在 bridge 内部。后续 bridge 的所有操作（load_weights、export_weights）
都使用 bridge 内部保存的参数，不再读取 hf_config。
```

#### 6.2.3 设置 MindSpeed（`_init_device_mesh`）

```python
def _init_device_mesh(self):
    # 1. 保存原始配置
    original_hf_config = self.model_config.hf_config
    thinker_config = getattr(original_hf_config, "thinker_config", None)
    
    # 2. 临时替换
    if thinker_config is not None:
        thinker_config.architectures = ["Qwen3MoeForCausalLM"]
        self.model_config.hf_config = thinker_config
    
    try:
        # 3. apply_patch() 会读取 hf_config 中的架构参数
        #    例如：num_hidden_layers、hidden_size、num_attention_heads 等
        #    这些参数在 thinker_config 中都有
        #    然后它会注册 MindSpeed 的 CANN 算子
        apply_patch(self.model_config, self.engine_config, self.optimizer_config)
        
        # 4. 初始化 NPU 设备网格
        #    根据 TP=4, PP=2 创建通信组
        MegatronEngine._init_device_mesh(self)
    finally:
        # 5. 恢复原始配置
        self.model_config.hf_config = original_hf_config
```

**MindSpeed 算子替换清单**：

| 原始算子（CUDA） | 替换后（CANN） | 作用 |
|-----------------|---------------|------|
| FlashAttention | FlashAttentionNPU | 注意力计算 |
| RMSNorm | RMSNormCANN | 层归一化 |
| SwiGLU | SwiGLUCANN | 激活函数 |
| RotaryEmbedding | RotaryEmbeddingCANN | 位置编码 |

#### 6.2.4 提取权重（`_build_megatron_module`）

```python
def _build_megatron_module(self):
    # 1. 保存原始路径
    orig_local_path = self.model_config.local_path
    
    # 2. 提取 thinker 权重到临时目录
    #    - 从 checkpoint 目录读取所有 .safetensors 文件
    #    - 只保留 thinker. 开头的权重
    #    - 去掉 thinker. 前缀
    #    - 写入临时目录
    #    - 同时复制并修改 config.json
    tmp_dir = self._prepare_thinker_checkpoint(orig_local_path)
    self.model_config.local_path = tmp_dir  # 指向临时目录
    
    try:
        # 3. 调用父类构建 Megatron 模型
        #    父类会:
        #    a. 创建 GPTModel（标准的 Megatron GPT 模型）
        #    b. bridge.load_weights(module, self.model_config.local_path)
        #       → 从临时目录加载权重（此时权重名已经不带 thinker. 前缀）
        #       → 老版 mbridge 能正确匹配参数名
        module = super()._build_megatron_module()
        return module
    finally:
        # 4. 恢复原始路径，清理临时目录
        self.model_config.local_path = orig_local_path
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

**权重提取的详细过程**：

```
原始 checkpoint 目录：
├── model-00001-of-00004.safetensors
│   ├── thinker.embed_tokens.weight          → 提取 → embed_tokens.weight
│   ├── thinker.layers.0.self_attn.qkv_proj → 提取 → layers.0.self_attn.qkv_proj
│   ├── talker.embed_tokens.weight           → 丢弃
│   └── visual.encoder.xxx                  → 丢弃
├── model-00002-of-00004.safetensors
│   ├── thinker.layers.10.self_attn.qkv_proj → 提取 → layers.10.self_attn.qkv_proj
│   ├── thinker.layers.10.mlp.gate.weight   → 提取 → layers.10.mlp.gate.weight
│   └── audio_tower.xxx                     → 丢弃
├── model-00003-of-00004.safetensors
│   ├── thinker.model.norm.weight           → 提取 → model.norm.weight
│   └── thinker.lm_head.weight              → 提取 → lm_head.weight
├── model-00004-of-00004.safetensors
│   └── code2wav.xxx                        → 丢弃（全部丢弃，生成空文件）
└── config.json

处理后（临时目录）：
├── model-00001-of-00004.safetensors  ← 只包含 thinker 权重
│   ├── embed_tokens.weight
│   └── layers.0.self_attn.qkv_proj
├── model-00002-of-00004.safetensors  ← 只包含 thinker 权重
│   ├── layers.10.self_attn.qkv_proj
│   └── layers.10.mlp.gate.weight
├── model-00003-of-00004.safetensors  ← 只包含 thinker 权重
│   ├── model.norm.weight
│   └── lm_head.weight
└── config.json  ← 修改后的配置（architectures 已替换）
```

### 6.3 NPU 训练配置

```yaml
actor_rollout_ref:
  model:
    model_type: omni_model              # 触发 OmniMindSpeedMegatronEngine
  actor:
    strategy: mindspeed_megatron        # 使用 MindSpeed 后端
    mindspeed:
      use_mbridge: true                 # 使用 mbridge
      vanilla_mbridge: true             # 使用老版 mbridge（NPU 必须）
      tensor_model_parallel_size: 4     # TP=4
      pipeline_model_parallel_size: 2   # PP=2
      param_offload: true               # 参数卸载到 CPU
      optimizer_offload: true           # 优化器状态卸载到 CPU
      grad_offload: true                # 梯度卸载到 CPU
      mcore_kwargs:
        spec: '[mindspeed_llm.tasks.models.spec.qwen3_spec, layer_spec]'
        # 使用 MindSpeed 的 Qwen3-MoE 优化版 layer spec
  rollout:
    name: vllm_omni                     # 使用 vLLM-Omni 作为 rollout 后端
    mode: async
    engine_kwargs:
      vllm_omni:
        output_mode: ar                 # 自回归输出模式
```

### 6.4 NPU 启动脚本

启动脚本的关键参数：

```
# 必须的参数
model_type=omni_model                     → 触发自定义引擎
strategy=mindspeed_megatron               → 使用 MindSpeed 后端
device=npu                                → 指定 NPU 设备
vanilla_mbridge=True                      → 使用老版 mbridge

# 并行配置
tensor_model_parallel_size=4              → 4 路张量并行
pipeline_model_parallel_size=2            → 2 路流水线并行

# Rollout
rollout.name=vllm_omni                    → 使用 vLLM-Omni
rollout.mode=async                         → 异步 rollout 模式
+engine_kwargs.vllm_omni.output_mode=ar    → 自回归输出
+engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe"  → 注册 Omni 流水线
+engine_kwargs.vllm_omni.stage_configs_path="path/to/thinker_only.yaml"  → Thinker 专用 stage 配置
```

---

## 7. 文件清单与修改说明

### 7.1 新建文件

| 文件 | 作用 | 行数 | 难度 |
|------|------|------|------|
| `verl_omni/workers/engine/mindspeed/__init__.py` | 模块入口，导出引擎 | 15 行 | 低 |
| `verl_omni/workers/engine/mindspeed/omni_impl.py` | **核心引擎**，包含 3 个方法重写 | 215 行 | 高 |
| `verl_omni/trainer/config/omni_mindspeed_trainer.yaml` | NPU 训练配置 | 70 行 | 中 |
| `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_mindspeed_npu.sh` | 启动脚本 | 120 行 | 低 |

### 7.2 修改文件

| 文件 | 修改内容 | 改动量 |
|------|---------|--------|
| `verl_omni/workers/engine/__init__.py` | 添加 `try/except` 导入新引擎 | +8 行 |

### 7.3 代码量统计

| 文件类型 | 总行数 | 说明 |
|---------|--------|------|
| Python 代码 | ~230 行 | 引擎 + 导入 |
| 配置 | ~70 行 | YAML 配置 |
| 脚本 | ~120 行 | Shell 启动脚本 |
| **总计** | **~420 行** | |

### 7.4 预计工作量

| 任务 | 需要的时间 | 说明 |
|------|-----------|------|
| 理解现有代码 | 1 天 | 阅读 verl 和 verl-omni 的引擎代码 |
| 编写核心引擎 | 1-2 天 | 实现 3 个方法重写 |
| 编写配置和脚本 | 0.5 天 | YAML + Shell |
| 首次调试 | 1-2 天 | 需要 NPU 环境，解决 mbridge 兼容性问题 |
| **总计** | **3-5 天** | |

---

## 8. 常见问题与调试指南

### 8.1 mbridge 创建失败

**错误现象**：`_build_tf_config()` 报错，bridge 无法创建

**可能原因**：
1. `thinker_config` 不存在 → 检查 `hf_config` 是否有 `thinker_config` 属性
2. `architectures` 替换失败 → 确认 `thinker_config.architectures` 设置正确
3. `thinker_config` 缺少某些字段 → 对比 Qwen3-MoE 的 config.json，确认所有必填字段都在

**调试方法**：
```python
# 在 _build_tf_config 中添加调试代码
print("thinker_config keys:", thinker_config.keys())
print("actor architectures:", thinker_config.get("architectures"))
print("num_hidden_layers:", thinker_config.get("num_hidden_layers"))
```

### 8.2 权重加载失败

**错误现象**：`_build_megatron_module()` 报错，权重大小不匹配

**可能原因**：
1. 临时目录中的权重名与 Megatron 模型参数名不匹配
2. 某些权重被遗漏（如 `model.norm.weight` 和 `lm_head.weight`）
3. checkpoint 文件格式不是 safetensors

**调试方法**：
```python
# 验证临时目录中的权重
import os
from safetensors.torch import load_file
for f in os.listdir(tmp_dir):
    if f.endswith(".safetensors"):
        d = load_file(os.path.join(tmp_dir, f))
        print(f"{f}: {len(d)} tensors")
        for k in list(d.keys())[:3]:
            print(f"  {k}: {d[k].shape}")
```

### 8.3 MindSpeed 报错

**错误现象**：`apply_patch()` 报错或训练时算子报错

**可能原因**：
1. `qwen3_spec` 与 thinker 的 decoder 层不完全兼容
2. `apply_patch()` 读取了错误的 `hf_config` 字段

**解决方案**：
1. 如果不传 `spec`，使用默认 spec（可能性能略差，但更兼容）
2. 确认 `hf_config` 在调用 `apply_patch()` 时已替换为 `thinker_config`

### 8.4 训练 loss 不下降

**错误现象**：训练可以跑，但 loss 不下降或结果异常

**可能原因**：
1. 权重加载不正确（可能加载了错误的权重）
2. 权重同步不正确（vLLM-Omni 没有收到正确的权重更新）

**调试方法**：
1. 在训练前打印几个关键参数的数值，确认权重加载正确
2. 在权重同步后打印参数，确认同步成功

### 8.5 显存不足

**错误现象**：OOM（Out of Memory）

**解决方案**：
1. 减小 `micro_batch_size`
2. 增大 `tensor_model_parallel_size` 或 `pipeline_model_parallel_size`
3. 启用 `param_offload`、`optimizer_offload`、`grad_offload`
4. 减小 `max_model_len` 或 `max_prompt_length`

---

## 9. 附录：相关术语表

| 术语 | 全称 | 通俗解释 |
|------|------|---------|
| **Megatron** | Megatron-LM | NVIDIA 的模型并行训练框架，把大模型切分到多卡训练 |
| **MindSpeed** | - | 华为昇腾 NPU 上的 Megatron 兼容库，用 CANN 算子替代 CUDA |
| **mbridge** | Megatron Bridge | verl 中在 HF 格式和 Megatron 格式之间转换权重的桥梁 |
| **TP** | Tensor Parallel | 张量并行，把一层网络切分到多卡 |
| **PP** | Pipeline Parallel | 流水线并行，把不同层放到不同卡 |
| **CANN** | - | 昇腾 NPU 的运算库（类似 NVIDIA 的 CUDA） |
| **vLLM-Omni** | - | verl-omni 的标准推理引擎，支持多模态模型（Thinker/Talker/Code2Wav），支持 GPU 和 NPU |
| **vLLM** | - | 通用大模型推理引擎，vLLM-Omni 是 verl-omni 的定制版本 |
| **HF 格式** | HuggingFace 格式 | 人类可读的模型权重格式（如 `model.layers.0.self_attn.qkv_proj.weight`）|
| **Megatron 格式** | - | 训练用的模型权重格式（如 `decoder.layers.0.self_attention.linear_qkv.weight`）|
| **checkpoint** | - | 模型权重的保存文件（.safetensors 或 .bin）|
| **Rollout** | - | RL 训练中模型生成样本的阶段 |
| **FSDP** | Fully Sharded Data Parallel | 数据并行 + 模型分片，另一种训练并行策略 |
| **GSPO** | - | verl-omni 中使用的 RL 算法（Group-based Supervised Policy Optimization）|

---

## 附录：相关链接

- [PR #399](https://github.com/verl-project/verl-omni/pull/399) - GPU Megatron Qwen3-Omni 训练实现
- [Omni Megatron 方案设计文档](file:///Users/ltw/Documents/workspace/verl-omni/docs/algo/omni-megatron-impl方案.md) - 原始方案设计
- [verl-0.9.0 MindSpeed 引擎](file:///Users/ltw/Documents/workspace/verl-0.9.0/verl/workers/engine/mindspeed/) - 上游 verl 的 NPU 引擎实现参考
- [verl-0.9.0 NPU 训练脚本](file:///Users/ltw/Documents/workspace/verl-0.9.0/tests/special_npu/run_qwen3_8b_grpo_mindspeedllm.sh) - 上游 verl 的 NPU 训练脚本参考