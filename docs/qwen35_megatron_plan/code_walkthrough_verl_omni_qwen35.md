# Qwen3.5 训练全流程代码走读（verl + verl-omni 联合）

> 本文档基于 verl-0.9.0 与 verl-omni 源码，对 Qwen3.5-35B-A3B 在 NPU 上使用 FSDP 后端训练的完整流程进行逐层代码走读。适合对框架不熟悉、需要结合 verl 源码理解 verl-omni 适配逻辑的读者。

---

## 总览：一次训练怎么跑起来的？

```
[你的手] 敲下 bash run_qwen35_moe_thinker_gspo_npu.sh
    ↓
[verl-omni] main_omni.py 接收参数
    ↓
[verl] main_ppo.py::TaskRunnerV1.run() 启动 Ray 集群
    ↓
[verl] get_trainer_cls("omni_sync") → OmniPPOTrainerSync
    ↓
[verl] trainer.init() → 初始化所有组件
    ↓
[verl] trainer.fit() → 进入训练循环
    └── 每步循环: Rollout → Ref → Actor → 权重同步
```

---

## 第 1 步：启动脚本（用户视角）

**文件**：`examples/gspo_trainer/qwen35_moe/run_qwen35_moe_thinker_gspo_npu.sh`

这个脚本做的事情：**把 100 多个参数组织好，传给 Python 程序**。

关键参数拆解：

```bash
python3 -m verl_omni.trainer.main_omni \
    # 1. 数据相关
    data.train_files=... \
    data.val_files=... \

    # 2. 模型路径 — 告诉 verl "模型在哪"
    actor_rollout_ref.model.path=Qwen/Qwen3.5-35B-A3B \

    # 3. engine 配置 — 告诉 verl "用 FSDP 还是 FSDP2"
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \  # 注意：这里用 fsdp2，不是 fsdp

    # 4. rollout 配置 — 告诉 vLLM "Qwen3.5 走哪个 pipeline"
    actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_5_moe \

    # 5. 训练算法配置
    algorithm.adv_estimator=gspo \  # GSPO 算法

    # 6. 训练器模式 — 注册键是 "omni_sync"，告诉 verl 用 omni 专用的训练器
    trainer.v1.trainer_mode=omni_sync
```

---

## 第 2 步：入口函数（verl-omni → verl）

**文件**：`verl_omni/trainer/main_omni.py` → `verl/trainer/main_ppo.py`

```python
# main_omni.py 第 346 行
@hydra.main(config_path="./config", config_name="omni_trainer", version_base=None)
def main(config):
    auto_set_device(config)        # 检测是 NPU 还是 GPU
    validate_config(config)        # 校验参数
    run_omni(config)               # 启动训练
```

```python
# main_omni.py 第 330 行
def run_omni(config, task_runner_class=None):
    from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
    run_ppo(config, task_runner_class=TaskRunnerV1)
```

这里从 verl-omni 跳到了 **verl 的框架**。`run_ppo` 做两件事：

1. **初始化 Ray 集群**（第 57-75 行）
2. **创建 TaskRunnerV1 的远程实例并执行**（第 93-94 行）

```python
# verl/trainer/main_ppo.py 第 93 行
runner = task_runner_class.remote()
ray.get(runner.run.remote(config))
```

---

## 第 3 步：TaskRunnerV1.run() — 训练器的"总指挥"

**文件**：`verl/trainer/main_ppo.py` 第 134 行

```python
def run(self, config):
    # 1. 根据 trainer_mode 找到训练器类
    #    trainer_mode = "omni_sync"
    trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
    #    ↑ 从 TRAINER_REGISTRY 字典里找到 OmniPPOTrainerSync

    # 2. 初始化 TransferQueue（数据中转站）
    tq.init(config.transfer_queue)

    # 3. 创建训练器实例
    self.trainer = trainer_cls(config=config)

    # 4. 初始化训练器
    #    → 加载 tokenizer → 初始化数据加载器 → 创建 Ray 工作器组 → 加载模型
    self.trainer.init()

    # 5. 初始化 AgentLoop（负责生成回答的异步管理器）
    self.init_agent_loop_manager()

    # 6. 开始训练循环
    self.trainer.fit(self.agent_loop_manager)
```

**关键点**：`get_trainer_cls("omni_sync")` 是怎么找到 `OmniPPOTrainerSync` 的？

```
TRAINER_REGISTRY 是一个全局字典
  ↓
verl 默认注册了 "sync" → PPOTrainerSync
  ↓
verl-omni 的 __init__.py 被 import 时，自动执行：
  import verl_omni.trainer.omni  # noqa: F401
  ↓
verl_omni/trainer/omni/ray_omni_trainer.py 第 65 行：
  @register_trainer("omni_sync")  → 往 TRAINER_REGISTRY 又加了一条
  class OmniPPOTrainerSync(PPOTrainerSync):
  ↓
TRAINER_REGISTRY = {
    "sync": PPOTrainerSync,
    "omni_sync": OmniPPOTrainerSync,  # ← 新增的
}
```

---

## 第 4 步：PPOTrainer.init() — 初始化所有组件

**文件**：`verl/trainer/ppo/v1/trainer_base.py` 第 217 行

```python
def init(self):
    self._setup()   # 初始化所有组件
    self.on_init_end()  # 更新权重
```

```python
def _setup(self):
    self._init_tokenizer()  # ← 加载 tokenizer
    self._init_dataloader()  # ← 加载数据集
    self._init_resource_pool_mgr()  # ← 分配 GPU 资源
    self.resource_pool_manager.create_resource_pool()

    # 创建 Actor+Rollout+Ref 的工作器（Ray 远程对象）
    actor_rollout_cls = RayClassWithInitArgs(
        cls=self.role_worker_mapping[actor_role],
        config=self.config.actor_rollout_ref,
    )

    # ... 创建 RayWorkerGroup

    # 初始化模型
    self.actor_rollout_wg.init_model()  # ← 加载模型
```

---

## 第 5 步：_init_tokenizer() — 加载分词器 / 处理器

**文件**：`verl/trainer/ppo/v1/trainer_base.py` 第 646 行

```python
def _init_tokenizer(self):
    model_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.model)
    self.tokenizer = model_config.tokenizer
    self.processor = model_config.processor
```

这里调用 `omega_conf_to_dataclass` 把 YAML 配置转换为 Python 对象。对于 Qwen3.5，这个 `model_config` 是一个 **OmniModelConfig** 对象。

**关键**：`OmniModelConfig.__post_init__()` 在构造时执行了最重要的逻辑：

```python
# verl_omni/workers/config/omni/model.py 第 142 行
def __post_init__(self):
    # 1. 读取模型的 config.json，获取 architecture
    #    config.json 里的 architectures = ["Qwen3_5MoeForConditionalGeneration"]
    self.architecture = json.load(f)["architectures"][0]

    # 2. 在 OmniModelBase 注册表中查找适配器
    adapter_cls = OmniModelBase.get_class_by_name(
        self.architecture,      # "Qwen3_5MoeForConditionalGeneration"
        self.model_stage,       # "thinker"
        self.external_lib,
    )

    # 3. 用适配器加载 tokenizer 和 processor
    self.tokenizer = adapter_cls.configure_tokenizer(...)
    self.processor = adapter_cls.configure_processor(...)
```

**注册表查找过程**：

```
OmniModelBase._registry = {
    ("Qwen3_5MoeForConditionalGeneration", "thinker"): Qwen35MoeThinkerAdapter,
    ("Qwen3OmniMoeForConditionalGeneration", "thinker"): Qwen3OmniThinkerAdapter,
    ...
}
```

**`Qwen35MoeThinkerAdapter.configure_tokenizer`** 做了什么？

```python
# verl_omni/pipelines/qwen35moe/thinker_training_adapter.py 第 39 行
@classmethod
def configure_tokenizer(cls, model_path, model_config):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        # 从 config.json 的 chat_template 字段加载
        chat_template=model_config.hf_config.chat_template,
    )
    # 对于 Qwen3.5，pad_token 和 eos_token 是同一个 token
    # 需要手动 copy 一份，避免 pad=pad=eos 导致训练时
    # 还没生成完就截断了
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    return tokenizer
```

**`configure_processor`** 做了什么？

```python
@classmethod
def configure_processor(cls, model_path, model_config):
    # 必须返回一个真正的 processor，不能返回 None
    # 因为 verl 的 DataLoader 需要 processor 来处理图像
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return processor
```

---

## 第 6 步：ActorRolloutRefWorker.init_model() — 创建三大金刚

**文件**：`verl_omni/workers/engine_workers.py` 第 606 行

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    model_config = omega_conf_to_dataclass(self.config.model)

    # 1. 构建 REF 模型（参考模型）
    if "ref" in self.role:
        ref_config = omega_conf_to_dataclass(self.config.ref)
        ref_config.model_config = deepcopy(model_config)
        self.ref = TrainingWorker(config=ref_training_config)  # ← 创建 TrainingWorker
        self.ref.reset()  # ← 初始化引擎，加载模型

    # 2. 构建 ACTOR 模型（训练模型）
    if "actor" in self.role:
        actor_config = omega_conf_to_dataclass(self.config.actor)
        actor_config.model_config = model_config
        self.actor = TrainingWorker(config=actor_training_config)
        self.actor.reset()  # ← 初始化引擎，加载模型
        self.actor.set_loss_fn(self.loss_fn)  # ← 设置损失函数

    # 3. 构建 Rollout 引擎（vLLM-Omni）
    rollout_cls = get_rollout_class(rollout_config.name, rollout_config.mode)
    self.rollout = rollout_cls(config=rollout_config, ...)
```

---

## 第 7 步：TrainingWorker() — 引擎注册和创建

**文件**：`verl/workers/engine_workers.py` 第 76 行

```python
class TrainingWorker(Worker, DistProfilerExtension):
    def __init__(self, config: TrainingWorkerConfig):
        ...
        self.model_config.model_type = self.config.model_type
        # 根据 model_type 和 backend 创建引擎
        # model_type = "omni_model", backend = "fsdp2"
        self.engine = EngineRegistry.new(
            model_type=self.config.model_type,  # "omni_model"
            backend=self.engine_config.strategy,  # "fsdp2"
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )
```

**EngineRegistry 查找过程**：

```
EngineRegistry._engines["omni_model"]["fsdp2"] = {
    "cuda": OmniFSDPEngine,  # ← 找到了！
    "npu": OmniFSDPEngine,   # ← 或者这个
}
```

注册发生在 `omni_impl.py` 第 42 行：

```python
@EngineRegistry.register(
    model_type="omni_model",
    backend=["fsdp", "fsdp2"],
    device=["cuda", "npu"]
)
class OmniFSDPEngine(FSDPEngineWithLMHead):
    ...
```

---

## 第 8 步：OmniFSDPEngine._build_module() — 加载模型

**文件**：`verl_omni/workers/engine/fsdp/omni_impl.py` 第 145 行

这是**最关键的一步**，完整还原模型加载过程：

```python
def _build_module(self):
    architecture = self.model_config.architecture  # "Qwen3_5MoeForConditionalGeneration"
    torch_dtype = ...  # 根据配置决定

    # 1. 用 HuggingFace 的 AutoModel 加载模型
    #    (注意：不是 verl 的 get_hf_auto_model_class，而是 HF 的 AutoModelForMultimodalLM)
    module = AutoModelForMultimodalLM.from_pretrained(
        pretrained_model_name_or_path=self.model_config.local_path,
        torch_dtype=torch_dtype,
        config=self.model_config.hf_config,
        trust_remote_code=self.model_config.trust_remote_code,
    )

    # ↓ 对比：如果是普通语言模型，verl 的 FSDPEngine 用的是：
    #   auto_class = get_hf_auto_model_class(hf_config)
    #   module = auto_class.from_pretrained(...)
    #   并且会调用 apply_monkey_patch(model, ...)
    #   Omni 不走这个路径，它自己处理

    # 2. 找到适配器
    adapter_cls = OmniModelBase.get_class_by_name(
        architecture,       # "Qwen3_5MoeForConditionalGeneration"
        self.model_config.model_stage,  # "thinker"
    )

    # 3. 调用适配器的 configure_model
    #    → 打设备补丁（devices fix）
    #    → 剥离不需要的模块
    #    → 冻结不需要的部分
    module = adapter_cls.configure_model(module, self.model_config)

    # 4. 统一 dtype
    module.to(torch_dtype)

    # 5. 开启梯度检查点（省显存）
    if self.model_config.enable_gradient_checkpointing:
        module.gradient_checkpointing_enable(...)

    return module
```

---

## 第 9 步：Qwen35MoeThinkerAdapter.configure_model() — 适配器干活

**文件**：`verl_omni/pipelines/qwen35moe/thinker_training_adapter.py` 第 57 行

```python
@classmethod
def configure_model(cls, module, model_config):
    # 1. 打设备补丁
    #    → 解决 CPU 卸载时 pos_embed.weight 在 CPU 上
    #      但 forward 计算在 NPU 的错位问题
    apply_qwen3_5_vision_device_fix()

    # 2. 调用父类的 configure_model（剥离不需要的模块）
    #    → get_strip_modules() 返回空列表，所以不删任何东西
    return super().configure_model(module, model_config)
```

**对比 Qwen3-Omni 的 configure_model**：

```python
# Qwen3OmniThinkerAdapter.configure_model (第 48 行)
@classmethod
def configure_model(cls, module, model_config):
    # 1. 把 Qwen3OmniMoeForConditionalGeneration 注册到
    #    AutoModelForCausalLM，这样 vLLM 才能识别
    AutoModelForCausalLM.register(...)

    # 2. 重定向 forward 到 thinker 的 forward
    module.forward = module.thinker.forward

    # 3. 剥离 talker、code2wav、code_predictor
    #    因为 thinker-only 训练不需要这些模块

    return super().configure_model(module, model_config)
```

**Qwen3.5 不需要这些**，因为它本身就是 thinker-only 的模型，没有 talker 和 codec。

---

## 第 10 步：设备补丁 — qwen3_5_moe_vision.py

**文件**：`verl_omni/models/transformers/qwen3_5_moe_vision.py`

**问题重现**：REF 模型（参考模型）用了 FSDP2 的 CPU 卸载（`CPUOffloadPolicy(pin_memory=True)`），参数字典在 CPU 上。但 Qwen3.5 的视觉位置编码插值里，需要创建一个 index tensor，原来的代码是：

```python
# 原来的逻辑（在 HuggingFace 源码里）
target_device = self.pos_embed.weight.device  # CPU！
index_tensor = torch.arange(..., device=target_device)  # 在 CPU 上创建
# 接着跟 NPU 上的 hidden_states 运算 → 报错！
```

**补丁的原理**：用猴子补丁（monkey patch）替换原始方法，让 index tensor 从**激活值**（已经在 NPU 上）获取设备：

```python
# 补丁后的逻辑
# 1. 在 forward 入口处，记录激活值的设备
original_forward = VLMVisionModel.forward
def patched_forward(self, hidden_states, grid_thw, *args, **kwargs):
    self._verl_omni_compute_device = hidden_states.device  # 记录！
    return original_forward(self, hidden_states, grid_thw, *args, **kwargs)
VLMVisionModel.forward = patched_forward

# 2. 在位置编码插值处，用激活的设备
original_interpolate = pos_embed.interpolate
def patched_interpolate(self, grid_thw, *args, **kwargs):
    target_device = getattr(self, "_verl_omni_compute_device", None)
    if target_device is not None:
        # 临时把 pos_embed 搬到正确的设备
        self.pos_embed = self.pos_embed.to(target_device)
    return original_interpolate(self, grid_thw, *args, **kwargs)
pos_embed.__class__.interpolate = patched_interpolate
```

---

## 第 11 步：OmniFSDPEngine._build_fsdp_module() — FSDP 包装

**文件**：`verl/workers/engine/fsdp/transformer_impl.py` 第 365 行

模型加载完成后，需要 FSDP 包装才能分布式训练：

```python
def _build_model_optimizer(self):
    module = self._build_module()  # ← 刚刚加载的模型

    # 如果是 LoRA，先包装 LoRA
    if self._is_lora:
        module = self._build_lora_module(module)

    # FSDP 包装
    module = self._build_fsdp_module(module)

    # 创建优化器
    optimizer = self._build_optimizer(module)
    lr_scheduler = self._build_lr_scheduler(optimizer)

    self.module = module
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
```

```python
def _build_fsdp_module(self, module):
    if self.engine_config.strategy == "fsdp2":  # 我们的配置
        mp_policy = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)

        offload_policy = None
        if self.engine_config.forward_only:  # REF 模型 → CPU 卸载
            offload_policy = CPUOffloadPolicy(pin_memory=True)

        apply_fsdp2(module, fsdp_kwargs, self.engine_config)
        fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)

    return module
```

---

## 第 12 步：init_agent_loop_manager() — 起 Rollout 引擎

**文件**：`verl/trainer/main_ppo.py` 第 112 行

```python
def init_agent_loop_manager(self):
    # 默认用 AgentLoopManagerTQ
    self.agent_loop_manager = AgentLoopManagerTQ.create(
        config=self.config,
        llm_client=self.trainer.get_llm_client(),  # vLLM-Omni 客户端
        reward_loop_worker_handles=self.trainer.get_reward_handles(),
    )
```

**AgentLoopManagerTQ** 负责：

1. 从数据集中取一批 prompt
2. 发给 vLLM-Omni 引擎生成回答
3. 把生成结果放到 TransferQueue 里
4. 训练器从 TransferQueue 里取出来训练

**Rollout 适配器**是 `Qwen35MoeRolloutAdapter`，它告诉 vLLM-Omni：

- pipeline_id = `qwen3_5_moe_thinker_only`
- 只有一个 stage（thinker stage）
- 在 NPU 上额外设置 `weight_nz_mode=0`

---

## 第 13 步：PPOTrainer.fit() — 训练循环

**文件**：`verl/trainer/ppo/v1/trainer_base.py` 第 387 行

```python
def fit(self, agent_loop_manager):
    # 初始化跟踪器
    self.logger = Tracking(...)

    # 验证
    if self.config.trainer.get("val_before_train", True):
        self._validate()

    # 训练循环
    while current_epoch < total_epochs and global_steps <= total_training_steps:
        # 1. 执行一步训练
        batch = self.step(metrics, timing_raw)

        # 2. 保存 checkpoint
        if global_steps % save_freq == 0:
            self._save_checkpoint()

        # 3. 同步权重到 vLLM 引擎
        self.on_step_end()
        #    → OmniPPOTrainerSync.on_step_end()
        #    → checkpoint_manager.update_weights()

        # 4. 验证
        if global_steps % test_freq == 0:
            self._validate()

        # 5. 记录指标
        self.logger.log(data=metrics, step=global_steps)
        global_steps += 1
```

---

## 第 14 步：PPOTrainer._step_once() — 单步训练的核心

**文件**：`verl/trainer/ppo/v1/trainer_base.py` 第 536 行

```python
def _step_once(self, metrics, timing_raw, sample_batch_size):
    # 1. 从 replay buffer 采样一批
    batch, off_policy_metrics = self.replay_buffer.sample(...)

    # 2. 计算奖励（如果 rollout 阶段没算的话）
    if self.reward_loop_manager.reward_loop_worker_handles is None:
        batch = self._compute_reward_colocate(batch, metrics=metrics)

    # 3. 平衡 batch 到各 DP 组
    batch = self._balance_batch(batch, metrics=metrics)

    # 4. 计算 old_log_prob（actor 的旧概率）
    batch = self._compute_old_log_prob(batch, metrics=metrics)

    # 5. 计算 ref_log_prob（参考模型概率，用于 KL 散度）
    if self.use_reference_policy:
        batch = self._compute_ref_log_prob(batch, metrics=metrics)
        # 注意：这里会触发 REF 模型的 forward
        # REF 模型用了 CPU 卸载 → 参数在 CPU 上
        # → 触发设备补丁！

    # 6. 计算优势函数 advantage
    batch = self._compute_advantage(batch, metrics=metrics)

    # 7. 更新 actor 参数
    batch = self._update_actor(batch, metrics=metrics)

    return batch
```

---

## 完整调用链总结（所有文件一览）

```
┌──────────────────────────────────────────────────────────────────┐
│  run_qwen35_moe_thinker_gspo_npu.sh                              │
│  │ 设置环境变量、检测卡数、组装 100+ 参数                        │
│  ▼                                                               │
├──────────────────────────────────────────────────────────────────┤
│  verl_omni/trainer/main_omni.py                                  │
│  │ @hydra.main() 解析 YAML 配置 → run_omni()                     │
│  ▼                                                               │
├──────────────────────────────────────────────────────────────────┤
│  verl/trainer/main_ppo.py                                        │
│  │ run_ppo() → 初始化 Ray → TaskRunnerV1.run()                   │
│  │ run(): get_trainer_cls("omni_sync") → OmniPPOTrainerSync      │
│  │        trainer.init() → trainer.fit(agent_loop_manager)        │
│  ▼                                                               │
├──────────────────────────────────────────────────────────────────┤
│  PPOTrainer.init() (verl/trainer/ppo/v1/trainer_base.py)         │
│  │ _init_tokenizer()                                              │
│  │   → OmniModelConfig.__post_init__()  (verl_omni/workers/config/omni/model.py) │
│  │     → OmniModelBase.get_class_by_name("Qwen3_5MoeForConditionalGeneration", "thinker") │
│  │       → Qwen35MoeThinkerAdapter.configure_tokenizer()          │
│  │       → Qwen35MoeThinkerAdapter.configure_processor()          │
│  │ _init_dataloader()                                             │
│  │ _init_resource_pool_mgr()                                       │
│  │ actor_rollout_wg.init_model()                                  │
│  ▼                                                               │
├──────────────────────────────────────────────────────────────────┤
│  ActorRolloutRefWorker.init_model() (verl_omni/workers/engine_workers.py) │
│  │ 创建 REF TrainingWorker → ref.reset()                          │
│  │   → EngineRegistry.new("omni_model", "fsdp2") → OmniFSDPEngine │
│  │   → OmniFSDPEngine.initialize() → _build_model_optimizer()     │
│  │     → _build_module()  (verl_omni/workers/engine/fsdp/omni_impl.py) │
│  │       → AutoModelForMultimodalLM.from_pretrained() 加载模型     │
│  │       → Qwen35MoeThinkerAdapter.configure_model()              │
│  │         → apply_qwen3_5_vision_device_fix() 打设备补丁          │
│  │       → _build_fsdp_module()  FSDP2 包装                        │
│  │ 创建 ACTOR TrainingWorker → actor.reset()  (同上流程)          │
│  │ 创建 Rollout 引擎                                              │
│  │   → Qwen35MoeRolloutAdapter.get_pipeline_id() → "qwen3_5_moe_thinker_only" │
│  ▼                                                               │
├──────────────────────────────────────────────────────────────────┤
│  PPOTrainer.fit() (verl/trainer/ppo/v1/trainer_base.py)           │
│  │ 训练循环（每步调用 _step_once()）：                              │
│  │   1. sample() → 从 replay buffer 拿一批数据                    │
│  │   2. _compute_reward_colocate() → 计算奖励                     │
│  │   3. _compute_old_log_prob() → actor 算旧概率                  │
│  │   4. _compute_ref_log_prob() → ref 算参考概率                  │
│  │      → 触发 REF 模型 forward → 触发设备补丁                    │
│  │   5. _compute_advantage() → 计算优势函数                        │
│  │   6. _update_actor() → 更新 actor 参数                         │
│  │   7. on_step_end() → 权重同步到 vLLM 引擎                      │
│  └──────────────────────────────────────────────────────────────────
```

---

## 一张图看懂 verl vs verl-omni 的分工

```
┌─────────────────────────────────────────────────────────────────┐
│                        verl (框架基座)                            │
│                                                                  │
│  main_ppo.py       训练器注册/调度                                 │
│  trainer_base.py   PPOTrainer 基类、训练循环、_init_tokenizer    │
│  trainer_sync.py   PPOTrainerSync（基础同步训练器）               │
│  main_ppo.py       TaskRunnerV1（Ray 任务编排）                   │
│  engine_workers.py  TrainingWorker、EngineRegistry.new()          │
│  transformer_impl.py  FSDPEngine._build_module()、_build_fsdp...  │
│  model.py           HFModelConfig（基础模型配置）                  │
│  agent_loop_tq.py   AgentLoopWorkerTQ（生成回答）                 │
└───────────────────┬─────────────────────────────────────┘
                    │ 通过 import 扩展
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    verl-omni (模型适配层)                         │
│                                                                  │
│  main_omni.py         自定义入口，使用 omni 配置                  │
│  ray_omni_trainer.py  OmniPPOTrainerSync（注册 omni_sync）        │
│  omni_impl.py         OmniFSDPEngine（注册 omni_model+fsdp2）     │
│  omni/model.py        OmniModelConfig（自动检测 architecture）    │
│  model_base.py        OmniModelBase（注册表 + configure_model）   │
│  thinker_training_adapter.py  Qwen35MoeThinkerAdapter             │
│  omni_rollout_adapter.py      Qwen35MoeRolloutAdapter             │
│  qwen3_5_moe_vision.py        设备补丁（猴子补丁）               │
└─────────────────────────────────────────────────────────────────┘
```

**总结：verl 是骨架，verl-omni 是肉。verl 提供了训练框架（FSDP、Ray、训练循环），verl-omni 把 Qwen3.5 模型"塞进去"让它能跑起来。**

