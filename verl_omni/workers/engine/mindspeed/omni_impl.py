# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MindSpeed engines for Qwen3-Omni Thinker model on NPU.

Two engines are provided:

1. **OmniMindSpeedMegatronEngine** (``vanilla_mbridge=True``, legacy):
   Registered as ``model_type="omni_model", backend=["mindspeed_megatron"], device=["npu"]``.
   Uses the old mbridge path. Requires manual ``hf_config`` swapping and
   thinker weight extraction. See docstring for details.

2. **OmniMindspeedMegatronBridgeEngine** (``vanilla_mbridge=False``, recommended):
   Registered as ``model_type="omni_model", backend=["megatron"], device=["npu"]``.
   Uses the new Megatron-Bridge path, which natively supports Qwen3-Omni
   architecture detection. No ``hf_config`` swapping or weight extraction
   needed.
"""

import importlib.metadata as _ilm
import json
import logging
import os
import shutil
import sys
import tempfile
from types import ModuleType

import torch

from safetensors.torch import load_file, save_file

from verl.trainer.config.config import CheckpointConfig
from verl.workers.config import (
    McoreOptimizerConfig,
    MindSpeedEngineConfig,
)
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngine
from verl.workers.engine.mindspeed.transformer_impl import (
    MindSpeedMegatronEngineWithLMHead,
    MindspeedEngineWithLMHead,
)
from verl_omni.workers.config import OmniModelConfig

logger = logging.getLogger(__name__)


@EngineRegistry.register(model_type="omni_model", backend=["mindspeed_megatron"], device=["npu"])
class OmniMindSpeedMegatronEngine(MindSpeedMegatronEngineWithLMHead):
    """MindSpeed Megatron engine for Qwen3-Omni Thinker model on NPU."""

    def __init__(
        self,
        model_config: OmniModelConfig,
        engine_config: MindSpeedEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    # ------------------------------------------------------------------
    # 1. Bridge compatibility: 让老版 mbridge 把 Omni 当作 Qwen3-MoE
    # ------------------------------------------------------------------

    def _build_tf_config(self):
        """Build transformer config using thinker_config instead of full Omni config.

        The old mbridge (``vanilla_mbridge=True``) only recognizes standard
        architectures like ``Qwen3MoeForCausalLM``. We temporarily replace the
        hf_config with thinker_config and set the architectures field so that
        the bridge can create a valid Megatron transformer config.
        """
        original_hf_config = self.model_config.hf_config
        thinker_config = getattr(original_hf_config, "thinker_config", None)
        if thinker_config is None:
            raise ValueError(
                "OmniMindSpeedMegatronEngine requires a Qwen3-Omni model with "
                "thinker_config in hf_config, but thinker_config was not found."
            )

        # 让 thinker_config 伪装成 Qwen3-MoE 配置
        thinker_config.architectures = ["Qwen3MoeForCausalLM"]
        self.model_config.hf_config = thinker_config

        try:
            super()._build_tf_config()
        finally:
            # 恢复原始 hf_config（bridge 已创建，不受影响）
            self.model_config.hf_config = original_hf_config

    # ------------------------------------------------------------------
    # 2. MindSpeed patch: 用 thinker_config 调用 apply_patch
    # ------------------------------------------------------------------

    def _init_device_mesh(self):
        """Initialize device mesh with MindSpeed patch applied.

        ``apply_patch()`` reads model architecture parameters from
        ``hf_config`` (e.g. ``num_hidden_layers``, ``hidden_size``).
        These fields are in ``thinker_config``, not the top-level Omni config,
        so we temporarily swap them.
        """
        original_hf_config = self.model_config.hf_config
        thinker_config = getattr(original_hf_config, "thinker_config", None)
        if thinker_config is not None:
            thinker_config.architectures = ["Qwen3MoeForCausalLM"]
            self.model_config.hf_config = thinker_config

        try:
            apply_patch(self.model_config, self.engine_config, self.optimizer_config)
            # 调用 MegatronEngine._init_device_mesh 而非父类的
            # MindSpeedMegatronEngineWithLMHead._init_device_mesh
            # （因为父类已经调用了 apply_patch，我们在这里已经调过了）
            MegatronEngine._init_device_mesh(self)
        finally:
            self.model_config.hf_config = original_hf_config

    # ------------------------------------------------------------------
    # 3. Weight extraction: 提取 thinker 权重，去掉前缀
    # ------------------------------------------------------------------

    def _build_megatron_module(self):
        """Build Megatron module with thinker weights extracted from Omni checkpoint."""
        orig_local_path = self.model_config.local_path

        # 将 thinker 权重提取到临时目录，去掉 thinker. 前缀
        tmp_dir = self._prepare_thinker_checkpoint(orig_local_path)
        self.model_config.local_path = tmp_dir

        try:
            module = super()._build_megatron_module()
            return module
        finally:
            self.model_config.local_path = orig_local_path
            # 清理临时目录
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {tmp_dir}: {e}")

    def _prepare_thinker_checkpoint(self, checkpoint_path: str) -> str:
        """Extract thinker weights from the Omni checkpoint.

        Args:
            checkpoint_path: Path to the original Omni checkpoint directory.

        Returns:
            Path to a temporary directory containing the processed checkpoint.
        """
        tmp_dir = tempfile.mkdtemp(prefix="omni_thinker_ckpt_")

        # 查找所有 safetensors 文件
        safetensors_files = [
            fname for fname in os.listdir(checkpoint_path)
            if fname.endswith(".safetensors")
        ]

        if safetensors_files:
            self._process_safetensors(checkpoint_path, tmp_dir, safetensors_files)
        else:
            # 尝试 PyTorch 格式 (.bin / .pt)
            pytorch_files = [
                fname for fname in os.listdir(checkpoint_path)
                if fname.endswith(".bin") or fname.endswith(".pt")
            ]
            if pytorch_files:
                self._process_pytorch(checkpoint_path, tmp_dir, pytorch_files)
            else:
                raise FileNotFoundError(
                    f"No .safetensors, .bin or .pt files found in {checkpoint_path}"
                )

        # 复制并修改 config.json
        self._prepare_config_json(checkpoint_path, tmp_dir)

        # 复制 tokenizer 文件（如果存在）
        for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
            src = os.path.join(checkpoint_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp_dir, fname))

        logger.info(f"Prepared thinker checkpoint at {tmp_dir}")
        return tmp_dir

    def _process_safetensors(self, src_dir: str, dst_dir: str, files: list[str]):
        """Process safetensors files: keep only thinker weights, strip prefix."""
        for fname in files:
            filepath = os.path.join(src_dir, fname)
            tensor_dict = load_file(filepath)
            # 只保留 thinker. 开头的权重
            new_dict = {}
            for key, value in tensor_dict.items():
                if key.startswith("thinker."):
                    new_key = key[len("thinker."):]  # 去掉 thinker. 前缀
                    new_dict[new_key] = value
                # 丢弃 talker, code2wav, visual, audio_tower 等
            if new_dict:
                save_file(new_dict, os.path.join(dst_dir, fname))

    def _process_pytorch(self, src_dir: str, dst_dir: str, files: list[str]):
        """Process PyTorch format files: keep only thinker weights, strip prefix."""
        for fname in files:
            ckpt = torch.load(
                os.path.join(src_dir, fname),
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(ckpt, dict):
                new_ckpt = {}
                for key, value in ckpt.items():
                    if key.startswith("thinker."):
                        new_key = key[len("thinker."):]
                        new_ckpt[new_key] = value
                if new_ckpt:
                    torch.save(new_ckpt, os.path.join(dst_dir, fname))

    def _prepare_config_json(self, src_dir: str, dst_dir: str):
        """Copy and modify config.json to make it look like a Qwen3-MoE config."""
        src_config_path = os.path.join(src_dir, "config.json")
        if not os.path.exists(src_config_path):
            logger.warning("config.json not found in %s, skipping", src_dir)
            return

        with open(src_config_path, "r") as f:
            config = json.load(f)

        # 提取 thinker_config 字段到顶层
        thinker_config = config.get("thinker_config", {})
        if thinker_config:
            dst_config = dict(thinker_config)
            # 复制顶层非子配置字段
            for key, value in config.items():
                if not key.endswith("_config") and key not in dst_config:
                    dst_config[key] = value
        else:
            dst_config = dict(config)

        # 覆盖为 mbridge 能识别的架构
        dst_config["architectures"] = ["Qwen3MoeForCausalLM"]
        dst_config["model_type"] = "qwen3_moe"

        dst_config_path = os.path.join(dst_dir, "config.json")
        with open(dst_config_path, "w") as f:
            json.dump(dst_config, f, indent=2)

        logger.info("Prepared config.json at %s", dst_config_path)


# ------------------------------------------------------------------
# OmniMindspeedMegatronBridgeEngine: 新 Megatron-Bridge 路径（推荐）
# ------------------------------------------------------------------


def _patch_torch_cuda_for_npu():
    """Patch specific ``torch.cuda`` functions that fail on NPU with NPU equivalents.

    Megatron-Core's ``set_random_seed`` calls ``torch.cuda.get_rng_state()``,
    which raises ``AssertionError: Torch not compiled with CUDA enabled`` on
    NPU.  Instead of replacing the entire ``torch.cuda`` module (which would
    break type annotations like ``torch.cuda.ExternalStream``), we patch only
    the functions that are actually called at runtime.

    Also creates a mock ``transformer_engine`` module so that
    Megatron-Bridge v0.5.0's PEFT/LoRA submodule (``import transformer_engine.pytorch as te``)
    can be imported on NPU without ``ModuleNotFoundError``.

    Returns ``True`` if the patch was applied, ``False`` otherwise.
    """
    if not torch.cuda.is_available() and hasattr(torch, "npu") and torch.npu.is_available():
        logger.info("Patching torch.cuda runtime functions → torch.npu for NPU compatibility")
        torch.cuda.get_rng_state = torch.npu.get_rng_state
        torch.cuda.set_rng_state = torch.npu.set_rng_state
        torch.cuda.manual_seed = torch.npu.manual_seed
        torch.cuda.manual_seed_all = torch.npu.manual_seed_all
        torch.cuda.current_device = torch.npu.current_device
        torch.cuda.device_count = torch.npu.device_count
        torch.cuda.synchronize = torch.npu.synchronize

        # Create mock transformer_engine module for Megatron-Bridge v0.5.0 PEFT imports
        if "transformer_engine" not in sys.modules:
            logger.info("Creating mock transformer_engine module for NPU compatibility")
            _te = ModuleType("transformer_engine")
            _te.pytorch = ModuleType("transformer_engine.pytorch")
            sys.modules["transformer_engine"] = _te
            sys.modules["transformer_engine.pytorch"] = _te.pytorch

        # Patch importlib.metadata.version() and also megatron.core.utils
        # directly, because megatron.core.utils is imported at module load time
        # (via the parent class chain) and its local `version` reference is
        # already captured — patching importlib.metadata.version alone won't
        # affect it.
        _orig_version = _ilm.version
        _patched_version = lambda name, _orig=_orig_version: (
            "1.7.0.dev0" if name == "transformer-engine" else _orig(name)
        )
        _ilm.version = _patched_version

        # Directly patch megatron.core.utils.get_te_version_str() since it
        # already captured the local `version` reference at import time.
        try:
            import megatron.core.utils as _mcore_utils
            _mcore_utils.get_te_version_str = lambda: "1.7.0.dev0"
        except (ImportError, AttributeError):
            pass

        return True
    return False


@EngineRegistry.register(model_type="omni_model", backend=["megatron"], device=["npu"])
class OmniMindspeedMegatronBridgeEngine(MindspeedEngineWithLMHead):
    """MindSpeed Megatron-Bridge engine for Qwen3-Omni Thinker on NPU.

    Uses the new Megatron-Bridge (``vanilla_mbridge=False``) which natively
    supports Qwen3-Omni architecture detection via
    ``AutoBridge.from_hf_pretrained()``.  No ``hf_config`` swapping or
    thinker weight extraction needed.

    Restriction
    -----------
    ``vanilla_mbridge`` must be ``False``.
    """

    def __init__(
        self,
        model_config: OmniModelConfig,
        engine_config: MindSpeedEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        if engine_config.vanilla_mbridge:
            raise ValueError(
                "OmniMindspeedMegatronBridgeEngine requires vanilla_mbridge=False "
                "(new Megatron-Bridge). Use OmniMindSpeedMegatronEngine for "
                "vanilla_mbridge=True (old mbridge)."
            )
        # Redirect torch.cuda → torch.npu before super().__init__() to avoid
        # AssertionError: Torch not compiled with CUDA enabled
        # when set_random_seed() calls torch.cuda.get_rng_state() on NPU.
        _patch_torch_cuda_for_npu()
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)