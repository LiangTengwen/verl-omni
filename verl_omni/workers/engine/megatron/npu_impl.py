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

"""NPU-compatible patches for Megatron engine.

Monkey-patches ``verl.workers.engine.megatron.utils.set_random_seed`` at import
time to avoid calling ``torch.cuda.get_rng_state()``, which is not available
on Ascend NPU.

The original ``set_random_seed`` calls
``tensor_parallel.model_parallel_cuda_manual_seed(seed)``, which internally
invokes ``torch.cuda.get_rng_state()``. On NPU, ``torch.cuda`` is not compiled,
causing an ``AssertionError: Torch not compiled with CUDA enabled``.

This patch replaces the seed function with an NPU-safe version that uses
``torch.npu.manual_seed_all(seed)`` instead.

Also blocks ``megatron.bridge.diffusion.models`` import via a ``sys.meta_path``
finder.  ``megatron.bridge/__init__.py`` unconditionally imports this submodule
to register diffusion bridges, but the import chain eventually reaches
``megatron.bridge.models.qwen3_asr``, which registers a ``qwen3_asr`` model
type that conflicts with newer Transformers versions.  Since we only train
Qwen3.5 (not diffusion models), we short-circuit this import entirely.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys
import types
from typing import Callable

from verl.workers.engine.megatron import utils as _megatron_utils

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARN)

# ---------------------------------------------------------------------------
# Safety net: patch Transformers register methods directly.
# ---------------------------------------------------------------------------
# ``megatron.bridge`` imports ``megatron.bridge.models`` at module level,
# which triggers ``megatron.bridge.models.qwen3_asr.hf_qwen3_asr``, which
# calls ``AutoConfig/AutoModel/AutoProcessor.register("qwen3_asr", ...)``.
# Newer Transformers versions already include ``qwen3_asr``, so we patch
# the three classmethods to force ``exist_ok=True``.
# ---------------------------------------------------------------------------


def _patch_auto_registers() -> None:
    """Patch ``AutoConfig``, ``AutoModel``, and ``AutoProcessor`` to always use ``exist_ok=True``."""
    # --- Patch 1: CONFIG_MAPPING.register (used by AutoConfig) ---
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        _orig_cfg_register = CONFIG_MAPPING.register

        def _patched_cfg_register(key, value, exist_ok=False):
            return _orig_cfg_register(key, value, exist_ok=True)

        CONFIG_MAPPING.register = _patched_cfg_register
        logger.info("Patched CONFIG_MAPPING.register (exist_ok=True)")
    except Exception:
        logger.warning("Failed to patch CONFIG_MAPPING.register", exc_info=True)

    # --- Patch 2: AutoModel.register ---
    try:
        from transformers import AutoModel

        _orig_model_register = AutoModel.register.__func__

        def _patched_model_register(cls, config_class, model_class, exist_ok=False):
            return _orig_model_register(cls, config_class, model_class, exist_ok=True)

        AutoModel.register = classmethod(_patched_model_register)
        logger.info("Patched AutoModel.register (exist_ok=True)")
    except Exception:
        logger.warning("Failed to patch AutoModel.register", exc_info=True)

    # --- Patch 3: AutoProcessor.register ---
    try:
        from transformers import AutoProcessor

        _orig_proc_register = AutoProcessor.register.__func__

        def _patched_proc_register(cls, config_class, processor_class, exist_ok=False):
            return _orig_proc_register(cls, config_class, processor_class, exist_ok=True)

        AutoProcessor.register = classmethod(_patched_proc_register)
        logger.info("Patched AutoProcessor.register (exist_ok=True)")
    except Exception:
        logger.warning("Failed to patch AutoProcessor.register", exc_info=True)


# Apply at module level so patches are in effect before ``megatron.bridge``
# is imported (in the same process, including Ray actors).
_patch_auto_registers()

# ---------------------------------------------------------------------------
# sys.meta_path finder that blocks the diffusion models import.
#
# ``megatron.bridge/__init__.py`` (older versions) unconditionally does:
#   ``import megatron.bridge.diffusion.models  # registers diffusion bridges``
#
# This triggers the same ``qwen3_asr`` conflict.  We short-circuit the import
# so the patches above are never even tested.
# ---------------------------------------------------------------------------


class _DiffusionModelsBlocker:
    """sys.meta_path finder that blocks ``megatron.bridge.diffusion.models``.

    Returns a dummy ``ModuleSpec`` so the module is never actually loaded.
    Kept for compatibility with older Megatron-Bridge versions.
    """

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("megatron.bridge.diffusion.models"):
            from importlib.util import spec_from_loader

            dummy = types.ModuleType(fullname)
            dummy.__path__ = []
            dummy.__package__ = fullname

            class _DummyLoader:
                def create_module(self, spec):
                    return dummy

                def exec_module(self, module):
                    pass

            return spec_from_loader(fullname, _DummyLoader())
        return None


sys.meta_path.insert(0, _DiffusionModelsBlocker())

# ---------------------------------------------------------------------------
# Patch ModelProviderMixin.apply_overrides_and_finalize to filter out
# overrides that are not valid provider attributes.
#
# ``override_transformer_config`` values (e.g. ``use_flash_attn``) are
# passed verbatim to ``provider.apply_overrides_and_finalize()``, but the
# provider (e.g. ``Qwen35VLMoEModelProvider``) may not have all of them as
# dataclass fields.  This causes an ``AttributeError``.
#
# MindSpeed-Bridge's ``FinalizedProviderConfigFeature`` also fixes this,
# but it may not be activated in all setups (e.g. when the autoload .pth
# hook is absent).  This patch provides a belt-and-suspenders guarantee.
# ---------------------------------------------------------------------------


def _patch_apply_overrides_and_finalize(module: types.ModuleType) -> None:
    """Patch ``ModelProviderMixin.apply_overrides_and_finalize`` to skip unknown fields."""
    try:
        ModelProviderMixin = module.ModelProviderMixin
    except AttributeError:
        return

    _orig = ModelProviderMixin.apply_overrides_and_finalize

    def _filtered(self, dtype=None, overrides=None):
        if overrides is not None:
            overrides = {k: v for k, v in overrides.items() if hasattr(self, k)}
            if not overrides:
                overrides = None
        return _orig(self, dtype=dtype, overrides=overrides)

    ModelProviderMixin.apply_overrides_and_finalize = _filtered
    logger.info("Patched ModelProviderMixin.apply_overrides_and_finalize (skip unknown provider fields)")


class _ModelProviderPatcher(importlib.abc.MetaPathFinder):
    """sys.meta_path finder that patches ``ModelProviderMixin`` after module load."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "megatron.bridge.models.model_provider":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchedLoader(spec.loader)
        return spec


class _PatchedLoader(importlib.abc.Loader):
    """Loader wrapper that patches ``apply_overrides_and_finalize`` after exec."""

    def __init__(self, loader: importlib.abc.Loader):
        self.loader = loader

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module: types.ModuleType) -> None:
        self.loader.exec_module(module)
        _patch_apply_overrides_and_finalize(module)

    def get_code(self, fullname):
        get_code = getattr(self.loader, "get_code", None)
        return get_code(fullname) if get_code else None

    def get_source(self, fullname):
        get_source = getattr(self.loader, "get_source", None)
        return get_source(fullname) if get_source else None

    def get_filename(self, fullname):
        get_filename = getattr(self.loader, "get_filename", None)
        return get_filename(fullname) if get_filename else None


# Insert after the diffusion blocker so the blocker gets first shot at
# ``megatron.bridge.diffusion.models``, but our patcher intercepts
# ``megatron.bridge.models.model_provider``.
sys.meta_path.insert(1, _ModelProviderPatcher())


def _npu_safe_set_random_seed(seed: int) -> None:
    """NPU-safe replacement for ``verl.workers.engine.megatron.utils.set_random_seed``.

    Skips ``tensor_parallel.model_parallel_cuda_manual_seed`` (which calls
    ``torch.cuda.get_rng_state()``) and instead seeds the NPU RNG directly.
    """
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Seed the NPU device RNG.  torch.npu.manual_seed_all is the NPU
    # equivalent of torch.cuda.manual_seed_all.
    try:
        torch.npu.manual_seed_all(seed)
    except (AttributeError, RuntimeError):
        # NPU may not be initialized yet (e.g. before device mesh setup).
        pass


def _patch_set_random_seed() -> None:
    """Replace ``set_random_seed`` in the verl megatron utils module.

    IMPORTANT: We must also patch the local reference in
    ``verl.workers.engine.megatron.transformer_impl``, because that module
    imports ``set_random_seed`` via ``from .utils import set_random_seed``
    at module level (line 84), which creates a **local binding** in the
    ``transformer_impl`` module's namespace.  Patching
    ``utils.set_random_seed`` alone does NOT propagate to that local binding.
    """
    import sys

    original: Callable = _megatron_utils.set_random_seed
    if original is _npu_safe_set_random_seed:
        return  # Already patched.

    # 1. Patch the canonical location in the utils module.
    _megatron_utils.set_random_seed = _npu_safe_set_random_seed
    logger.info(
        "Patched verl.workers.engine.megatron.utils.set_random_seed to NPU-safe version "
        "(original=%s.%s)",
        getattr(original, "__module__", "?"),
        getattr(original, "__qualname__", "?"),
    )

    # 2. Also patch every module that imported ``set_random_seed`` via
    #    ``from .utils import set_random_seed``, so their local reference
    #    points to the NPU-safe version.
    _target_modules = [
        "verl.workers.engine.megatron.transformer_impl",
    ]
    for mod_name in _target_modules:
        mod = sys.modules.get(mod_name)
        if mod is not None and getattr(mod, "set_random_seed", None) is original:
            mod.set_random_seed = _npu_safe_set_random_seed
            logger.info("Also patched %s.set_random_seed (local reference)", mod_name)


# Apply the patch at import time so it is in effect before any Megatron engine
# is instantiated.  The ``verl_omni.workers.engine`` package is imported early
# in the training worker lifecycle, well before the ``EngineRegistry.new`` call.
_patch_set_random_seed()