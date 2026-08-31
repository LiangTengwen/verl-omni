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
"""

from __future__ import annotations

import logging
from typing import Callable

from verl.workers.engine.megatron import utils as _megatron_utils

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARN)


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