# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU worker for Ambarella external accelerator models.

This module provides a specialized CPU worker that handles models running
on external Ambarella hardware. Since the KV cache is managed by the
Ambarella device, this worker skips vLLM's KV cache allocation to save memory.
"""
from typing import List, Optional, Tuple, Type

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.worker.cpu_model_runner import CPUModelAmbaRunner, CPUModelRunnerBase
from vllm.worker.cpu_pooling_model_runner import CPUPoolingModelAmbaRunner
from vllm.worker.cpu_worker import CPUWorker

logger = init_logger(__name__)

VLLM_AMBA_EXT_VERSION = "20260601"

class CPUWorkerAmba(CPUWorker):
    """A specialized CPU worker for Ambarella external accelerator models.

    This worker overrides KV cache related methods to avoid memory allocation,
    since the actual KV cache is managed by the external Ambarella device.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        model_runner_cls: Optional[Type[CPUModelRunnerBase]] = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            model_runner_cls=model_runner_cls,
        )
        logger.info("CPUWorkerAmba initialized - KV cache managed by Ambarella device, "
            f"vLLM_AMBA_EXT_VERSION: {VLLM_AMBA_EXT_VERSION}")

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Return virtual blocks for scheduler without actual memory allocation."""
        block_size = self.cache_config.block_size or 16
        max_model_len = self.model_config.max_model_len

        # Calculate enough virtual blocks to handle max_model_len tokens
        num_virtual_blocks = (max_model_len // block_size + 1) * 4

        logger.info(f"CPUWorkerAmba: Returning {num_virtual_blocks} virtual blocks "
                   f"(no actual memory allocation)")

        return num_virtual_blocks, 0

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize cache with placeholder structures only."""
        logger.info(f"CPUWorkerAmba: initialize_cache with "
                   f"num_gpu_blocks={num_gpu_blocks} (virtual)")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()

    def _validate_num_cpu_blocks(self, num_cpu_blocks: int) -> None:
        """Skip validation for Amba models."""
        return

    def _init_cache_engine(self) -> None:
        """Initialize placeholder cache structures (no memory allocation)."""
        logger.info("CPUWorkerAmba: Creating placeholder cache engine")

        # Create empty placeholder structures
        # Must have at least pipeline_parallel_size elements for virtual_engine indexing
        pp_size = self.parallel_config.pipeline_parallel_size

        self.cache_engine = []
        # cpu_cache needs to be a list of lists, one per virtual engine
        # Each inner list would normally contain KV cache tensors per layer
        # For Amba, we use empty lists since KV cache is on the device
        self.cpu_cache = [[] for _ in range(pp_size)]

        if hasattr(self.model_runner, 'block_size'):
            self.model_runner.block_size = self.cache_config.block_size or 16

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        """Return placeholder KV cache structure."""
        return self.cpu_cache
