# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class Int_FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Interruptible FlowMatchEulerDiscreteScheduler.
    支持 replan_timesteps 功能。
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,  # shift =5.0
        use_dynamic_shifting=False,
        base_shift: Optional[float] = 0.5,  # base_shift =0.5
        max_shift: Optional[float] = 1.15,  # max_shift = 1.15
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item() # min=0.00498
        self.sigma_max = self.sigmas[0].item()  # max=1

        # [新增] 标记是否重规划
        self.is_rescheduled = False

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index
    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching (源码逻辑)
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            # 这里调用了上面的 index_for_timestep
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def scale_model_input(
        self,
        sample: torch.Tensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("Pass a value for `mu` when `use_dynamic_shifting` is True")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )
            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        
        # 正常初始化：sigmas 后面拼接一个 0.0
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None
        self.is_rescheduled = False

    def replan_timesteps(self, current_t: float, m_steps: int, device: Union[str, torch.device] = None):
        """
        [动态重规划]
        在 FlowMatch 中，t (或者 sigma) 是 0.0 到 1.0 的连续值 (或者 0-1000)。
        我们在这里接收的是 'timesteps' 单位的值 (例如 762.33)。
        """
        if m_steps < 1: m_steps = 1
        
        # 1. 直接在 [current_t, 0] 区间插值
        new_timesteps_np = np.linspace(current_t, 0, m_steps + 1).astype(np.float32)
        
        # 2. 更新 self.timesteps (去掉最后的 0)
        self.timesteps = torch.from_numpy(new_timesteps_np[:-1]).to(device)
        
        # 3. 关键：更新 self.sigmas
        # FlowMatch 的 step 函数严重依赖 self.sigmas[index] 和 self.sigmas[index+1]
        # FlowMatch 的本质公式是 x_prev = x_curr + (sigma_next - sigma_curr) * v
        # 这里的 (sigma_next - sigma_curr) 就是步长。
        # 我们必须把新生成的均匀时间点，算成 sigma，存入 self.sigmas。
        # 这样，step() 函数在计算时，就会使用新的“均匀步长”。
        new_sigmas = torch.from_numpy(new_timesteps_np).to(device) / self.config.num_train_timesteps
        self.sigmas = new_sigmas
        
        # 4. 重置 index 为 0 (因为新序列是从当前点开始的)
        self._step_index = 0
        self.is_rescheduled = True
        
        # 返回新的 timesteps (去掉第一个，因为第一个是当前正在跑的 t)
        return torch.from_numpy(new_timesteps_np[1:-1]).to(device)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        
        # 简单容错：找最近的值
        indices = (torch.abs(schedule_timesteps - timestep) < 1e-3).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item() if len(indices) > 0 else 0

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        
        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        # 核心逻辑：取 sigma 和 sigma_next
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps