import os
from typing import Optional, Sequence

import numpy as np
import torch

from metakkl.system import DuffingOscillator, Noise
from metakkl.train import TrainingDirection
from scripts.utils import duffing
from scripts.utils.duffing import OptimizerType


def run(
        param_lmbda: Sequence[float],
        param_system_init_state: Sequence[np.ndarray],
        system_noise: Optional[Noise],
        system_output_noise: Optional[Noise],
        observer_init_state: Optional[np.ndarray],
        direction: TrainingDirection,
        merged: bool,
        output_loss: bool,
        batch_size: int,
        shuffle: bool,
        lr: Optional[float],
        num_optim_steps: int,
        model_path: str,
        param_save_path: str
):
    if len(param_lmbda) == 0 or len(param_system_init_state) == 0:
        raise ValueError

    for lmbda in param_lmbda:
        model = torch.load(model_path)
        system = DuffingOscillator(lmbda=lmbda, noise=system_noise)

        for system_init_state in param_system_init_state:
            if observer_init_state is not None and observer_init_state.size == 0:
                observer_init_state = duffing.get_observer_init_state(model=model, system_init_state=system_init_state)

            dataset = duffing.generate_dataset(systems=(system,), system_init_states=(system_init_state,),
                                               system_output_noise=system_output_noise,
                                               observer_init_state=observer_init_state)

            if lr is None:
                lr = model.lr.data.item()

            duffing.train_model(
                dataset=dataset,
                model=model,
                direction=direction,
                merged=merged,
                output_loss=output_loss,
                num_epochs=1,
                batch_size=batch_size,
                shuffle=shuffle,
                lr=lr,
                pde_loss=False,
                optim_type=OptimizerType.SGD,
                num_optim_steps=num_optim_steps,
                log_dir_path=None
            )

            save_path = param_save_path.format(lmbda=lmbda, x0=system_init_state[0], x1=system_init_state[1])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model, save_path)
