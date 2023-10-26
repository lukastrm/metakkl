import os
from collections.abc import Sequence
from typing import Optional

import torch
import numpy as np

from metakkl.system import DuffingOscillator, Noise
from metakkl.train import TrainingDirection

from scripts.utils import duffing
from scripts.utils.duffing import OptimizerType


def run(
        param_lmbda: Sequence[float],
        param_system_init_state: Sequence[np.ndarray],
        system_noise: Optional[Noise],
        system_output_noise: Optional[Noise],
        direction: TrainingDirection,
        merged: bool,
        num_epochs: int,
        batch_size: int,
        lr: float,
        pde_loss: bool,
        model_path: Optional[str],
        save_path: str,
        log_dir_path: Optional[str]
):
    systems = [DuffingOscillator(lmbda=a_i, noise=system_noise) for a_i in param_lmbda]
    dataset = duffing.generate_dataset(systems=systems, system_init_states=param_system_init_state,
                                       system_output_noise=system_output_noise, observer_init_state=None)

    if model_path is None:
        model = duffing.create_model(dataset=dataset)
    else:
        model = torch.load(model_path)

    duffing.train_model(
        dataset=dataset,
        model=model,
        direction=direction,
        merged=merged,
        output_loss=False,
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        lr=lr,
        pde_loss=pde_loss,
        optim_type=OptimizerType.ADAM,
        num_optim_steps=None,
        log_dir_path=log_dir_path
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
