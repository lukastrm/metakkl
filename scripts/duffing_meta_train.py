import os
from collections.abc import Sequence
from typing import Optional

import torch
import numpy as np

from metakkl.system import DuffingOscillator, Noise
from metakkl.train import TrainingDirection
from scripts.utils import duffing


def run(
        param_lmbda: Sequence[float],
        param_system_init_state: Sequence[np.ndarray],
        system_noise: Optional[Noise],
        system_output_noise: Optional[Noise],
        direction: TrainingDirection,
        merged: bool,
        task_output_loss: bool,
        meta_batch_size: int,
        meta_lr: float,
        num_meta_iter: int,
        task_batch_size: int,
        random_batch_offset: bool,
        shuffle_batch: bool,
        task_lr: Optional[float],
        num_task_iter: int,
        model_path: Optional[str],
        save_path: str,
        log_dir_path: Optional[str]
):
    systems = [DuffingOscillator(lmbda=lmbda, noise=system_noise) for lmbda in param_lmbda]
    dataset = duffing.generate_dataset(systems=systems, system_init_states=param_system_init_state,
                                       system_output_noise=system_output_noise, observer_init_state=None)

    if model_path is None:
        model = duffing.create_model(dataset=dataset)
    else:
        model = torch.load(model_path)

    duffing.meta_train(
        dataset=dataset,
        model=model,
        direction=direction,
        merged=merged,
        task_output_loss=task_output_loss,
        meta_batch_size=meta_batch_size,
        meta_lr=meta_lr,
        num_meta_iter=num_meta_iter,
        task_batch_size=task_batch_size,
        random_batch_offset=random_batch_offset,
        shuffle_batch=shuffle_batch,
        task_lr=task_lr,
        num_task_iter=num_task_iter,
        log_dir_path=log_dir_path
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
