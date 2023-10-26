import enum
from typing import Optional, Sequence, Tuple

import torch
import torch.utils.data
import numpy as np

from metakkl.logging import MetaTrainingConsoleLogger, TrainingConsoleLogger, LoggerGroup, \
    MetaTrainingTensorboardLogger, TrainingTensorboardLogger
from metakkl.system import DuffingOscillator
from metakkl.dataset import KKLObserverDataset, generate_merged_kkl_observer_data
from metakkl.nn import Normalizer, Denormalizer, KKLObserverNetwork, PDELoss
from metakkl.train import TaskSet, MetaTrainer, Trainer, TrainingDirection
from metakkl.utils import Noise

from scripts.utils import defaults


class OptimizerType(enum.Enum):
    SGD = enum.auto()
    ADAM = enum.auto()


def get_observer_init_state(model: KKLObserverNetwork, system_init_state: np.ndarray):
    with torch.no_grad():
        observer_init_state = model.forward_map(torch.from_numpy(system_init_state.astype(np.float32)))

    return observer_init_state


def generate_dataset(
        systems: Sequence[DuffingOscillator],
        system_init_states: Sequence[np.ndarray],
        system_output_noise: Optional[Noise],
        observer_init_state: Optional[np.ndarray],
        t0=defaults.DUFFING_TIME_START,
        tn=defaults.DUFFING_TIME_END,
        n_steps=defaults.DUFFING_NUM_STEPS
) -> KKLObserverDataset:
    observer = defaults.DUFFING_OBSERVER
    data = generate_merged_kkl_observer_data(systems=systems, system_init_states=system_init_states, observer=observer,
                                             z0=observer_init_state, t0=t0, tn=tn, n=n_steps,
                                             system_output_noise=system_output_noise)
    dataset = KKLObserverDataset(systems, observer, (len(systems) * len(system_init_states)), *data)
    return dataset


def create_model(dataset: KKLObserverDataset) -> KKLObserverNetwork:
    model = defaults.kkl_observer_model(system_state_size=dataset.x.shape[1], observer_state_size=dataset.z.shape[1])

    x_mean = torch.mean(dataset.x, dim=0)
    x_std = torch.std(dataset.x, dim=0)
    z_mean = torch.mean(dataset.z, dim=0)
    z_std = torch.std(dataset.z, dim=0)

    normalizer_x = Normalizer(mean=x_mean, std=x_std)
    denormalizer_x = Denormalizer(mean=x_mean, std=x_std)
    normalizer_z = Normalizer(mean=z_mean, std=z_std)
    denormalizer_z = Denormalizer(mean=z_mean, std=z_std)

    model.forward_map.normalizer = normalizer_x
    model.forward_map.denormalizer = denormalizer_z
    model.inverse_map.normalizer = normalizer_z
    model.inverse_map.denormalizer = denormalizer_x

    return model


def simulate_model(dataset: KKLObserverDataset, model: KKLObserverNetwork, forward: bool, inverse: bool) \
        -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    with torch.no_grad():
        observer_state_pred = model.forward_map(dataset.x) if forward else None
        system_state_pred = model.inverse_map(dataset.z) if inverse else None

    return system_state_pred, observer_state_pred


def train_model(
        dataset: KKLObserverDataset,
        model: KKLObserverNetwork,
        direction: TrainingDirection,
        merged: bool,
        output_loss: bool,
        num_epochs: int,
        batch_size: int,
        shuffle: bool,
        lr: float,
        pde_loss: bool,
        optim_type: OptimizerType,
        num_optim_steps: Optional[int],
        log_dir_path: Optional[str]
) -> Trainer:
    loss_function = torch.nn.MSELoss()
    pde_loss_function = PDELoss(observer=dataset.observer) if pde_loss else None

    if optim_type == OptimizerType.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim_type == OptimizerType.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=1,
                                                           threshold=1e-4)

    loggers = [TrainingConsoleLogger()]

    if log_dir_path is not None:
        tb_logger = TrainingTensorboardLogger(dir_path=log_dir_path)
        loggers.append(tb_logger)

    logger_group = LoggerGroup(loggers=loggers)

    with logger_group:
        trainer = Trainer(
            model=model,
            direction=direction,
            merged=merged,
            output_loss=output_loss,
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            pde_loss_function=pde_loss_function,
            logger=logger_group)
        trainer.train(num_optim_steps=num_optim_steps)

    return trainer


def meta_train(
        dataset: KKLObserverDataset,
        model: KKLObserverNetwork,
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
        log_dir_path: Optional[str]) -> MetaTrainer:
    task_set = TaskSet(dataset=dataset, batch_size=task_batch_size, shuffle_batch=shuffle_batch,
                       random_batch_offset=random_batch_offset)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    meta_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=meta_optimizer, mode='min', patience=100,
                                                                   threshold=1e-4, min_lr=1e-6)
    loss_function = torch.nn.MSELoss()

    loggers = [MetaTrainingConsoleLogger()]

    if log_dir_path is not None:
        tb_logger = MetaTrainingTensorboardLogger(dir_path=log_dir_path)
        loggers.append(tb_logger)

    logger_group = LoggerGroup(loggers=loggers)

    with logger_group:
        trainer = MetaTrainer(
            model=model,
            direction=direction,
            merged=merged,
            task_output_loss=task_output_loss,
            task_set=task_set,
            num_meta_iter=num_meta_iter,
            meta_batch_size=meta_batch_size,
            loss_function=loss_function,
            meta_optimizer=meta_optimizer,
            meta_lr_scheduler=meta_lr_scheduler,
            num_task_iter=num_task_iter,
            task_lr=task_lr,
            logger=logger_group)
        trainer.train()

    return trainer

