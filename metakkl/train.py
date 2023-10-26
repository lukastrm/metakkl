import enum
from typing import Callable, Optional, Iterable

import numpy as np
import torch
import torch.utils.data

from metakkl.dataset import KKLObserverDataset
from metakkl.nn import KKLObserverNetwork, PDELoss, Normalizer
from metakkl.logging import MetaTrainingLogRecord, MetaTrainingLogger, TrainingLogger, TrainingLogRecord
from metakkl.utils import clone_module, update_module

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class TrainingDirection(enum.Enum):
    UNIDIRECTIONAL_FORWARD = enum.auto()
    UNIDIRECTIONAL_INVERSE = enum.auto()
    BIDIRECTIONAL = enum.auto()


class TaskSet:
    def __init__(self, dataset: KKLObserverDataset, batch_size: int, shuffle_batch: bool, random_batch_offset: bool):
        self.dataset: KKLObserverDataset = dataset
        self.batch_size: int = batch_size
        self.shuffle_batch: bool = shuffle_batch
        self.random_batch_offset: bool = random_batch_offset

    def sample(self, num_batches: int) -> Iterable:
        task_idx = np.random.randint(self.dataset.num_tasks)
        start_idx = task_idx * self.dataset.task_size
        stop_idx = start_idx + self.dataset.task_size

        if self.random_batch_offset:
            sample_size = (num_batches * self.batch_size)
            offset = np.random.randint(self.dataset.task_size - sample_size)
            start_idx += offset
            stop_idx = start_idx + sample_size

        subset = torch.utils.data.Subset(dataset=self.dataset, indices=list(range(start_idx, stop_idx)))
        data_loader = torch.utils.data.DataLoader(dataset=subset, batch_size=self.batch_size,
                                                  shuffle=self.shuffle_batch)
        return data_loader


class Trainer:
    def __init__(
            self,
            model: KKLObserverNetwork,
            direction: TrainingDirection,
            merged: bool,
            output_loss: bool,
            dataset: KKLObserverDataset,
            shuffle: bool,
            batch_size: int,
            num_epochs: int,
            loss_function: LossFunction,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
            pde_loss_function: Optional[PDELoss] = None,
            lmbda: float = 1.0,
            logger: Optional[TrainingLogger] = None
    ):
        self.model: KKLObserverNetwork = model
        self.direction: TrainingDirection = direction
        self.merged: bool = merged
        self.output_loss: bool = output_loss
        self.dataset = dataset
        self.data_loader: torch.utils.data.DataLoader = \
            torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        self.num_epochs: int = num_epochs
        self.loss_function: LossFunction = loss_function
        self.pde_loss_function: Optional[PDELoss] = pde_loss_function
        self.lmbda: float = lmbda
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = scheduler
        self.logger: Optional[TrainingLogger] = logger

    def train(self, num_optim_steps: Optional[int] = None):
        try:
            self._train_impl(num_optim_steps=num_optim_steps)
        except KeyboardInterrupt:
            return

    def _train_impl(self, num_optim_steps: Optional[int] = None):
        y_ind = self.dataset.systems[0].output_indices

        normalizer_x = self.model.forward_map.normalizer
        normalizer_y = Normalizer(normalizer_x.mean[y_ind], normalizer_x.std[y_ind])
        normalizer_z = self.model.inverse_map.normalizer

        for epoch in range(self.num_epochs):
            loss = 0.0
            loss_x = 0.0
            loss_y = 0.0
            loss_z = 0.0
            batch_idx = 0

            for batch_idx, batch in enumerate(self.data_loader):
                if num_optim_steps is not None and num_optim_steps <= batch_idx:
                    break

                x_label, x_dot_label, y_label, z_label, _ = batch

                loss_batch = torch.tensor(0.0)

                if self.direction != TrainingDirection.UNIDIRECTIONAL_INVERSE:
                    z_pred = self.model.forward_map(x_label)

                    z_label_norm = normalizer_z(z_label)
                    z_pred_norm = normalizer_z(z_pred)

                    loss_batch_z = self.loss_function(z_pred_norm, z_label_norm)

                    if self.pde_loss_function is not None:
                        pde_loss = self.pde_loss_function(self.model.forward_map, x_label, x_dot_label, y_label, z_pred)
                        loss_batch_z += self.lmbda * pde_loss

                    loss_batch += loss_batch_z
                    loss_z += loss_batch_z.item()
                elif self.merged:
                    with torch.no_grad():
                        z_pred = self.model.forward_map(x_label)
                else:
                    z_pred = None

                if self.direction != TrainingDirection.UNIDIRECTIONAL_FORWARD:
                    if self.merged:
                        x_pred = self.model.inverse_map(z_pred)
                    else:
                        x_pred = self.model.inverse_map(z_label)

                    x_label_norm = normalizer_x(x_label)
                    x_pred_norm = normalizer_x(x_pred)
                    y_label_norm = normalizer_y(y_label)
                    y_pred_norm = x_pred_norm[:, y_ind]

                    loss_batch_x = self.loss_function(x_pred_norm, x_label_norm)
                    loss_batch_y = self.loss_function(y_pred_norm, y_label_norm)

                    if self.output_loss:
                        loss_batch += loss_batch_y
                    else:
                        loss_batch += loss_batch_x

                    loss_x += loss_batch_x.item()
                    loss_y += loss_batch_y.item()

                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

                loss += loss_batch.item()

            loss /= (batch_idx + 1)
            loss_x /= (batch_idx + 1)
            loss_y /= (batch_idx + 1)
            loss_z /= (batch_idx + 1)

            self.scheduler.step(loss)

            if self.logger is not None:
                lr = self.optimizer.param_groups[0]['lr']
                record = TrainingLogRecord(idx=epoch, loss=loss, loss_x=loss_x, loss_y=loss_y, loss_z=loss_z, lr=lr)
                self.logger.log(record)


class MetaTrainer:
    def __init__(
            self,
            model: KKLObserverNetwork,
            direction: TrainingDirection,
            merged: bool,
            task_output_loss: bool,
            task_set: TaskSet,
            num_meta_iter: int,
            meta_batch_size: int,
            loss_function: LossFunction,
            meta_optimizer: torch.optim.Optimizer,
            meta_lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
            num_task_iter: int,
            task_lr: Optional[float],
            logger: Optional[MetaTrainingLogger] = None
    ):
        self.model: KKLObserverNetwork = model
        self.direction: TrainingDirection = direction
        self.merged: bool = merged
        self.task_output_loss: bool = task_output_loss
        self.task_set: TaskSet = task_set
        self.num_meta_iter: int = num_meta_iter
        self.meta_batch_size: int = meta_batch_size
        self.loss_function: LossFunction = loss_function
        self.meta_optimizer: torch.optim.Optimizer = meta_optimizer
        self.meta_lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = meta_lr_scheduler
        self.num_task_iter: int = num_task_iter
        self.task_lr: Optional[float] = task_lr
        self.logger: Optional[MetaTrainingLogger] = logger

        self.meta_batch_loss = torch.tensor(float('inf'))

    def train(self):
        try:
            self._train_impl()
        except KeyboardInterrupt:
            return

    def _train_impl(self):
        train_forward = self.direction != TrainingDirection.UNIDIRECTIONAL_INVERSE
        train_inverse = self.direction != TrainingDirection.UNIDIRECTIONAL_FORWARD

        y_ind = self.task_set.dataset.systems[0].output_indices

        normalizer_x = self.model.forward_map.normalizer
        normalizer_y = Normalizer(normalizer_x.mean[y_ind], normalizer_x.std[y_ind])
        normalizer_z = self.model.inverse_map.normalizer

        for meta_iter_idx in range(self.num_meta_iter):
            loss_meta_batch = torch.tensor(0.0)
            loss_meta_batch_x = torch.tensor(0.0)
            loss_meta_batch_z = torch.tensor(0.0)
            loss_task_min = float('inf')
            loss_task_max = 0.0

            for meta_batch_idx in range(self.meta_batch_size):
                data_loader = self.task_set.sample(num_batches=self.num_task_iter + 1)
                data_loader_iter = iter(data_loader)
                task_model: KKLObserverNetwork = clone_module(self.model)
                loss_task = 0.0

                for task_iter_idx in range(self.num_task_iter):
                    x_update_label, _, y_update_label, z_update_label, _, = next(data_loader_iter)

                    loss_task_update = torch.tensor(0.0)

                    if train_forward:
                        z_update_pred = task_model.forward_map(x_update_label)

                        z_update_label_norm = normalizer_z(z_update_label)
                        z_update_pred_norm = normalizer_z(z_update_pred)

                        loss_task_update_z = self.loss_function(z_update_pred_norm, z_update_label_norm)
                        loss_task_update += loss_task_update_z
                    elif self.merged:
                        with torch.no_grad():
                            z_update_pred = task_model.forward_map(x_update_label)
                    else:
                        z_update_pred = None

                    if train_inverse:
                        if self.merged:
                            x_update_pred = task_model.inverse_map(z_update_pred)
                        else:
                            x_update_pred = task_model.inverse_map(z_update_label)

                        x_update_label_norm = normalizer_x(x_update_label)
                        x_update_pred_norm = normalizer_x(x_update_pred)
                        y_update_label_norm = normalizer_y(y_update_label)
                        y_update_pred_norm = x_update_pred_norm[:, y_ind]

                        loss_task_update_x = self.loss_function(x_update_pred_norm, x_update_label_norm)
                        loss_task_update_y = self.loss_function(y_update_pred_norm, y_update_label_norm)

                        if self.task_output_loss:
                            loss_task_update += loss_task_update_y
                        else:
                            loss_task_update += loss_task_update_x

                    model_params = [param for (name, param) in task_model.named_parameters() if name != 'lr']
                    allow_unused = self.direction != TrainingDirection.BIDIRECTIONAL
                    grad = torch.autograd.grad(loss_task_update, model_params, create_graph=True, retain_graph=True,
                                               allow_unused=allow_unused)

                    for param, grad in zip(model_params, grad):
                        if grad is None and allow_unused:
                            continue

                        lr = task_model.lr if self.task_lr is None else self.task_lr
                        param.update = -(lr * grad)

                    update_module(module=task_model)

                x_query_label, x_dot_query_label, y_query_label, z_query_label, _ = next(data_loader_iter)

                if train_forward:
                    z_query_pred = task_model.forward_map(x_query_label)

                    z_query_label_norm = normalizer_z(z_query_label)
                    z_query_pred_norm = normalizer_z(z_query_pred)

                    loss_task_query_z = self.loss_function(z_query_pred_norm, z_query_label_norm)
                    loss_meta_batch_z += loss_task_query_z
                    loss_task += loss_task_query_z.item()
                elif self.merged:
                    with torch.no_grad():
                        z_query_pred = task_model.forward_map(x_query_label)
                else:
                    z_query_pred = None

                if train_inverse:
                    if self.merged:
                        x_query_pred = task_model.inverse_map(z_query_pred)
                    else:
                        x_query_pred = task_model.inverse_map(z_query_label)

                    x_query_label_norm = normalizer_x(x_query_label)
                    x_query_pred_norm = normalizer_x(x_query_pred)

                    loss_task_query_x = self.loss_function(x_query_pred_norm, x_query_label_norm)
                    loss_meta_batch_x += loss_task_query_x
                    loss_task += loss_task_query_x.item()

                loss_task_min = min(loss_task_min, loss_task)
                loss_task_max = max(loss_task_max, loss_task)

            loss_meta_batch_x /= self.meta_batch_size
            loss_meta_batch_z /= self.meta_batch_size

            if train_forward:
                loss_meta_batch += loss_meta_batch_z

            if train_inverse:
                loss_meta_batch += loss_meta_batch_x

            self.meta_optimizer.zero_grad()
            loss_meta_batch.backward()
            self.meta_optimizer.step()

            self.meta_lr_scheduler.step(loss_meta_batch.item())

            if self.logger is not None:
                lr_task = self.model.lr.data.item() if self.task_lr is None else self.task_lr
                lr_meta = self.meta_optimizer.param_groups[0]['lr']

                record = MetaTrainingLogRecord(
                    idx=meta_iter_idx,
                    loss_meta_batch=loss_meta_batch.item(),
                    loss_meta_batch_x=loss_meta_batch_x.item(),
                    loss_meta_batch_z=loss_meta_batch_z.item(),
                    loss_task_query_min=loss_task_min,
                    loss_task_query_max=loss_task_max,
                    lr_task=lr_task,
                    lr_meta=lr_meta)
                self.logger.log(record)

            self.meta_batch_loss = loss_meta_batch
