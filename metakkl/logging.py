import abc
import os
from dataclasses import dataclass
from typing import Iterable, TypeVar, Generic

import torch.utils.tensorboard as tensorboard

LogRecordT = TypeVar('LogRecordT')


@dataclass
class TrainingLogRecord:
    idx: int
    loss: float
    loss_x: float
    loss_y: float
    loss_z: float
    lr: float


@dataclass
class MetaTrainingLogRecord:
    idx: int
    loss_meta_batch: float
    loss_meta_batch_x: float
    loss_meta_batch_z: float
    loss_task_query_min: float
    loss_task_query_max: float
    lr_task: float
    lr_meta: float


class Logger(abc.ABC, Generic[LogRecordT]):
    def log(self, record: LogRecordT):
        raise NotImplementedError


class LoggerGroup(Logger[LogRecordT]):
    def __init__(self, loggers: Iterable[Logger[LogRecordT]]):
        self.loggers: Iterable[Logger[LogRecordT]] = loggers

    def log(self, record: LogRecordT):
        for logger in self.loggers:
            logger.log(record=record)

    def __enter__(self):
        for logger in self.loggers:
            if hasattr(logger, '__enter__'):
                logger.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for logger in self.loggers:
            if hasattr(logger, '__exit__'):
                logger.__exit__(exc_type, exc_val, exc_tb)


class ConsoleLogger(Logger[LogRecordT]):
    def __init__(self, keys: Iterable[str]):
        self.keys: Iterable[str] = keys

    # noinspection PyMethodMayBeStatic
    def log(self, record: LogRecordT):
        components = []

        for key in self.keys:
            value = record.__dict__[key]

            if isinstance(value, int):
                components.append(f'{key}: {value:06d}')
            elif isinstance(value, float):
                components.append(f'{key}: {value:10.6f}')
            else:
                raise RuntimeError

        msg = ', '.join(components)
        print(msg)


class TrainingConsoleLogger(ConsoleLogger[TrainingLogRecord]):
    def __init__(self):
        keys = ['idx', 'loss_x', 'loss_z']
        super().__init__(keys)


class MetaTrainingConsoleLogger(ConsoleLogger[MetaTrainingLogRecord]):
    def __init__(self):
        keys = ['idx', 'loss_meta_batch_x', 'loss_meta_batch_z', 'lr_task']
        super().__init__(keys)


class TensorboardLogger(Logger[LogRecordT]):
    def __init__(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        self.tb_writer: tensorboard.SummaryWriter = tensorboard.SummaryWriter(log_dir=dir_path)

    def __enter__(self):
        self.tb_writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.tb_writer.close()


class TrainingTensorboardLogger(TensorboardLogger[TrainingLogRecord]):
    def log(self, record: TrainingLogRecord):
        self.tb_writer.add_scalar('loss/loss', record.loss, record.idx)
        self.tb_writer.add_scalar('loss/loss_x', record.loss_x, record.idx)
        self.tb_writer.add_scalar('loss/loss_y', record.loss_y, record.idx)
        self.tb_writer.add_scalar('loss/loss_z', record.loss_z, record.idx)
        self.tb_writer.add_scalar('lr/lr', record.lr, record.idx)


class MetaTrainingTensorboardLogger(TensorboardLogger[MetaTrainingLogRecord]):
    def log(self, record: MetaTrainingLogRecord):

        self.tb_writer.add_scalar('loss/loss_meta_batch', record.loss_meta_batch, record.idx)
        self.tb_writer.add_scalar('loss/loss_meta_batch_x', record.loss_meta_batch_x, record.idx)
        self.tb_writer.add_scalar('loss/loss_meta_batch_z', record.loss_meta_batch_z, record.idx)
        self.tb_writer.add_scalar('loss/loss_task_query_min', record.loss_task_query_min, record.idx)
        self.tb_writer.add_scalar('loss/loss_task_query_max', record.loss_task_query_max, record.idx)
        self.tb_writer.add_scalar('lr/lr_task', record.lr_task, record.idx)
        self.tb_writer.add_scalar('lr/lr_meta', record.lr_meta, record.idx)


TrainingLogger = Logger[TrainingLogRecord]
MetaTrainingLogger = Logger[MetaTrainingLogRecord]
