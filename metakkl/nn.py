from typing import Tuple, Optional, cast

import numpy as np
import torch
from torch.autograd import function

from metakkl.system import Observer


class Normalizer(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


class Denormalizer(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor) -> torch.Tensor:
        return (tensor * self.std) + self.mean


class PDELoss(torch.nn.Module):
    def __init__(self, observer: Observer):
        super().__init__()
        self.observer: Observer = observer

    def forward(self, model: torch.nn.Module, x_label: torch.Tensor, x_dot_label: torch.Tensor, y_label: torch.Tensor,
                z_pred: torch.Tensor) -> torch.Tensor:
        a = torch.from_numpy(self.observer.a.astype(dtype=np.float32))
        b = torch.from_numpy(self.observer.b.astype(dtype=np.float32))

        dtdx = torch.autograd.functional.jacobian(cast(function, model), x_label)
        idx = torch.arange(x_label.shape[0])
        dtdx = dtdx[idx, :, idx, :]

        dtdx_mul_f = torch.bmm(dtdx, torch.unsqueeze(x_dot_label, 2))

        m_mul_t = torch.matmul(a, torch.unsqueeze(z_pred, 2))
        k_mul_h = torch.matmul(b, torch.unsqueeze(y_label, 2))

        pde = dtdx_mul_f - m_mul_t - k_mul_h
        loss_batch = torch.linalg.norm(pde, dim=1)
        loss_pde = torch.mean(loss_batch)

        return loss_pde


class MappingNetwork(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, n_hidden_layers: int, hidden_layer_size: int,
                 normalizer: Optional[Normalizer] = None, denormalizer: Optional[Denormalizer] = None):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        layer_size = input_size

        for _ in range(n_hidden_layers):
            self.layers.append(torch.nn.Linear(layer_size, hidden_layer_size))
            layer_size = hidden_layer_size

        self.layers.append(torch.nn.Linear(layer_size, output_size))
        self.normalizer = normalizer
        self.denormalizer = denormalizer

    def forward(self, tensor: torch.Tensor) -> Tuple:
        if self.normalizer is not None:
            tensor = self.normalizer(tensor)

        activation = torch.nn.ReLU()

        for layer in self.layers[:-1]:
            tensor = activation(layer(tensor))

        tensor = self.layers[-1](tensor)

        if self.denormalizer is not None:
            tensor = self.denormalizer(tensor)

        return tensor


class KKLObserverNetwork(torch.nn.Module):
    def __init__(self, forward_map: MappingNetwork, inverse_map: MappingNetwork):
        super().__init__()
        self.forward_map: MappingNetwork = forward_map
        self.inverse_map: MappingNetwork = inverse_map
        self.lr: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(1e-6))

    def forward(self, system_state: torch.Tensor, observer_state: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        raise DeprecationWarning
