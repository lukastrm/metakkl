from typing import Tuple, Optional, Callable, Sequence

import numpy as np
import torch.utils.data

from metakkl.system import System, Observer
from metakkl.utils import Noise


def _solve_rk4(function: Callable, t0: float, h: float, n: int, x0: np.ndarray, params: Optional[np.ndarray]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.zeros((n + 1, x0.shape[0]))
    x_dot = np.zeros((n + 1, x0.shape[0]))
    t = np.zeros(n + 1)
    x[0] = x0
    t[0] = t0

    xn = x0
    tn = t0

    for step in range(n):
        param = params[step] if params is not None else None
        k1 = function(tn, xn, param)
        k2 = function(tn + h / 2, xn + h / 2 * k1, param)
        k3 = function(tn + h / 2, xn + h / 2 * k2, param)
        k4 = function(tn + h, xn + h * k3, param)

        xn = xn + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        tn = tn + h

        x_dot[step] = k1
        x[step + 1] = xn
        t[step + 1] = tn

    x_dot[-1] = function(tn, xn, params[-1] if params is not None else None)

    return x, x_dot, t


def _calculate_t0_pre(gain: np.ndarray, z_max: int, e: float) -> int:
    w, v = np.linalg.eig(gain)
    min_ev = np.min(np.abs(np.real(w)))
    s = np.sqrt(z_max * gain.shape[0])
    t = 1 / min_ev * np.log(e / (np.linalg.cond(v) * s))
    return t


def generate_system_data(system: System, x0: np.ndarray, t0: float, h: float, n: int,
                         u: Optional[np.ndarray] = None, output_noise: Optional[Noise] = None) -> \
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:

    x, x_dot, t = _solve_rk4(function=system.function, t0=t0, h=h, n=n, x0=x0, params=u)
    system_output_indices = system.output_indices

    if len(system_output_indices) > 0:
        y = x[:, system_output_indices].copy()

        if output_noise is not None:
            y += np.random.normal(loc=output_noise[0], scale=output_noise[1], size=y.shape)
    else:
        y = None

    return x, x_dot, y, t


def generate_kkl_observer_data(system: System, x0: np.ndarray, observer: Observer, t0: float, tn: float, n: int,
                               z0: Optional[np.ndarray], system_output_noise: Optional[Noise] = None) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h = (tn - t0) / n
    x, x_dot, y, t = generate_system_data(system=system, x0=x0, t0=t0, h=h, n=n, output_noise=system_output_noise)

    if z0 is None:
        t0_pre = _calculate_t0_pre(observer.a, 10, 1e-6)
        n_pre = int(abs(t0_pre/h))
        _, _, y_pre, _ = generate_system_data(system=system, x0=x0, t0=t0, h=-h, n=n_pre)
        y_pre = np.flip(y_pre)
        z0_pre = np.random.rand(observer.state_size)
        z_pre, _, _, _ = generate_system_data(system=observer, x0=z0_pre, t0=t0_pre, h=h, n=n_pre, u=y_pre)
        z0 = z_pre[-1]

    z, _, _, _ = generate_system_data(system=observer, x0=z0, t0=t0, h=h, n=n, u=y)

    x = torch.from_numpy(x.astype(dtype=np.float32))
    x_dot = torch.from_numpy(x_dot.astype(dtype=np.float32))
    y = torch.from_numpy(y.astype(dtype=np.float32))
    z = torch.from_numpy(z.astype(dtype=np.float32))
    t = torch.from_numpy(t.astype(dtype=np.float32))

    return x, x_dot, y, z, t


def generate_merged_kkl_observer_data(systems: Sequence[System], system_init_states: Sequence[np.ndarray],
                                      observer: Observer, z0: Optional[np.ndarray], t0: float, tn: float, n: int,
                                      system_output_noise: Optional[Noise] = None) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_list = []
    x_dot_list = []
    y_list = []
    z_list = []
    t_list = []

    for system in systems:
        for x0 in system_init_states:
            x, x_dot, y, z, t = generate_kkl_observer_data(system=system, x0=x0, observer=observer, z0=z0, t0=t0, tn=tn,
                                                           n=n, system_output_noise=system_output_noise)
            x_list.append(x)
            x_dot_list.append(x_dot)
            y_list.append(y)
            z_list.append(z)
            t_list.append(t)

    x_merged = torch.cat(x_list, dim=0)
    x_dot_merged = torch.cat(x_dot_list, dim=0)
    y_merged = torch.cat(y_list, dim=0)
    z_merged = torch.cat(z_list, dim=0)
    t_merged = torch.cat(t_list, dim=0)

    return x_merged, x_dot_merged, y_merged, z_merged, t_merged


class KKLObserverDataset(torch.utils.data.Dataset):
    def __init__(self, systems: Sequence[System], observer: Observer, num_tasks: int, x: torch.Tensor,
                 x_dot: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor):
        self.systems: Sequence[System] = systems
        self.observer: Observer = observer
        self.num_tasks: int = num_tasks
        self.task_size: int = int(x.shape[0] / num_tasks)
        self.x: torch.Tensor = x
        self.x_dot: torch.Tensor = x_dot
        self.y: torch.Tensor = y
        self.z: torch.Tensor = z
        self.t: torch.Tensor = t

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.x_dot[index], self.y[index], self.z[index], self.t[index]
