import os
from typing import Optional

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from metakkl.nn import KKLObserverNetwork
from scripts.utils import duffing, style
from metakkl.system import Noise, DuffingOscillator


_NPZ_KEY_PARAM_SYSTEM_INIT_STATE_0 = 'param_system_init_state_0'
_NPZ_KEY_PARAM_SYSTEM_INIT_STATE_1 = 'param_system_init_state_1'
_NPZ_KEY_ERROR_MAP = 'error_map'


def _simulate_error(model: KKLObserverNetwork, lmbda: float, system_init_state: np.ndarray, system_noise: Optional[Noise],
                    system_output_noise: Optional[Noise], observer_init_state: Optional[np.ndarray]):
    if observer_init_state is not None and observer_init_state.size == 0:
        observer_init_state = duffing.get_observer_init_state(model=model, system_init_state=system_init_state)

    system = DuffingOscillator(lmbda=lmbda, noise=system_noise)
    dataset = duffing.generate_dataset(systems=(system,), system_init_states=(system_init_state,),
                                       system_output_noise=system_output_noise,
                                       observer_init_state=observer_init_state)
    t = dataset.t
    x_pred, _ = duffing.simulate_model(dataset=dataset, model=model, forward=False, inverse=True)
    error = torch.norm(x_pred - dataset.x, dim=1) / torch.norm(dataset.x, dim=1)
    return error, t


def simulate(
        lmbda: float,
        param_system_init_state_0: np.ndarray,
        param_system_init_state_1: np.ndarray,
        system_noise: Optional[Noise],
        system_output_noise: Optional[Noise],
        observer_init_state: Optional[np.ndarray],
        param_model_path: str,
        data_path: str
):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    error_map = np.zeros((len(param_system_init_state_0), len(param_system_init_state_1)))

    for i, x0 in enumerate(param_system_init_state_0):
        for j, x1 in enumerate(param_system_init_state_1):
            system_init_state = np.array((x0, x1))

            path = param_model_path.format(lmbda=lmbda, x0=x0, x1=x1)
            model = torch.load(path)
            error, t = _simulate_error(model=model, lmbda=lmbda, system_init_state=system_init_state, system_noise=system_noise,
                                       system_output_noise=system_output_noise, observer_init_state=observer_init_state)
            error_map[i, j] = torch.mean(error, dim=0)

    data = {
        _NPZ_KEY_PARAM_SYSTEM_INIT_STATE_0: param_system_init_state_0,
        _NPZ_KEY_PARAM_SYSTEM_INIT_STATE_1: param_system_init_state_1,
        _NPZ_KEY_ERROR_MAP: error_map
    }

    np.savez(data_path, **data)


def plot(
        data_path: str,
        show: bool,
        save_path: Optional[str],
        figsize
):
    data = np.load(data_path)
    param_system_init_state_0 = data[_NPZ_KEY_PARAM_SYSTEM_INIT_STATE_0]
    param_system_init_state_1 = data[_NPZ_KEY_PARAM_SYSTEM_INIT_STATE_1]
    error_map = data[_NPZ_KEY_ERROR_MAP]

    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.subplots()

    N = 256
    vals = np.ones((N, 4))
    split = int(N / 3)
    vals[:split, 0] = np.linspace(1, style.TU_BERLIN_ORANGE[0], split)
    vals[:split, 1] = np.linspace(1, style.TU_BERLIN_ORANGE[1], split)
    vals[:split, 2] = np.linspace(1, style.TU_BERLIN_ORANGE[2], split)
    vals[split:2 * split, 0] = np.linspace(style.TU_BERLIN_ORANGE[0], style.TU_BERLIN_RED[0], split)
    vals[split:2 * split, 1] = np.linspace(style.TU_BERLIN_ORANGE[1], style.TU_BERLIN_RED[1], split)
    vals[split:2 * split, 2] = np.linspace(style.TU_BERLIN_ORANGE[2], style.TU_BERLIN_RED[2], split)
    vals[2 * split:, 0] = np.linspace(style.TU_BERLIN_RED[0], style.TU_BERLIN_VIOLET[0], N - 2 * split)
    vals[2 * split:, 1] = np.linspace(style.TU_BERLIN_RED[1], style.TU_BERLIN_VIOLET[1], N - 2 * split)
    vals[2 * split:, 2] = np.linspace(style.TU_BERLIN_RED[2], style.TU_BERLIN_VIOLET[2], N - 2 * split)
    colormap = matplotlib.colors.ListedColormap(vals)

    pc = ax.pcolormesh(param_system_init_state_0, param_system_init_state_1, error_map, cmap=colormap,
                       rasterized=True)
    ax.grid(which='both', dashes=(1, 2))
    plt.colorbar(pc, label=r'Error $\bar{e}_t^*(T\sim x(0),e^*_x)$')
    ax.set_xlabel(r'$x_1(0)$')
    ax.set_ylabel(r'$x_2(0)$')

    plt.tight_layout(pad=0.0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(fname=save_path, dpi=400)

    if show:
        plt.show()
