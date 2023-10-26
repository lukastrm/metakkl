import os
from typing import Optional, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt

from metakkl.nn import KKLObserverNetwork
from scripts.utils import duffing
from metakkl.system import Noise, DuffingOscillator


_NPZ_KEY_T = 't'
_NPZ_KEY_ERROR_MEAN_LIST = 'error_mean_list'
_NPZ_KEY_ERROR_STD_LIST = 'error_std_list'


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
        param_system_init_state: Sequence[np.ndarray],
        system_noise: Optional[Noise],
        system_output_noise: Optional[Noise],
        observer_init_state: Optional[np.ndarray],
        model_paths: Sequence[str],
        param_model_paths: Sequence[str],
        data_path: str
):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    t = None
    error_mean_list = []
    error_std_list = []

    for path in model_paths:
        model = torch.load(path)
        model_error = []

        for system_init_state in param_system_init_state:
            error, t = _simulate_error(model=model, lmbda=lmbda, system_init_state=system_init_state, system_noise=system_noise,
                                       system_output_noise=system_output_noise, observer_init_state=observer_init_state)
            model_error.append(error)

        error = np.array(model_error)
        error_mean = np.mean(error, axis=0)
        error_std = np.std(error, axis=0)

        error_mean_list.append(error_mean)
        error_std_list.append(error_std)

    for param_path in param_model_paths:
        model_error = []

        for system_init_state in param_system_init_state:
            path = param_path.format(a=lmbda, x0=system_init_state[0], x1=system_init_state[1])
            model = torch.load(path)
            error, t = _simulate_error(model=model, lmbda=lmbda, system_init_state=system_init_state, system_noise=system_noise,
                                       system_output_noise=system_output_noise, observer_init_state=observer_init_state)
            model_error.append(error)

        error = np.array(model_error)
        error_mean = np.mean(error, axis=0)
        error_std = np.std(error, axis=0)

        print(error_mean.mean())

        error_mean_list.append(error_mean)
        error_std_list.append(error_std)

    data = {
        _NPZ_KEY_T: t,
        _NPZ_KEY_ERROR_MEAN_LIST: error_mean_list,
        _NPZ_KEY_ERROR_STD_LIST: error_std_list
    }

    np.savez(data_path, **data)


def plot(
        data_path: str,
        show: bool,
        save_path: Optional[str],
        figsize,
        colors,
        xlabel,
        ylabel
):
    data = np.load(data_path)
    t = data[_NPZ_KEY_T]
    error_mean_list = data[_NPZ_KEY_ERROR_MEAN_LIST]

    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.subplots()

    for i, error_mean in enumerate(error_mean_list):
        ax.plot(t, error_mean, linewidth=0.8, color=colors[i])

    ax.margins(x=0.0)
    ax.grid(which='both', dashes=(1, 2))
    ax.tick_params(which='both', direction='in')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout(pad=0.0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(fname=save_path)

    if show:
        plt.show()
