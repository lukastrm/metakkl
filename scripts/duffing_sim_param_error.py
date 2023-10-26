import os
from typing import Optional, Sequence, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np

from metakkl.dataset import KKLObserverDataset
from metakkl.nn import KKLObserverNetwork
from metakkl.system import Noise, DuffingOscillator
from scripts.utils import duffing, style

_NPZ_KEY_PARAM_LMBDA = 'param_lmbda'
_NPZ_KEY_PARAM_ERROR_X_LIST = 'param_error_x_list'
_NPZ_KEY_PARAM_ERROR_Z_LIST = 'param_error_z_list'


def _simulate_model_error(model: KKLObserverNetwork, dataset: Optional[KKLObserverDataset], forward: bool,
                          inverse: bool) -> Tuple[float, float]:
    x_pred, z_pred = duffing.simulate_model(dataset=dataset, model=model, forward=forward, inverse=inverse)

    error_x = torch.mean(torch.norm(x_pred - dataset.x, dim=1) / torch.norm(dataset.x, dim=1)).item() if inverse \
        else None
    error_z = torch.mean(torch.norm(z_pred - dataset.z, dim=1) / torch.norm(dataset.z, dim=1)).item() if forward \
        else None

    return error_x, error_z


def simulate(
        param_lmbda: Sequence[float],
        system_init_state: np.ndarray,
        system_noise: Optional[Noise],
        system_output_noise: Optional[Noise],
        observer_init_state: Optional[np.ndarray],
        model_paths: Optional[Sequence[str]],
        param_model_paths: Optional[Sequence[str]],
        sim_error_x: bool,
        sim_error_z: bool,
        data_path: str
):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    static_models = []

    if model_paths is not None:
        for path in model_paths:
            model = torch.load(path)
            static_models.append(model)

    param_error_x_list = np.ndarray((len(model_paths) + len(param_model_paths), len(param_lmbda)))
    param_error_z_list = np.ndarray((len(model_paths) + len(param_model_paths), len(param_lmbda)))

    for i, lmbda in enumerate(param_lmbda):
        system = DuffingOscillator(lmbda=lmbda, noise=system_noise)

        if observer_init_state is None or observer_init_state.size > 0:
            dataset = duffing.generate_dataset(systems=(system,), system_init_states=(system_init_state,),
                                               system_output_noise=system_output_noise,
                                               observer_init_state=observer_init_state)
        else:
            dataset = None

        for j, model in enumerate(static_models):
            if dataset is None:
                _observer_init_state = duffing.get_observer_init_state(model=model, system_init_state=system_init_state)
                dataset = duffing.generate_dataset(systems=(system,), system_init_states=(system_init_state,),
                                                   system_output_noise=system_output_noise,
                                                   observer_init_state=_observer_init_state)

            error_x, error_z = _simulate_model_error(model=model, dataset=dataset, forward=sim_error_z,
                                                     inverse=sim_error_x)

            if sim_error_x:
                param_error_x_list[j, i] = error_x

            if sim_error_z:
                param_error_z_list[j, i] = error_z

        for j, param_path in enumerate(param_model_paths):
            path = param_path.format(lmbda=lmbda)
            model = torch.load(path)

            if dataset is None:
                _observer_init_state = duffing.get_observer_init_state(model=model, system_init_state=system_init_state)
                dataset = duffing.generate_dataset(systems=(system,), system_init_states=(system_init_state,),
                                                   system_output_noise=system_output_noise,
                                                   observer_init_state=_observer_init_state)

            error_x, error_z = _simulate_model_error(model=model, dataset=dataset, forward=sim_error_z,
                                                     inverse=sim_error_x)

            if sim_error_x:
                param_error_x_list[len(static_models) + j, i] = error_x

            if sim_error_z:
                param_error_z_list[len(static_models) + j, i] = error_z

    data = {
        _NPZ_KEY_PARAM_LMBDA: param_lmbda
    }

    if sim_error_x:
        data[_NPZ_KEY_PARAM_ERROR_Z_LIST] = param_error_z_list

    if sim_error_x:
        data[_NPZ_KEY_PARAM_ERROR_X_LIST] = param_error_x_list

    np.savez(data_path, **data)


def plot_param_error_x(
        data_path: str,
        param_lambda_train: Optional[np.ndarray],
        show: bool,
        save_path: Optional[str],
        figsize,
        colors
):
    data = np.load(data_path)
    param_lmbda = data[_NPZ_KEY_PARAM_LMBDA]
    param_error_list = data[_NPZ_KEY_PARAM_ERROR_X_LIST]

    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.subplots()

    for i, param_error in enumerate(param_error_list):
        ax.plot(param_lmbda, param_error, linewidth=0.8, color=colors[i])
        print(param_error.mean())

    if param_lambda_train is None:
        ax.margins(x=0.0)
    else:
        for x in param_lambda_train:
            ax.axvline(x, linewidth=0.8, dashes=(3, 1), color=style.TU_BERLIN_BLACK)

        ax.margins(x=0.1)

    ax.grid(which='both', dashes=(1, 2))
    ax.tick_params(which='both', direction='in')
    ax.set_xlabel(r'Parameter $\lambda$')
    ax.set_ylabel(r'Error $\bar{e}_t^*(T\sim\lambda,e^*_x)$')

    plt.tight_layout(pad=0.0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(fname=save_path)

    if show:
        plt.show()


def plot_param_error_z(
        data_path: str,
        show: bool,
        save_path: Optional[str],
        figsize,
        colors
):
    data = np.load(data_path)
    param_lmbda = data[_NPZ_KEY_PARAM_LMBDA]
    param_error_list = data[_NPZ_KEY_PARAM_ERROR_Z_LIST]

    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.subplots()

    for i, error in enumerate(param_error_list):
        ax.plot(param_lmbda, error, linewidth=0.8, color=colors[i])
        print(error.mean())

    ax.margins(x=0.0)
    ax.grid(which='both', dashes=(1, 2))
    ax.tick_params(which='both', direction='in')
    ax.set_xlabel(r'Parameter $\lambda$')
    ax.set_ylabel(r'Error $\bar{e}_t^*(T\sim\lambda,e^*_z)$')

    plt.tight_layout(pad=0.0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(fname=save_path)

    if show:
        plt.show()
