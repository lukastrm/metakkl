import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt

from scripts.utils import duffing, style
from metakkl.system import Noise, DuffingOscillator


_NPZ_KEY_T = 't'
_NPZ_KEY_X_LABEL = 'x_label'
_NPZ_KEY_X_PRED_LIST = 'x_pred_list'


def simulate(
        lmbda: float,
        system_init_state: np.ndarray,
        system_noise: Optional[Noise],
        system_output_noise: Optional[Noise],
        observer_init_state: Optional[np.ndarray],
        model_paths: Sequence[str],
        data_path: str,
):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    t = None
    x_label = None
    x_pred_list = []

    for model_path in model_paths:
        model = torch.load(model_path)

        if observer_init_state is not None and observer_init_state.size == 0:
            observer_init_state = duffing.get_observer_init_state(model=model, system_init_state=system_init_state)

        system = DuffingOscillator(lmbda=lmbda, noise=system_noise)
        dataset = duffing.generate_dataset(systems=(system,), system_init_states=(system_init_state,),
                                           system_output_noise=system_output_noise,
                                           observer_init_state=observer_init_state)
        x_pred, _ = duffing.simulate_model(dataset=dataset, model=model, forward=False, inverse=True)

        t = dataset.t
        x_label = dataset.x
        x_pred_list.append(x_pred)

    save_data = {
        _NPZ_KEY_T: t,
        _NPZ_KEY_X_LABEL: x_label,
        _NPZ_KEY_X_PRED_LIST: x_pred_list
    }

    np.savez(data_path, **save_data)


def plot(
        data_path: str,
        show: bool,
        save_path: Optional[str],
        figsize,
        colors,
        xticks=None
):
    data = np.load(data_path)
    t = data[_NPZ_KEY_T]
    x_label = data[_NPZ_KEY_X_LABEL]
    x_pred_list = data[_NPZ_KEY_X_PRED_LIST]

    fig = plt.figure(figsize=figsize)
    ax: List[plt.Axes] = fig.subplots(nrows=2, ncols=1, sharex=True)

    for i, ax_i in enumerate(ax):
        ax[i].plot(t, x_label[:, i], color=style.TU_BERLIN_BLACK, linewidth=0.8, dashes=(1, 1))

        for j, x_pred in enumerate(x_pred_list):
            ax[i].plot(t, x_pred[:, i], linewidth=0.8, color=colors[j])

        ax[i].margins(x=0.0, y=0.1)
        ax[i].grid(which='both', dashes=(1, 2))
        ax[i].tick_params(which='both', direction='in')
        ax[i].set_ylabel(f'$x_{i+1}(t)$')

    ax[-1].set_xlabel('Time $t$')

    if xticks is not None:
        ax[-1].set_xticks(xticks)

    fig.align_ylabels(ax)
    plt.tight_layout(pad=0.0)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
