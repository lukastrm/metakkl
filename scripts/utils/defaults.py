import numpy as np
import torch

from metakkl.nn import MappingNetwork, KKLObserverNetwork
from metakkl.system import Observer, DuffingOscillator


# Duffing system parameters
DUFFING_TIME_START = 0.0
DUFFING_TIME_END = 50.0
DUFFING_NUM_STEPS = 1000
DUFFING_SYSTEM_LMBDA = 1.0
DUFFING_SYSTEM_INIT_STATE = np.array([0.5, 0.5])
DUFFING_OBSERVER_A = -np.diag(np.arange(start=1, stop=6))
DUFFING_OBSERVER_B = np.ones([5, 1])
DUFFING_OBSERVER_STATE_SIZE = 5
DUFFING_OBSERVER = Observer(a=DUFFING_OBSERVER_A, b=DUFFING_OBSERVER_B, state_size=DUFFING_OBSERVER_STATE_SIZE,
                            input_size=DuffingOscillator.OUTPUT_SIZE)
DUFFING_OBSERVER_INIT_STATE = np.zeros(DUFFING_OBSERVER_STATE_SIZE)

# Model parameters
_DEFAULT_N_HIDDEN_LAYERS = 5
_DEFAULT_HIDDEN_LAYER_SIZE = 50


def kkl_observer_model(system_state_size: int, observer_state_size: int) -> KKLObserverNetwork:
    forward_map = MappingNetwork(input_size=system_state_size, output_size=observer_state_size,
                                 n_hidden_layers=_DEFAULT_N_HIDDEN_LAYERS, hidden_layer_size=_DEFAULT_HIDDEN_LAYER_SIZE)
    inverse_map = MappingNetwork(input_size=observer_state_size, output_size=system_state_size,
                                 n_hidden_layers=_DEFAULT_N_HIDDEN_LAYERS, hidden_layer_size=_DEFAULT_HIDDEN_LAYER_SIZE)
    model = KKLObserverNetwork(forward_map=forward_map, inverse_map=inverse_map)
    return model


def set_training_seeds():
    np.random.seed(888)
    torch.manual_seed(9)


def set_validation_seeds():
    np.random.seed(555)
    torch.manual_seed(99)
