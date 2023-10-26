from typing import Optional, Sequence

import numpy as np

from metakkl.utils import Noise


class System:
    def __init__(self, state_size: int, input_size: int = 0, output_size: int = 0):
        self.state_size: int = state_size
        self.input_size: int = input_size
        self.output_size: int = output_size

    @property
    def output_indices(self) -> Sequence[int]:
        raise NotImplementedError

    def function(self, time: float, state: np.ndarray, sys_input: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class NoisySystem:
    def __init__(self, noise: Optional[Noise] = None):
        self.noise: Optional[Noise] = noise

    def generate_system_noise(self) -> Optional[np.ndarray]:
        noise = None

        if self.noise is not None:
            noise = np.random.normal(self.noise[0], self.noise[1])

        return noise


class Observer(System):
    def __init__(self, a: np.ndarray, b: np.ndarray, state_size: int, input_size: int = 0, output_size: int = 0):
        super().__init__(state_size, input_size, output_size)
        self.a: np.ndarray = a
        self.b: np.ndarray = b

    @property
    def output_indices(self) -> Sequence[int]:
        return []

    def function(self, time: float, state: np.ndarray, sys_input: np.ndarray) -> np.ndarray:
        return np.dot(self.a, state) + np.dot(self.b, sys_input)


class DuffingOscillator(System, NoisySystem):
    STATE_SIZE = 2
    OUTPUT_SIZE = 1
    OBSERVABLE_STATE_INDICES = [0]

    def __init__(self, lmbda: float = 1.0, noise: Optional[Noise] = None):
        System.__init__(self, state_size=DuffingOscillator.STATE_SIZE, output_size=DuffingOscillator.OUTPUT_SIZE)
        NoisySystem.__init__(self, noise=noise)

        self.lmbda: float = lmbda

    @property
    def output_indices(self) -> Sequence[int]:
        return DuffingOscillator.OBSERVABLE_STATE_INDICES

    def function(self, time: float, state: np.ndarray, sys_input: np.ndarray) -> np.ndarray:
        state_dot = np.array([
            self.lmbda * state[1] ** 3,
            -self.lmbda * state[0]
        ])

        noise = self.generate_system_noise()

        if noise is not None:
            state_dot += noise

        return state_dot
