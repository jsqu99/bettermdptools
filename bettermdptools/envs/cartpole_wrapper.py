"""
Author: John Mansfield
BSD 3-Clause License
"""

import gymnasium as gym

from bettermdptools.envs.cartpole_model import DiscretizedCartPole


class CustomTransformObservation(gym.ObservationWrapper):
    def __init__(self, env, func, observation_space):
        """
        Helper class that modifies the observation space. The v26 gymnasium TransformObservation wrapper does not
        accept an observation_space parameter, which is needed in order to match the lambda conversion (tuple->int).
        Instead, we subclass gym.ObservationWrapper (parent class of gym.TransformObservation)
        to set both the conversion function and new observation space.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment to be wrapped.
        func : lambda
            Function that converts the observation.
        observation_space : gymnasium.spaces.Space
            New observation space.
        """
        super().__init__(env)
        if observation_space is not None:
            self.observation_space = observation_space
        self.func = func

    def observation(self, observation):
        """
        Applies a function to the observation received from the environment's step function,
        which is passed back to the user.

        Parameters
        ----------
        observation : Tuple
            Base environment observation tuple.

        Returns
        -------
        int
            The converted observation.
        """
        return self.func(observation)


class CartpoleWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        dpole
    ):
        """
        Cartpole wrapper that modifies the observation space and creates a transition/reward matrix P.

        Parameters
        ----------
        env : gymnasium.Env
            Base environment.
        dpole: DiscretizedCartPole
        """
        self.dpole = dpole
        self._P = dpole.P
        self._transform_obs = dpole.transform_obs
        env = CustomTransformObservation(
            env, self._transform_obs, gym.spaces.Discrete(dpole.n_states)
        )
        super().__init__(env)

    @property
    def P(self):
        """
        Returns
        -------
        dict
            Transition/reward matrix.
        """
        return self._P

    @property
    def transform_obs(self):
        """
        Returns
        -------
        lambda
            Function that converts the observation.
        """
        return self._transform_obs
