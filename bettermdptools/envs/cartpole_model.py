"""
generative AI experiment - discretized cartpole transition and reward (P) matrix with adaptive angle binning
created with chatGPT

Example usage:
dpole = DiscretizedCartPole(10, 10, 10, .1, .5)  # Example bin sizes for each variable and adaptive angle binning center/outer resolution

"""

import numpy as np


class DiscretizedCartPole:
    def __init__(
        self,
        position_bins,
        velocity_bins,
        angle_bins,
        angular_velocity_bins
    ):
        """
        Initializes the DiscretizedCartPole model.

        Parameters:
        - position_bins (int): Number of discrete bins for the cart's position.
        - velocity_bins (int): Number of discrete bins for the cart's velocity.
        - angular_velocity_bins (int): Number of discrete bins for the pole's angular velocity.

        Attributes:
        - state_space (int): Total number of discrete states in the environment.
        - P (dict): Transition probability matrix where P[state][action] is a list of tuples (probability, next_state,
        reward, done).
        - transform_obs (lambda): Function to transform continuous observations into a discrete state index.
        """
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.angle_bins =  angle_bins
        self.angular_velocity_bins = angular_velocity_bins
        self.action_space = 2  # Left or Right

        # Define the range for each variable
        """
        self.position_range = (-2.4, 2.4)
        self.velocity_range = (-3, 3)
        self.angle_range = (-12 * np.pi / 180, 12 * np.pi / 180)
        self.angular_velocity_range = (-1.5, 1.5)
        """
        self.state_space = np.prod(
            [
                len(self.position_bins),
                len(self.velocity_bins),
                len(self.angle_bins),
                len(self.angular_velocity_bins),
            ]
        )
        self.P = {
            state: {action: [] for action in range(self.action_space)}
            for state in range(self.state_space)
        }
        self.setup_transition_probabilities()
        self.n_states = (
            len(self.angle_bins)
            * len(self.velocity_bins)
            * len(self.position_bins)
            * len(self.angular_velocity_bins)
        )
        """
        Explanation of transform_obs lambda: 
        This lambda function will take cartpole observations, determine which bins they fall into, 
        and then convert bin coordinates into a single index.  This makes it possible 
        to use traditional reinforcement learning and planning algorithms, designed for discrete spaces, with continuous 
        state space environments. 
        
        Parameters:
        - obs (list): A list of continuous observations [position, velocity, angle, angular_velocity].

        Returns:
        - int: A single integer representing the discretized state index.
        """
        self.transform_obs = lambda obs: (
            np.ravel_multi_index(
                (
                    np.clip(
                        np.digitize(
                            obs[0],
                            self.position_bins,
                        )
                        - 1,
                        0,
                        len(self.position_bins) - 1,
                    ),
                    np.clip(
                        np.digitize(
                            obs[1],
                            self.velocity_bins,
                        )
                        - 1,
                        0,
                        len(self.velocity_bins) - 1,
                    ),
                    np.clip(
                        np.digitize(obs[2], self.angle_bins) - 1,
                        0,
                        len(self.angle_bins) - 1,
                    ),
                    # Use adaptive angle bins
                    np.clip(
                        np.digitize(
                            obs[3],
                                self.angular_velocity_bins
                        )
                        - 1,
                        0,
                        len(self.angular_velocity_bins) - 1,
                    ),
                ),
                (
                    len(self.position_bins),
                    len(self.velocity_bins),
                    len(self.angle_bins),
                    len(self.angular_velocity_bins),
                ),
            )
        )

    def setup_transition_probabilities(self):
        """
        Sets up the transition probabilities for the environment. This method iterates through all possible
        states and actions, simulates the next state, and records the transition probability
        (deterministic in this setup), reward, and termination status.
        """
        for state in range(self.state_space):
            position, velocity, angle, angular_velocity = self.index_to_state(state)
            for action in range(self.action_space):
                next_state, reward, done = self.compute_next_state(
                    position, velocity, angle, angular_velocity, action
                )
                self.P[state][action].append((1, next_state, reward, done))

    def index_to_state(self, index):
        """
        Converts a single index into a multidimensional state representation.

        Parameters:
        - index (int): The flat index representing the state.

        Returns:
        - list: A list of indices representing the state in terms of position, velocity, angle, and angular velocity bins.
        """
        totals = [
            len(self.position_bins),
            len(self.velocity_bins),
            len(self.angle_bins),
            len(self.angular_velocity_bins),
        ]
        multipliers = np.cumprod([1] + totals[::-1])[:-1][::-1]
        components = [int((index // multipliers[i]) % totals[i]) for i in range(4)]
        return components

    def compute_next_state(
        self, position_idx, velocity_idx, angle_idx, angular_velocity_idx, action
    ):
        """
        Computes the next state based on the current state indices and the action taken. Applies simplified physics calculations to determine the next state.

        Parameters:
        - position_idx (int): Current index of the cart's position.
        - velocity_idx (int): Current index of the cart's velocity.
        - angle_idx (int): Current index of the pole's angle.
        - angular_velocity_idx (int): Current index of the pole's angular velocity.
        - action (int): Action taken (0 for left, 1 for right).

        Returns:
        - tuple: Contains the next state index, the reward, and the done flag indicating if the episode has ended.
        """
        position =self.position_bins[position_idx]
        velocity = self.velocity_bins[velocity_idx]
        angle = self.angle_bins[angle_idx]
        angular_velocity = self.angular_velocity_bins[angular_velocity_idx]

        # Simulate physics here (simplified)
        force = 10 if action == 1 else -10
        new_velocity = velocity + (force + np.cos(angle) * -10.0) * 0.02
        new_position = position + new_velocity * 0.02
        new_angular_velocity = angular_velocity + (-3.0 * np.sin(angle)) * 0.02
        new_angle = angle + new_angular_velocity * 0.02

        new_position_idx = np.clip(
            np.digitize(
                new_position, self.position_bins
            )
            - 1,
            0,
            len(self.position_bins) - 1,
        )
        new_velocity_idx = np.clip(
            np.digitize(
                new_velocity, self.velocity_bins
            )
            - 1,
            0,
            len(self.velocity_bins) - 1,
        )
        new_angle_idx = np.clip(
            np.digitize(new_angle, self.angle_bins) - 1, 0, len(self.angle_bins) - 1
        )
        new_angular_velocity_idx = np.clip(
            np.digitize(
                new_angular_velocity,
                self.angular_velocity_bins,
            )
            - 1,
            0,
            len(self.angular_velocity_bins) - 1,
        )

        new_state_idx = np.ravel_multi_index(
            (
                new_position_idx,
                new_velocity_idx,
                new_angle_idx,
                new_angular_velocity_idx,
            ),
            (
                len(self.position_bins),
                len(self.velocity_bins),
                len(self.angle_bins),
                len(self.angular_velocity_bins),
            ),
        )

        reward = 1 if np.abs(new_angle) < 12 * np.pi / 180 else -1
        done = (
            True
            if np.abs(new_angle) >= 12 * np.pi / 180 or np.abs(new_position) > 2.4
            else False
        )

        return new_state_idx, reward, done
