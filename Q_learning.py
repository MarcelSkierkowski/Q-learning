import random
import numpy as np


class Qlearning:
    """class designed to train a model using q-learning. """

    def __init__(self, alpha: float, gamma: float, epsilon: float):

        self._action_space = None
        self._discrete_space = None

        self.Q_table = None
        self._reward = 0

        self._action = None

        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

        self.reward_list = []
        self.avr_reward_list = []

    def save_model(self, filename: str) -> None:
        """save the model to a file"""
        np.savez(filename, name1=self.Q_table, name2=self._action_space, name3=self._discrete_space)

    def load_model(self, filename: str) -> None:
        """upload model from file"""
        data = np.load(filename)
        self.Q_table = data['name1']
        self._action_space = data['name2']
        self._discrete_space = data['name3']

    def q_show(self, state: np.array) -> np.array:
        return self.Q_table[state]

    def add_reward(self, reward: float) -> None:
        """rewards and penalties for the agent"""
        self._reward += reward

    def get_reward(self) -> float:
        return self._reward

    def action_space_init(self, values: np.array) -> None:
        """actions that can be executed"""
        self._action_space = values

    def get_action_space(self) -> np.array:
        return self._action_space

    def discrete_space_init(self, discretize: np.array) -> None:
        """into how many parts to divide each of the observed phenomena"""
        self._discrete_space = tuple(discretize)

    def get_discrete_space(self) -> np.array:
        return self._discrete_space

    def q_table_init(self) -> None:
        self.Q_table = np.random.uniform(low=-1, high=1, size=self._discrete_space)

    def q_table_update(self, state: np.array, next_state: np.array) -> None:
        """
        Calculates the value of the reward for the action performed.

        To calculate the value, the previous value of the reward is used,
        the currently received reward
        and the potentially possible reward in the next move
    """

        # previous saved award value
        new_value = (1 - self._alpha) * self.Q_table[state][self._action]

        # the currently earned award plus the potential maximum award in the next action
        new_value += self._alpha * (self._reward + self._gamma * np.max(self.Q_table[next_state]))

        self.Q_table[state][self._action] = new_value
        self._reward = 0

    def epsilon_linear_decrease(self, episodes: int) -> None:
        """
            It decreases the epsilon linearly in each step.
            From the maximal value to zero.
        """
        self._epsilon -= self._epsilon / episodes

    def epsilon_exponential_decrease(self, tmp: float) -> None:
        val = 1 / (1 + (0.002 * tmp))
        self._epsilon = self._epsilon * val

    def get_epsilon(self) -> float:
        """Returns the current value of epsilon"""
        return self._epsilon

    def make_decision(self, state: np.array) -> int:
        """
            Chooses the next action
            The epsilon value determines the probability that the next action will be random.
        """
        if random.uniform(0, 1) < 1 - self._epsilon:
            # Exploit learned values.
            self._action = np.argmax(self.Q_table[state])
        else:
            # Explore action space
            self._action = random.randint(0, len(self._action_space) - 1)

        return self._action_space[self._action]
