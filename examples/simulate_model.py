import gym
import numpy as np
from Q_learning import Qlearning


def discretize_observation_space(state: np.array, low: np.array, high:np.array, number_of_states:np.array) -> tuple:
    """Converts the current, continuous, observed values to the corresponding state in q_table"""

    arg = (state - low) * (number_of_states - 1) / (high - low)
    arg = np.round(arg).astype(int)

    return tuple(arg)


def show_simulation(agent: Qlearning) -> None:
    done = False

    # Random, initial values
    state = env.reset()

    while not done:

        # Rendering a graphical window
        env.render()

        # Replace the actual continuous conditions with the corresponding state in q_table
        state = discretize_observation_space(state, env.observation_space.low, env.observation_space.high,
                                             agent.get_discrete_space()[:-1])

        action = agent.make_decision(state)
        next_state, reward, done, info = env.step(action)

        state = next_state


if __name__ == '__main__':

    # Select the appropriate simulation
    env = gym.make('MountainCar-v0')

    # Alpha, gamma and epsilon are arbitrary because we are just exploiting knowledge
    q = Qlearning(1, 1, 0)

    # Load the selected model
    q.load_model('MountainCar\\models\\2022_03_28--23_53_07.npz')

    # Run the simulation a selected number of times.
    episodes = 3
    for _ in range(episodes):
        show_simulation(q)
