"""
#   Details of the simulation used are described in the following link
#   https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
"""

import datetime
import gym
import numpy as np

from Q_learning import Qlearning

"""Constant simulation values"""
ALPHA = 0.9
GAMMA = 0.85
EPSILON = 0.99

EPISODES = 9000


def discretize_observation_space(state: np.array, low: np.array, high: np.array, number_of_states: np.array) -> tuple:
    """Converts the current, continuous, observed values to the corresponding state in q_table"""

    arg = (state - low) * (number_of_states - 1) / (high - low)
    arg = np.round(arg).astype(int)

    return tuple(arg)


def learn_agent(agent: Qlearning) -> None:
    """Here is the process of training an agent"""

    for i in range(EPISODES):

        # to view statistics
        rew = 0

        agent.epsilon_exponential_decrease(i / EPISODES)

        done = False

        # Random, initial values
        state = env.reset()

        # Replace the initial continuous conditions with the corresponding state in q_table
        state = discretize_observation_space(state, env.observation_space.low, env.observation_space.high,
                                             np.array(agent.get_discrete_space()[:-1]))

        while not done:
            """
            # There is one full EPISODE in this loop.
            # Ends when goal is reached or after 200 steps of simulation.
            """

            action = agent.make_decision(state)

            next_state, reward, done, info = env.step(action)

            # A small penalty in each step to make the agent want to reach the goal as quickly as possible.
            agent.add_reward(-0.1)

            # reward for reaching a goal
            if next_state[0] >= 0.49:
                agent.add_reward(10)

            # Replace the actual continuous conditions with the corresponding state in q_table
            next_state = discretize_observation_space(next_state, env.observation_space.low, env.observation_space.high,
                                                      np.array(agent.get_discrete_space()[:-1]))

            # Actual reward for statistics
            rew += agent.get_reward()

            # Update q_table
            agent.q_table_update(state, next_state)

            state = next_state

        # Statistics
        show = 500
        if (i + 1) % show == 0:
            print(f"Episode: {i + 1} / {EPISODES} -> {100 * (i + 1) / EPISODES}%")
            print(f"Epsilon: {agent.get_epsilon()}")
            print(f"Avr reward: {rew / show}\n")
            rew = 0


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    q = Qlearning(ALPHA, GAMMA, EPSILON)

    # START INITIALIZE MODEL
    """
    # There are 3 discrete deterministic actions:
    #   0 -> Accelerate to the left
    #   1 -> Don't accelerate
    #   2 -> Accelerate to the right
    """
    q.action_space_init([0, 1, 2])

    """
    # The observation space:
    #   0 -> position of the car along the x-axis   Min: -1.2   Max: 0.6
    #   1 -> velocity of the car                    Min: -0.07  MAx: 0.07

    # Divided the continuous observation space into 90 velocity intervals and 90 position intervals.
    # Experimentally selected values.
    """
    q.discrete_space_init((90, 90, len(q.get_action_space())))

    # Init Q table
    q.q_table_init()

    # END INITIALIZE MODEL

    """
    # Instead of initializing the model, you can load the already pre-trained one for further development. 
    # All necessary parameters are stored in the file and will be automatically filled in the class.
    """
    # q.load_model('models\\2022_03_28--23_53_07.npz')

    # Train model
    learn_agent(q)

    # Save results
    models_dir = "models\\" + datetime.datetime.now().strftime("%Y_%m_%d--%H_%M_%S") + ".npz"
    q.save_model(models_dir)
