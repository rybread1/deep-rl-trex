import tensorflow as tf
from environment import Environment
from agent import Agent


if __name__ == '__main__':

    # create environment object
    env = Environment()

    save_path = 'testing_model/model-weights'

    agent = Agent(env,
                  tf.keras.optimizers.Adam(learning_rate=0.0001),
                  memory_length=50000,
                  dueling=True,
                  noisy_net=False,
                  egreedy=True,
                  loss='mse',
                  save_weights=False,
                  verbose_action=False)

    agent.load_weights(save_path)
    env.init_game()

    for episode in range(10000000):
        env.demo(agent)
