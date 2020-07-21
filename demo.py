import tensorflow as tf
from environment import Environment
from agent import Agent


if __name__ == '__main__':

    # create environment object
    env = Environment()

    load_path = 'model/model-weights'
    save_path = 'model/model-weights'

    agent = Agent(env,
                  tf.keras.optimizers.Adam(learning_rate=0.0001),
                  memory_length=50000,
                  dueling=True,
                  loss='mse',
                  load_weights=load_path,
                  save_weights=None,
                  verbose_action=False)

    env.init_game()

    for episode in range(10000000):
        env.demo(agent)
