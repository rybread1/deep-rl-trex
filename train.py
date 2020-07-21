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
                  save_weights=save_path,
                  verbose_action=False)

    agent.set_epsilon_decay_schedule(epsilon=0.0001, epsilon_min=0.0001, annealed_steps=20000)
    agent.set_beta_schedule(beta_start=0.9, beta_max=1, annealed_samplings=100)

    agent.pretraining_steps = 10000

    env.init_game()

    for episode in range(10000000):
        env.run(episode, agent, batch_size=32, log_fn='log-07-18-2020.csv')
