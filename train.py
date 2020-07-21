import tensorflow as tf
from environment import Environment
from agent import Agent


if __name__ == '__main__':

    # create environment object
    env = Environment()

    memory_fp = 'memory/memory.pkl'

    load_path = 'testing_model/model-weights'
    save_path = 'testing_model/model-weights'

    mem_length = 80000

    agent = Agent(env,
                  tf.keras.optimizers.Adam(learning_rate=0.0001),
                  memory_length=mem_length,
                  dueling=True,
                  loss='mse',
                  load_memory=memory_fp,
                  save_memory=memory_fp,
                  load_weights=load_path,
                  save_weights=save_path,
                  verbose_action=False)

    agent.set_epsilon_decay_schedule(epsilon=0.00001, epsilon_min=0.000001, annealed_steps=20000)
    agent.set_beta_schedule(beta_start=0.9, beta_max=1, annealed_samplings=1)

    agent.pretraining_steps = mem_length - agent.memory.length
    print(f'pretraining for {agent.pretraining_steps} steps...')

    env.init_game()

    for episode in range(10000000):
        env.run(episode, agent, batch_size=32, log_fn='log-07-22-2020.csv')
