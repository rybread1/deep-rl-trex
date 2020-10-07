import tensorflow as tf
from environment import Environment
from agent import Agent
from logger import Logger

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    tf.device("/gpu:0")

    # create environment object
    env = Environment(space_sleep=0.55, no_action_sleep=0.03)
    logger = Logger(fp=None)

    memory_fp = 'C:/Users/ryano/rl_projects/trex_memory/memory.pkl'
    save_path = 'C:/Users/ryano/repos/deep-rl-trex/model/model-weights-noisy'

    mem_length = 150000

    agent = Agent(env,
                  tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse',
                  memory_length=mem_length,
                  dueling=True,
                  noisy_net=True,
                  egreedy=True,
                  save_memory=memory_fp,
                  save_weights=save_path,
                  verbose_action=True)

    # agent.load_weights(save_path)
    # agent.load_memory(memory_fp)
    agent.set_beta_schedule(beta_start=0.4, beta_max=1, annealed_samplings=30000)
    # agent.set_epsilon_decay_schedule(epsilon=0.0000001, epsilon_min=0.0000001, annealed_steps=50000)

    agent.pretraining_steps = 5000
    print(f'pretraining for {agent.pretraining_steps} steps...')

    env.init_game()

    for episode in range(10000000):
        env.run(episode, agent, batch_size=32, logger=logger)

