import pandas as pd
from datetime import datetime
from os import path, mkdir


class Logger:
    def __init__(self, fp=None):
        self.fp = fp
        self.log_dir = None
        self.csv_fp = None
        self.create_log_dir()

    def create_log_dir(self):
        self.log_dir = f'log/{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        if self.fp:
            self.log_dir = self.fp
            assert path.isdir(self.log_dir), 'Enter an existing log directory!'

        if not path.isdir(self.log_dir):
            mkdir(path.join(self.log_dir))

        self.csv_fp = path.join(self.log_dir, 'log.csv')

    def log(self, agent, log_data, verbose=True):

        tot_run_time = datetime.now() - agent.start_time

        if verbose:
            print(
                '\n=======================================================\n',
                f'epoch: {log_data["epoch"]}\n'
                f'    epoch steps: {log_data["epoch_steps"]}\n'
                f'    epoch rewards: {log_data["epoch_tot_rewards"]}\n'
                f'    epoch time: {log_data["epoch_time"]}\n'
                f'    epoch avg q: {log_data["epoch_avg_q"]}\n'
                f'    total_steps: {agent.total_steps}\n'
                f'    epsilon: {agent.epsilon}\n'
                f'    beta: {agent.memory.beta}\n'
                f'    memory len: {agent.memory.length}\n'
                f'    total run time: {tot_run_time}\n',
                '=======================================================\n'
            )

        data = [log_data["epoch"], log_data["epoch_steps"], log_data["epoch_tot_rewards"], log_data["epoch_time"],
                log_data["epoch_avg_q"], agent.total_steps, agent.epsilon, agent.memory.beta, agent.memory.length,
                tot_run_time, agent.dueling, agent.noisy_net, agent.pretraining_steps, agent.tau]

        if path.exists(self.csv_fp):
            pd.DataFrame(data).T.to_csv(self.csv_fp, index=False, header=False, mode='a')

        else:
            cols = ['epoch', 'epoch_steps', 'epoch_rewards', 'epoch_time', 'epoch_avg_q', 'total_steps',
                    'epsilon', 'beta', 'memory_len', 'total_run_time', 'dueling', 'noisy_net', 'pretraining_steps',
                    'tau']

            pd.DataFrame(data, index=cols).T.to_csv(self.csv_fp, index=False)

