import pandas as pd
from datetime import datetime
from os import path, mkdir


class Logger:
    def __init__(self, append_existing_log=False):
        self.append_existing_log = append_existing_log
        self.log_dir = None
        self.csv_fp = None
        self.create_log_dir()

    def create_log_dir(self):
        self.log_dir = f'log/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        if self.append_existing_log:
            assert type(self.append_existing_log) == 'str', 'Enter the file name of the log you want to append to'
            self.log_dir = self.append_existing_log

        if not path.isdir(self.log_dir):
            mkdir(path.join(self.log_dir))

        self.csv_fp = path.join(self.log_dir, 'log.csv')

    def log(self, agent, log_data, verbose=True):

        tot_run_time = datetime.now() - agent.start_time

        if verbose:
            print(f'epoch: {log_data["epoch"]}, '
                  f'epoch steps: {log_data["epoch_steps"]}, '
                  f'epoch rewards: {log_data["epoch_tot_rewards"]}, '
                  f'epoch time: {log_data["epoch_time"]}, '
                  f'epoch avg q: {log_data["epoch_avg_q"]}, '
                  f' total_steps: {agent.total_steps}, '
                  f'epsilon: {agent.epsilon},'
                  f'beta: {agent.memory.beta},'
                  f'memory len: {agent.memory.length},'
                  f'total run time: {tot_run_time}')

        data = [log_data["epoch"], log_data["epoch_steps"], log_data["epoch_tot_rewards"], log_data["epoch_time"],
                log_data["epoch_avg_q"], agent.total_steps, agent.epsilon, agent.memory.beta, agent.memory.length,
                tot_run_time]

        if path.exists(self.log_dir):
            pd.DataFrame(data).T.to_csv(self.csv_fp, index=False, header=False, mode='a')

        else:
            cols = ['epoch', 'epoch_steps', 'epoch_rewards', 'epoch_time', 'epoch_avg_q', 'total_steps',
                    'epsilon', 'beta', 'memory_len', 'total_run_time']

            pd.DataFrame(data, index=cols).T.to_csv(self.csv_fp, index=False)

