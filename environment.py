from mss import mss
import time
from PIL import Image
from collections import deque
from win32api import GetSystemMetrics

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from action_space import ActionSpace

# Helper libraries
import numpy as np
import datetime


class Environment:
    def __init__(self):
        self.url = 'http://www.trex-game.skipser.com/'
        self.window_width = GetSystemMetrics(0) * 0.37
        self.window_height = GetSystemMetrics(1) * 0.8

        self.sct = mss()
        self.bbox = {'top': 350, 'left': 50, 'width': 630, 'height': 80}
        self.terminal_bbox = {'top': 360, 'left': 357, 'width': 5, 'height': 5}

        self.state = None
        self.frame_history = deque(maxlen=4)

        self.driver = None
        self.window_element = None
        self.action_space = None
        self.actions = None

    def render(self):
        opts = Options()
        opts.add_argument(f"--width={self.window_width}")
        opts.add_argument(f"--height={self.window_height}")

        self.driver = webdriver.Firefox(executable_path='/Users/ryano/Downloads/geckodriver-v0.27.0-win64/geckodriver',
                                        options=opts)
        self.driver.get(self.url)
        self.window_element = self.driver.find_element_by_id("t")
        self.action_space = ActionSpace(self.window_element,
                                        space_sleep=0.58,
                                        no_action_sleep=0.02)

        self.actions = self.action_space.actions

    def step(self, action):

        self.actions[action].act()
        self.frame_history.append(self._grab_sct())
        next_state = self._update_state()
        terminal = self._is_terminal()

        if not terminal:
            reward = 0.1
        else:
            reward = -1

        return next_state, reward, terminal

    def reset(self):
        time.sleep(1)
        self.action_space.space.act()
        self.reset_frame_history()
        self.state = self._update_state()
        time.sleep(2.0)

    def reset_frame_history(self):
        for i in range(4):
            self.frame_history.append(self._grab_sct())

    def _update_state(self):
        return np.stack(self.frame_history).reshape(1, 20, 157, 4)

    def _grab_sct(self):
        sct_img = self.sct.grab(self.bbox)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX").convert('L')

        (width, height) = (img.width // 4, img.height // 4)
        img = img.resize((width, height))
        img_np = np.array(img) / 255.0
        return np.expand_dims(img_np, axis=0)

    def _is_terminal(self):
        sct_img = self.sct.grab(self.terminal_bbox)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX").convert('L')

        terminal_array_match = np.array([[83, 83, 83, 83, 83],
                                         [83, 83, 83, 83, 83],
                                         [83, 83, 83, 83, 83],
                                         [83, 83, 83, 83, 83],
                                         [83, 83, 247, 247, 247]])

        return (terminal_array_match == np.array(img)).all()

    def init_game(self):
        resp = input('enter "y" to start training: ')
        if resp == 'y':
            self.render()
        else:
            exit()

    def run(self, epoch, agent, batch_size, logger):

        self.reset()
        epoch_start_time = datetime.datetime.now()
        batch_loader = []
        q_values = []

        for step in range(10000000):

            action, q_value = agent.choose_action(self.state)
            next_state, reward, terminal = self.step(action)
            batch_loader.append([self.state, action, reward, next_state, terminal])
            q_values.append(q_value)
            self.state = next_state

            if terminal:

                agent.batch_store(batch_loader)
                if (agent.memory.length > agent.pretraining_steps) or (agent.memory.memory_size == agent.memory.length):
                    agent.replay(batch_size, epoch_steps=step)

                if (epoch % 40 == 0) and (epoch != 0):
                    agent.save_memory(agent.save_memory_fp)

                log_data = {
                    'epoch': epoch,
                    'epoch_steps': step,
                    'epoch_tot_rewards': sum([x[2] for x in batch_loader]),
                    'epoch_time': datetime.datetime.now() - epoch_start_time,
                    'epoch_avg_q': np.mean(q_values),
                }

                logger.log(agent, log_data, verbose=True)

                break

    def demo(self, agent):

        agent.epsilon = 0
        agent.epsilon_min = 0

        self.reset()
        for step in range(10000000):
            action = agent.choose_action(self.state)
            next_state, reward, terminal = self.step(action)
            self.state = next_state
            if terminal:
                break
