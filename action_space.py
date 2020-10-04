from selenium.webdriver.common.keys import Keys
import time
import random


class Action:
    def __init__(self, ele, sleep_time):
        self.ele = ele
        self.sleep_time = sleep_time

    def act(self):
        raise NotImplementedError()


class Space(Action):
    def __init__(self, ele, sleep_time=0.59):
        super().__init__(ele, sleep_time)

    def act(self):
        self.ele.send_keys(Keys.SPACE)
        time.sleep(self.sleep_time)
        return 0


class NoAction(Action):
    def __init__(self, ele, sleep_time=0.02):
        super().__init__(ele, sleep_time)

    def act(self):
        time.sleep(self.sleep_time)
        return 1


class ActionSpace:
    def __init__(self, ele, space_sleep=0.59, no_action_sleep=0.02):
        self.space = Space(ele, space_sleep)
        self.no_action = NoAction(ele, no_action_sleep)

        self.actions = [self.space, self.no_action]

    def sample(self):
        action = random.choice(list(range(len(self.actions))))
        return action


