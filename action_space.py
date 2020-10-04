from selenium.webdriver.common.keys import Keys
import time
import random


class Action:
    def __init__(self, ele):
        self.ele = ele

    def act(self):
        raise NotImplementedError()


class Space(Action):
    def __init__(self, ele):
        super().__init__(ele)

    def act(self):
        self.ele.send_keys(Keys.SPACE)
        time.sleep(.59)
        return 0


class NoAction(Action):
    def __init__(self, ele):
        super().__init__(ele)

    def act(self):
        time.sleep(0.02)
        return 1


class ActionSpace:
    def __init__(self, ele):
        self.space = Space(ele)
        self.no_action = NoAction(ele)

        self.actions = [self.space, self.no_action]

    def sample(self):
        action = random.choice(list(range(len(self.actions))))
        return action


