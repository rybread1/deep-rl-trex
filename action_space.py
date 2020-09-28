import pyautogui
import random
import time


class Action:
    def __init__(self, action):
        self.action = action

    def __repr__(self):
        return f'ActionObj("{self.action}")'

    def act(self):
        if self.action == 'space':
            pyautogui.press('space')
            time.sleep(.45)
            return 0

        if self.action == 'none':
            time.sleep(0.02)
            return 1


class ActionSpace:
    def __init__(self):
        self.space = Action('space')
        self.none = Action('none')

        self.actions = [self.space,  self.none]

    def sample(self):
        action = random.choice(list(range(len(self.actions))))
        return action


