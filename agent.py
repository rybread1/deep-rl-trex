import numpy as np
import tensorflow as tf
import datetime
from memory import ReplayMemory
import progressbar
import math


class Agent:
    def __init__(self,
                 environment,
                 optimizer,
                 memory_length,
                 dueling=True,
                 loss='mse',
                 load_weights=None,
                 save_weights=None,
                 verbose_action=False):

        self.environment = environment
        self._optimizer = optimizer
        self._loss = loss
        self.memory = ReplayMemory(memory_length)
        self.dueling = dueling

        # Initialize discount and exploration rate, etc
        self.total_steps = 0
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.00005
        self.tau = 0.05
        self.pretraining_steps = 0

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model(how='hard')

        if load_weights:
            self.load_weights(load_weights)

        self.save_weights_fp = save_weights
        self.start_time = datetime.datetime.now()
        self.verbose_action = verbose_action

    def load_weights(self, weights_fp):
        if weights_fp:
            print('loading weights...')
            self.q_network.load_weights(weights_fp)
            self.align_target_model(how='hard')

    def save_weights(self, weights_fp):
        if weights_fp:
            self.q_network.save_weights(weights_fp)

    def set_epsilon_decay_schedule(self, epsilon, epsilon_min, annealed_steps):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = math.log(self.epsilon / self.epsilon_min) / annealed_steps

    def set_beta_schedule(self, beta_start, beta_max, annealed_samplings):
        self.memory.beta = beta_start
        self.memory.beta_max = beta_max
        self.memory.beta_increment_per_sampling = (self.memory.beta_max - self.memory.beta) / annealed_samplings

    def predict(self, state, use_target=False):
        if use_target:
            return self.target_network.predict(state)
        else:
            return self.q_network.predict(state)

    def _decay_epsilon(self):
        self.epsilon = self.epsilon * np.exp(-self.epsilon_decay)

    def store(self, state, action, reward, next_state, terminated):
        self.memory.add((state, action, reward, next_state, terminated))
        self.total_steps += 1

        if (self.epsilon > self.epsilon_min) and (self.memory.length > self.pretraining_steps):
            self._decay_epsilon()

    def batch_store(self, batch_load):
        batch_load[-2][2] = -0.1  # custom reward altering
        for row in batch_load:
            self.store(*row)

    def _build_compile_model(self):
        inputs = tf.keras.layers.Input(shape=(32, 290, 4))
        conv1 = tf.keras.layers.Conv2D(32, (8, 8), strides=4, padding='same', activation='relu')(inputs)
        conv2 = tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu')(conv1)
        conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu')(conv2)
        conv3 = tf.keras.layers.Flatten()(conv3)

        advt = tf.keras.layers.Dense(256, activation='relu')(conv3)
        final = tf.keras.layers.Dense(2)(advt)

        if self.dueling:
            value = tf.keras.layers.Dense(256, activation='relu')(conv3)
            value = tf.keras.layers.Dense(1)(value)

            advt = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x, axis=1, keepdims=True))(final)
            final = tf.keras.layers.Add()([value, advt])

        model = tf.keras.models.Model(inputs=inputs, outputs=final)
        model.compile(optimizer=self._optimizer,
                      loss=self._loss,
                      metrics=['accuracy'])
        return model

    def align_target_model(self, how):
        assert how in ('hard', 'soft'), '"how" must be either "hard" or "soft"'

        if how == 'hard':
            self.target_network.set_weights(self.q_network.get_weights())

        elif how == 'soft':
            for t, e in zip(self.target_network.trainable_variables, self.q_network.trainable_variables):
                t.assign(t * (1 - self.tau) + (e * self.tau))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.environment.action_space.sample()
            if self.verbose_action:
                print(f'action: {action}, q: random')
            return action

        q_values = self.predict(state, use_target=False)
        action = np.argmax(q_values[0])
        if self.verbose_action:
            print(f'action: {action}, q: {q_values}')
        return action

    def train(self, batch, is_weights):

        td_errors = np.zeros(len(batch))
        states = np.zeros((len(batch), 32, 290, 4))
        targets = np.zeros((len(batch), 2))

        for i, (state, action, reward, next_state, terminated) in enumerate(batch):
            target, td_error = self._get_target(state, action, reward, next_state, terminated)
            states[i] = state.reshape(32, 290, 4)
            targets[i] = target
            td_errors[i] = td_error

        self.q_network.fit(states, targets, sample_weight=is_weights, batch_size=32, epochs=1, verbose=0)
        self.align_target_model(how='soft')

        return td_errors

    def replay(self, batch_size, epoch_steps=None):

        num_batches = 1
 
        if epoch_steps:
            num_batches = int(np.max([np.floor(epoch_steps / 4), 1]))

        bar = progressbar.ProgressBar(maxval=num_batches,
                                      widgets=[f'training - ', progressbar.widgets.Counter(), f'/{num_batches} ',
                                               progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for i in range(num_batches):
            leaf_idx, batch, is_weights = self.memory.get_batch(batch_size)  # prioritized experience replay
            td_errors = self.train(batch, is_weights)
            self.memory.update_sum_tree(leaf_idx, td_errors)

            bar.update(i + 1)

        bar.finish()
        self.save_weights(self.save_weights_fp)

    def _get_target(self, state, action, reward, next_state, terminated):
        target = self.predict(state, use_target=False)
        prev_target = target[0][action]

        if terminated:
            target[0][action] = reward
        else:
            a = np.argmax(self.predict(next_state, use_target=False)[0])
            target[0][action] = reward + (self.gamma * self.predict(next_state, use_target=True)[0][a])  # double Q Network

        td_error = abs(prev_target - target[0][action])

        return target, td_error



