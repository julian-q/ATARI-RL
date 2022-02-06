from models.utils.measures import LL
import numpy as np
import pickle

class SimpleREINFORCE:
    def __init__(self, layers):
        self.layers = layers
        self.log_likelihood = LL()

        self.init_parameters()
        
    def init_parameters(self):
        self.parameters = {}

        for i, layer in enumerate(self.layers):
            self.parameters[f'w{i + 1}'] = np.random.randn(layer['in_channels'], layer['out_channels'])
            self.parameters[f'b{i + 1}'] = np.random.randn(layer['out_channels'])

    def save_parameters(self):
        with open('./models/simple_reinforce.pkl', 'wb') as f:
            pickle.dump(self.parameters, f)

    def load_parameters(self):
        with open('./models/simple_reinforce.pkl', 'rb') as f:
            self.parameters = pickle.load(f)

    def forward(self, s, training=False):
        cache = {}

        for i, layer in enumerate(self.layers):
            w, b = self.parameters[f'w{i + 1}'], self.parameters[f'b{i + 1}']
            a_prev = s if i == 0 else cache[f'a{i}']

            z = a_prev @ w + b
            a = layer['activation'].evaluate(z)

            cache[f'z{i + 1}'], cache[f'a{i + 1}'] = z, a

        if training:
            return a, cache
        else:
            return a

    def backward(self, x, y, cache):
        gradients = {}

        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1:
                a = cache[f'a{i + 1}']
                da = self.log_likelihood.derivative(y, a)
            else:
                dz_next = gradients[f'dz{i + 2}']
                w_next = self.parameters[f'w{i + 2}']
                da = dz_next @ w_next.T

            z = cache[f'z{i + 1}']
            dz = da * layer['activation'].derivative(z)

            gradients[f'da{i + 1}'], gradients[f'dz{i + 1}'] = da, dz

            a_prev = x if i == 0 else cache[f'a{i}']
            n = y.shape[0]

            dw = (a_prev.T @ dz) / n
            db = np.sum(dz, axis=0) / n

            gradients[f'dw{i + 1}'], gradients[f'db{i + 1}'] = dw, db

        return gradients

    def train(self, env, n_episodes, alpha):
        for episode_num in range(n_episodes):
            all_gradients = []
            all_rewards = []

            prev_observation = env.reset()
            observation, reward, done, info = env.step(env.action_space.sample())
            x = (observation - prev_observation).reshape(1, -1)

            play_num = 0
            play_length = 0

            while not done:
                env.render()

                action_prob, cache = self.forward(x, training=True)
                action = 1 if np.random.rand() < action_prob else 0
                print('action_prob:', action_prob, 'action:', action)

                prev_observation = observation
                observation, reward, done, info = env.step(action + 2) # because this env is weird
                x = (observation - prev_observation).reshape(1, -1)

                play_length += 1

                y = np.array([[action]])
                gradients = self.backward(x, y, cache)
                all_gradients.append(gradients)

                if reward != 0:
                    print(f'play {play_num + 1:2d} finished after {play_length} steps with reward {reward:4.1f}')

                    all_rewards.extend([reward] * play_length)

                    play_num += 1
                    play_length = 0

            print(f'episode {episode_num + 1} finished')

            for gradients, reward in zip(all_gradients, all_rewards):
                for i in range(len(self.layers)):
                    self.parameters[f'w{i + 1}'] += alpha * reward * gradients[f'dw{i + 1}']
                    self.parameters[f'b{i + 1}'] += alpha * reward * gradients[f'db{i + 1}']

            self.save_parameters()

        env.close()


