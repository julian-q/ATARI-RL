import gym
from models.utils.activations import ReLU, Sigmoid
from models.simple_reinforce import SimpleREINFORCE
import matplotlib.pyplot as plt

env = gym.make('Pong-v0', render_mode='rgb_array')
observation_dim = env.observation_space.shape[0] \
                * env.observation_space.shape[1] \
                * env.observation_space.shape[2]

model = SimpleREINFORCE([
    {'in_channels': observation_dim, 'out_channels': 100, 'activation': ReLU()},
    {'in_channels': 100, 'out_channels': 50, 'activation': ReLU()},
    {'in_channels': 50, 'out_channels': 1, 'activation': Sigmoid()}
])

model.train(env, 8000, 0.01)

