import gym
from models.utils.activations import ReLU, Sigmoid
from models.simple_reinforce import SimpleREINFORCE
from models.reinforce import REINFORCE

env = gym.make('Pong-v0', render_mode='rgb_array')
observation_dim = env.observation_space.shape[0] \
                * env.observation_space.shape[1] \
                * env.observation_space.shape[2]

# note: wider hidden layers can increase performance if you have more RAM than my macbook
model = REINFORCE([
    {'in_channels': observation_dim, 'out_channels': 50, 'activation': ReLU()},
    {'in_channels': 50, 'out_channels': 10, 'activation': ReLU()},
    {'in_channels': 10, 'out_channels': 1, 'activation': Sigmoid()}
])

model.train(env, 8000, 0.01, 0.99)

