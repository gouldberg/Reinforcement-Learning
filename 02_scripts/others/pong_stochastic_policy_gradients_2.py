# -*- coding: utf-8 -*-

import os, sys
import numpy as np
# import cPickle as pickle
# import _pickle.pickle as pickle
import pickle
import gym


# ----------
# REFERENCE
# https://colab.research.google.com/drive/1KZeGjxS7OUHKotsuyoT0DtxVzKaUIq4B?usp=sharing#scrollTo=ZJUybWUALvQz

# THIS IS BASED ON
# Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

# Karpathy blog:
# http://karpathy.github.io/2016/05/31/rl/


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------------------------------------

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    # ----------
    # return I.astype(np.float).ravel()
    return I.astype(float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    # for t in reversed(xrange(0, r.size)):
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# -----------------------------------------------------------------------------------------------------------
# policy network (agent)
# -----------------------------------------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


# take the state of the game and decide what we should do (move Up or Down)
def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2 (move Up), and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# other functions
# -----------------------------------------------------------------------------------------------------------

from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 144)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    plt.close(anim._fig)
    display(HTML(anim.to_jshtml()))


def model_step(model, observation, prev_x):
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, _ = policy_forward(x)
    action = 2 if aprob >= 0.5 else 3  # roll the dice!

    return action, prev_x


def play_game(env, model):
    observation = env.reset()

    frames = []
    cumulated_reward = 0

    prev_x = None # used in computing the difference frame

    for t in range(1000):
        frames.append(env.render(mode = 'rgb_array'))
        action, prev_x = model_step(model, observation, prev_x)
        observation, reward, done, info = env.step(action)
        cumulated_reward += reward
        if done:
            print("Episode finished after {} timesteps, accumulated reward = {}".format(t+1, cumulated_reward))
            break
    print("Episode finished without success, accumulated reward = {}".format(cumulated_reward))
    env.close()
    display_frames_as_gif(frames)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# environment
# -----------------------------------------------------------------------------------------------------------

# not found
# env = gym.make('ALE/Pong-v5')

env = gym.make("Pong-v4")
# env = gym.wrappers.Monitor(env, '.', force=True)

print(env.unwrapped.get_action_meanings())
print(env.action_space)
print(env.observation_space)

observation = env.reset()
print(observation)


# -----------------------------------------------------------------------------------------------------------
# Run demo for environment
# -----------------------------------------------------------------------------------------------------------

# from pyvirtualdisplay import Display
#
# # display = Display(visible=0, size=(1400, 900))
# display = Display(visible=True, size=(1400, 900))
#
# display.start()
#
# observation = env.reset()
#
# cumulated_reward = 0
#
# frames = []
#
# for t in range(1000):
# #     print(observation)
#     frames.append(env.render(mode='rgb_array'))
#     # very stupid agent, just makes a random action within the allowd action space
#     action = env.action_space.sample()
# #     print("Action: {}".format(t+1))
#     observation, reward, done, info = env.step(action)
# #     print(reward)
#     cumulated_reward += reward
#     if done:
#         print("Episode finished after {} timesteps, accumulated reward = {}".format(t+1, cumulated_reward))
#         break
# print("Episode finished without success, accumulated reward = {}".format(cumulated_reward))
#
# env.close()


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# hyper parameters
# -----------------------------------------------------------------------------------------------------------

# 2-layer policy network with 200 hidden layer units
H = 200  # number of hidden layer neurons

batch_size = 10  # every how many episodes to do a param update?

# learning_rate = 1e-4
learning_rate = 1e-3

gamma = 0.99  # discount factor for reward

decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

resume = False  # resume from previous checkpoint?

render = False


# -----------------------------------------------------------------------------------------------------------
# model initialization
# -----------------------------------------------------------------------------------------------------------

base_path = 'C:\\Users\\kosei-wada\\Desktop\\rl'

model_path = os.path.join(base_path, '04_output\\pong\\stochastic_policy_gradients_2\\model.p')


# ----------
D = 80 * 80  # input dimensionality: 80x80 grid

if resume:
    model = pickle.load(open(model_path, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

# grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch

# rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


# -----------------------------------------------------------------------------------------------------------
# training
# -----------------------------------------------------------------------------------------------------------

hist = []

observation = env.reset()


# ----------
prev_x = None  # used in computing the difference frame

xs, hs, dlogps, drs = [], [], [], []

running_reward = None

reward_sum = 0

total_episodes = 8000
episode_number = 0

while True:
    # if render: env.render()

    # preprocess the observation, set input to network to be difference image
    # Ideally you'd want to feed at least 2 frames to the policy network so that it can detect motion.
    # To make thins a bit simpler, feed difference frames to the network (i.e. subtraction of current and last frame)
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    # a + 1 reward if the ball went past the opponent, a - 1 reward if we missed the ball,
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            # for k, v in model.iteritems():
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        hist.append((episode_number, reward_sum, running_reward))
        print('ep %d:  resetting env. episode reward total was %f. running mean: %f' % (episode_number, reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open(model_path, 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

        if episode_number == total_episodes:
            break

    # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
    #     print('ep %d: game finished, reward: %f' % (episode_number, reward))


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# play
# -----------------------------------------------------------------------------------------------------------

model_path = os.path.join(base_path, '04_output\\pong\\stochastic_policy_gradients_2\\model.p')

# env = gym.make('ALE/Pong-v5')
env = gym.make("Pong-v0")

play_game(env, model_path)



#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# play:  this works
# -----------------------------------------------------------------------------------------------------------

from matplotlib import animation
import matplotlib.pyplot as plt


model_path = os.path.join(base_path, '04_output\\pong\\stochastic_policy_gradients_2\\model.p')
animation_path = os.path.join(base_path, '04_output\\pong\\stochastic_policy_gradients_2\\anim.gif')


# ----------
# env = gym.make('ALE/Pong-v5')
env = gym.make("Pong-v0")


# ----------
observation = env.reset()

frames = []
cumulated_reward = 0

prev_x = None # used in computing the difference frame

for t in range(1000):
    frames.append(env.render(mode='rgb_array'))

    # ----------
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    # ----------
    # forward the policy network and sample an action from the returned probability
    aprob, _ = policy_forward(x)
    action = 2 if aprob >= 0.5 else 3  # roll the dice!
    observation, reward, done, info = env.step(action)
    cumulated_reward += reward
    # ----------
    if done:
        print("Episode finished after {} timesteps, accumulated reward = {}".format(t+1, cumulated_reward))
        break

print("Episode finished after {} timesteps, accumulated reward = {}".format(t + 1, cumulated_reward))
env.close()


# ----------
plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 144)
patch = plt.imshow(frames[0])
plt.axis('off')

def animate(i):
    patch.set_data(frames[i])

anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
plt.close(anim._fig)


# -----------
anim.save(animation_path, writer='PillowWriter')

# display(HTML(anim.to_jshtml()))
# display(anim)
