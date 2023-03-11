# -*- coding: utf-8 -*-

import os, sys
import gym

import numpy as np
import csv
from datetime import datetime
import random

from tensorflow.keras.layers import Input, Dense, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# util.py
# -----------------------------------------------------------------------------------------------------------

def now_str(str_format='%Y%m%d%H%M'):
    return datetime.now().strftime(str_format)


def idx2mask(idx, max_size):
    mask = np.zeros(max_size)
    mask[idx] = 1.0
    return mask


class RecordHistory:
    def __init__(self, csv_path, header):
        self.csv_path = csv_path
        self.header = header

    def generate_csv(self):
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def add_histry(self, history):
        history_list = [history[key] for key in self.header]
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(history_list)

    def add_list(self, array):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(array)


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# agent.model.py
# -----------------------------------------------------------------------------------------------------------

class Qnetwork:

    def __init__(self,
                 dim_state,
                 actions_list,
                 gamma=0.99,
                 lr=1e-3,
                 double_mode=True):
        self.dim_state = dim_state
        self.actions_list = actions_list
        self.action_len = len(actions_list)
        self.optimizer = Adam(lr=lr)
        self.gamma = gamma
        self.double_mode = double_mode

        self.main_network = self.build_graph()
        self.target_network = self.build_graph()
        self.trainable_network = self.build_trainable_graph(self.main_network)

    def build_graph(self):
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = self.action_len * 10
        nb_dense_2 = int(
            np.sqrt(self.action_len * 10 *
                    self.dim_state * 10))

        l_input = Input(shape=(self.dim_state,),
                        name='input_state')
        l_dense_1 = Dense(nb_dense_1,
                          activation='relu',
                          name='hidden_1')(l_input)
        l_dense_2 = Dense(nb_dense_2,
                          activation='relu',
                          name='hidden_2')(l_dense_1)
        l_dense_3 = Dense(nb_dense_3,
                          activation='relu',
                          name='hidden_3')(l_dense_2)
        l_output = Dense(self.action_len,
                         activation='linear',
                         name='output')(l_dense_3)

        model = Model(inputs=[l_input],
                      outputs=[l_output])
        model.summary()
        model.compile(optimizer=self.optimizer,
                      loss='mse')
        return model

    def build_trainable_graph(self, network):
        action_mask_input = Input(
            shape=(self.action_len,), name='a_mask_inp')
        q_values = network.output
        q_values_taken_action = Dot(
            axes=-1,
            name='qs_a')([q_values, action_mask_input])
        trainable_network = Model(
            inputs=[network.input, action_mask_input],
            outputs=q_values_taken_action)
        trainable_network.compile(
            optimizer=self.optimizer,
            loss='mse',
            metrics=['mae'])
        return trainable_network

    def sync_target_network(self, soft):
        weights = self.main_network.get_weights()
        target_weights = self.target_network.get_weights()
        for idx, w in enumerate(weights):
            target_weights[idx] *= (1 - soft)
            target_weights[idx] += soft * w
        self.target_network.set_weights(target_weights)

    def update_on_batch(self, exps):
        (state, action, reward, next_state, done) = zip(*exps)
        action_index = [self.actions_list.index(a) for a in action]
        action_mask = np.array([idx2mask(a, self.action_len) for a in action_index])
        state = np.array(state)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array(done)

        print('kw -- 1')
        next_target_q_values_batch = self.target_network.predict_on_batch(next_state)
        next_q_values_batch = self.main_network.predict_on_batch(next_state)

        if self.double_mode:
            future_return = [
                next_target_q_values[np.argmax(next_q_values)]
                for next_target_q_values, next_q_values
                in zip(next_target_q_values_batch,
                       next_q_values_batch)
            ]
        else:
            future_return = [
                np.max(next_q_values) for next_q_values
                in next_target_q_values_batch
            ]

        print('kw -- 2')
        y = reward + self.gamma * (1 - done) * future_return
        print('kw -- 3')
        loss, td_error = \
            self.trainable_network.train_on_batch(
             [state, action_mask], np.expand_dims(y, -1))
        print('kw -- 4')

        return loss, td_error


# -----------------------------------------------------------------------------------------------------------
# agent.policy.py
# -----------------------------------------------------------------------------------------------------------

class EpsilonGreedyPolicy:

    def __init__(self, q_network, epsilon):
        self.q_network = q_network
        self.epsilon = epsilon

    def get_action(self, state, actions_list):
        is_random_action = (np.random.uniform() < self.epsilon)
        if is_random_action:
            q_values = None
            action = np.random.choice(actions_list)
        else:
            state = np.reshape(state, (1, len(state)))
            q_values = self.q_network.main_network.predict_on_batch(state)[0]
            action = actions_list[np.argmax(q_values)]
        return action, self.epsilon, q_values


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# train:  setup
# -----------------------------------------------------------------------------------------------------------

"""
overview:
    OpenAI GymのPendulum-v0を環境として、Double_DQNの学習を行う

args:
    各種パラメータの設定値は、本コード中に明記される
    - result_dir:
        結果を出力するディレクトリのpath
    - max_episode:
        学習の繰り返しエピソード数(default: 300)
    - max_step:
        1エピソード内の最大ステップ数(default: 200)
    - gamma:
        割引率(default: 0.99)
output:
    result_dirで指定したpathに以下のファイルが出力される
    - episode_xxx.h5:
        xxxエピソードまで学習したDouble_DQNネットワークの重み
    - history.csv: エピソードごとの以下の3つのメトリックを記録するcsv
        - loss: DoubleDQNモデルを更新する際のlossの平均値
        - td_error: TD誤差の平均値
        - reward_avg: １ステップあたりの平均報酬

usage：
    python3 train.py
"""
env.close()

base_path = 'C:\\Users\\kosei-wada\\Desktop\\rl'


max_episode = 300  # 学習において繰り返す最大エピソード数
max_step = 200  # 1エピソードの最大ステップ数
n_warmup_steps = 10000  # warmupを行うステップ数
interval = 1  # モデルや結果を吐き出すステップ間隔
actions_list = [-1, 1]  # 行動(action)の取りうる値のリスト
gamma = 0.99  # 割引率
epsilon = 0.1  # ε-greedyのパラメータ
memory_size = 10000
batch_size = 32

result_dir = os.path.join(base_path, '\\04_output\\dqn_pendulum', now_str())


# -----------------------------------------------------------------------------------------------------------
# train:  instantiation
# -----------------------------------------------------------------------------------------------------------

os.makedirs(result_dir, exist_ok=True)
print(result_dir)

env = gym.make('Pendulum-v1')

dim_state = env.env.observation_space.shape[0]

q_network = Qnetwork(dim_state, actions_list, gamma=gamma)
policy = EpsilonGreedyPolicy(q_network, epsilon=epsilon)

header = ["num_episode", "loss", "td_error", "reward_avg"]

recorder = RecordHistory(os.path.join(result_dir, "history.csv"), header)

recorder.generate_csv()


# -----------------------------------------------------------------------------------------------------------
# train:  warmup
# -----------------------------------------------------------------------------------------------------------

print('warming up {:,} steps...'.format(n_warmup_steps))
memory = []
total_step = 0
step = 0
state = env.reset()[0]

while True:
    step += 1
    total_step += 1

    action = random.choice(actions_list)
    epsilon, q_values = 1.0, None

    next_state, reward, done, info, _ = env.step([action])

    # reward clipping
    if reward < -1:
        c_reward = -1
    else:
        c_reward = 1

    memory.append((state, action, c_reward, next_state, done))
    state = next_state

    if step > max_step:
        state = env.reset()
        step = 0
    if total_step > n_warmup_steps:
        break

memory = memory[-memory_size:]

print('warming up {:,} steps... done.'.format(n_warmup_steps))


# -----------------------------------------------------------------------------------------------------------
# train:  training
# -----------------------------------------------------------------------------------------------------------

print('training {:,} episodes...'.format(max_episode))

num_episode = 0
episode_loop = True

while episode_loop:
    num_episode += 1
    step = 0
    step_loop = True
    episode_reward_list, loss_list, td_list = [], [], []
    state = env.reset()[0]

    while step_loop:
        step += 1
        total_step += 1
        # ----------
        action, epsilon, q_values = policy.get_action(state, actions_list)
        # ----------
        next_state, reward, done, info = env.step([action])

        # reward clipping
        if reward < -1:
            c_reward = -1
        else:
            c_reward = 1

        memory.append((state, action, c_reward, next_state, done))
        episode_reward_list.append(c_reward)
        exps = random.sample(memory, batch_size)
        loss, td_error = q_network.update_on_batch(exps)
        loss_list.append(loss)
        td_list.append(td_error)

        q_network.sync_target_network(soft=0.01)
        state = next_state
        memory = memory[-memory_size:]

        # end of episode
        if step >= max_step:
            step_loop = False
            reward_avg = np.mean(episode_reward_list)
            loss_avg = np.mean(loss_list)
            td_error_avg = np.mean(td_list)
            print("{}episode  reward_avg:{} loss:{} td_error:{}".format(num_episode, reward_avg, loss_avg, td_error_avg))
            if num_episode % interval == 0:
                model_path = os.path.join(result_dir, 'episode_{}.h5'.format(num_episode))
                q_network.main_network.save(model_path)
                history = {
                    "num_episode": num_episode,
                    "loss": loss_avg,
                    "td_error": td_error_avg,
                    "reward_avg": reward_avg
                }
                recorder.add_histry(history)

    if num_episode >= max_episode:
        episode_loop = False

env.close()

print('training {:,} episodes... done.'.format(max_episode))
