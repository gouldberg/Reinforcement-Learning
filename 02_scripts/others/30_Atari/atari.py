# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from datetime import datetime

import tensorflow as tf
import tensorflow.contrib.layers as layers

import gym

# ----------
# REFERENCE
# https://github.com/imai-laboratory/dqn
# https://github.com/imai-laboratory/rlsaber/tree/master/rlsaber


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# network
# -----------------------------------------------------------------------------------------------------------

def _make_cnn(convs, hiddens, inpt, num_actions, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope('convnet'):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(
                    out,
                    num_outputs=num_outputs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding='VALID',
                    activation_fn=tf.nn.relu
                )
        conv_out = layers.flatten(out)
        with tf.variable_scope('action_value'):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(
                    action_out, num_outputs=hidden, activation_fn=tf.nn.relu)

            # final output layer
            action_scores = layers.fully_connected(
                action_out, num_outputs=num_actions, activation_fn=None)
        return action_scores

def make_cnn(convs, hiddens):
    return lambda *args, **kwargs: _make_cnn(convs, hiddens, *args, **kwargs)


# -----------------------------------------------------------------------------------------------------------
# build_graph.py
# -----------------------------------------------------------------------------------------------------------

def huber_loss(loss, delta=1.0):
    return tf.where(
        tf.abs(loss) < delta,
        tf.square(loss) * 0.5,
        delta * (tf.abs(loss) - 0.5 * delta)
    )


def build_train(q_func,
                num_actions,
                state_shape,
                optimizer,
                batch_size=32,
                grad_norm_clipping=10.0,
                gamma=1.0,
                scope='deepq',
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        obs_t_ph = tf.placeholder(tf.float32, [None] + state_shape, name='obs_t')
        act_t_ph = tf.placeholder(tf.int32, [None], name='action')
        rew_t_ph = tf.placeholder(tf.float32, [None], name='reward')
        obs_tp1_ph = tf.placeholder(tf.float32, [None] + state_shape, name='obs_tp1')
        done_mask_ph = tf.placeholder(tf.float32, [None], name='done')

        # q network
        q_t = q_func(obs_t_ph, num_actions, scope='q_func')
        q_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/q_func'.format(scope))

        # target q network
        q_tp1 = q_func(obs_tp1_ph, num_actions, scope='target_q_func')
        target_q_func_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, '{}/target_q_func'.format(scope))

        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)
        q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf.reduce_mean(huber_loss(td_error))

        # update parameters
        gradients = optimizer.compute_gradients(errors, var_list=q_func_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)
        optimize_expr = optimizer.apply_gradients(gradients)

        # update target network
        update_target_expr = []
        sorted_vars = sorted(q_func_vars, key=lambda v: v.name)
        sorted_target_vars = sorted(target_q_func_vars, key=lambda v: v.name)
        for var, var_target in zip(sorted_vars, sorted_target_vars):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # taking best action
        actions = tf.argmax(q_t, axis=1)
        def act(obs, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            return sess.run(actions, feed_dict={obs_t_ph: obs})

        # update network
        def train(obs_t, act_t, rew_t, obs_tp1, done, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            feed_dict = {
                obs_t_ph: obs_t,
                act_t_ph: act_t,
                rew_t_ph: rew_t,
                obs_tp1_ph: obs_tp1,
                done_mask_ph: done
            }
            td_error_val, _ = sess.run(
                [td_error, optimize_expr], feed_dict=feed_dict)
            return td_error_val

        # synchronize target network
        def update_target(sess=None):
            if sess is None:
                sess = tf.get_default_session()
            sess.run(update_target_expr)

        # estimate q value
        def q_values(obs, sess=None):
            if sess is None:
                sess = tf.get_default_session()
            return sess.run(q_t, feed_dict={obs_t_ph: obs})

        return act, train, update_target, q_values


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# action.spy
# -----------------------------------------------------------------------------------------------------------

action_space = {
    'PongDeterministic-v4': [1, 2, 3],
    'BreakoutDeterministic-v4': [1, 2, 3],
    'SpaceInvadersDeterministic-v4': [1, 2, 3, 4]
}

def get_action_space(env):
    return action_space[env]


# -----------------------------------------------------------------------------------------------------------
# agent.py
# -----------------------------------------------------------------------------------------------------------

class Agent:
    def __init__(self,
                q_func,
                actions,
                state_shape,
                replay_buffer,
                exploration,
                optimizer,
                gamma,
                grad_norm_clipping,
                phi=lambda s: s,
                batch_size=32,
                train_freq=4,
                learning_starts=1e4,
                target_network_update_freq=1e4):
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.actions = actions
        self.learning_starts = learning_starts
        self.target_network_update_freq = target_network_update_freq
        self.exploration = exploration
        self.replay_buffer = replay_buffer
        self.phi = phi

        self._act,\
        self._train,\
        self._update_target,\
        self._q_values = build_train(
            q_func=q_func,
            num_actions=len(actions),
            state_shape=state_shape,
            optimizer=optimizer,
            gamma=gamma,
            grad_norm_clipping=grad_norm_clipping
        )

        self.last_obs = None
        self.t = 0

    def act(self, obs, reward, training):
        # transpose state shape to WHC
        obs = self.phi(obs)
        # take the best action
        action = self._act([obs])[0]

        # epsilon greedy exploration
        if training:
            action = self.exploration.select_action(
                self.t, action, len(self.actions))

        if training:
            if self.t % self.target_network_update_freq == 0:
                self._update_target()

            if self.t > self.learning_starts and self.t % self.train_freq == 0:
                obs_t,\
                actions,\
                rewards,\
                obs_tp1,\
                dones = self.replay_buffer.sample(self.batch_size)
                td_errors = self._train(obs_t, actions, rewards, obs_tp1, dones)

            if self.last_obs is not None:
                self.replay_buffer.append(
                    obs_t=self.last_obs,
                    action=self.last_action,
                    reward=reward,
                    obs_tp1=obs,
                    done=False
                )

        self.t += 1
        self.last_obs = obs
        self.last_action = action
        return self.actions[action]

    def stop_episode(self, obs, reward, training=True):
        if training:
            # transpose state shape to WHC
            obs = self.phi(obs)
            self.replay_buffer.append(
                obs_t=self.last_obs,
                action=self.last_action,
                reward=reward,
                obs_tp1=obs,
                done=True
            )
        self.last_obs = None
        self.last_action = 0


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# train.py
# -----------------------------------------------------------------------------------------------------------

base_path = 'C:\\Users\\kosei-wada\\Desktop\\rl'

date = datetime.now().strftime("%Y%m%d%H%M%S")

args0 = {
    'env': 'PongDeterministic-v4',
    'outdir': date,
    'logdir': date,
    'load': None,
    'render': True,
    'demo': True,
    'record': True
}

class Sample:
    def __init__(self, ):
        pass

args = Sample()

for key in args0.keys():
    setattr(args, key, args0[key])

outdir = os.path.join(base_path, '04_output\\pong', args.outdir)
logdir = os.path.join(base_path, '04_output\\pong\\logs', args.logdir)

if not os.path.exists(outdir):
    os.makedirs(outdir)


# ----------
# atari_constants.py

# REPLAY_BUFFER_SIZE = 10 ** 5
# BATCH_SIZE = 32
# LEARNING_START_STEP = 10 ** 4
# FINAL_STEP = 10 ** 7
# GAMMA = 0.99
# UPDATE_INTERVAL = 4
# TARGET_UPDATE_INTERVAL = 10 ** 4
# STATE_WINDOW = 4
# EXPLORATION_TYPE = 'linear'
# EXPLORATION_EPSILON = 0.1
# EXPLORATION_DURATION = 10 ** 6
# CONVS = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
# FCS = [512]
#
# LR = 2.5e-4
# OPTIMIZER = 'rmsprop'
# MOMENTUM = 0.95
# EPSILON = 1e-2
# GRAD_CLIPPING = 10.0
#
# DEVICE = '/gpu:0'
# MODEL_SAVE_INTERVAL = 10 ** 6
# EVAL_INTERVAL = 10 ** 5
# EVAL_EPISODES = 10
# RECORD_EPISODES = 3


# ----------
# box_constants.py
REPLAY_BUFFER_SIZE = 10 ** 5
BATCH_SIZE = 16
LEARNING_START_STEP = 500
FINAL_STEP = 10 ** 6
GAMMA = 0.9
UPDATE_INTERVAL = 1
TARGET_UPDATE_INTERVAL = 100
STATE_WINDOW = 1
EXPLORATION_TYPE = 'constant'
EXPLORATION_EPSILON = 0.1
EXPLORATION_DURATION = 10 ** 4
CONVS = []
FCS = [50, 50]

LR = 1e-2
OPTIMIZER = 'adam'
MOMENTUM = 0.95
EPSILON = 1e-2
GRAD_CLIPPING = 10.0

DEVICE = '/gpu:0'
# DEVICE = '/cpu:0'
MODEL_SAVE_INTERVAL = 10 ** 4
EVAL_INTERVAL = 10 ** 4
EVAL_EPISODES = 10
RECORD_EPISODES = 3


# ----------
# environments
env = gym.make(args.env)

print(env.observation_space)

# box environment
if len(env.observation_space.shape) == 1:
    # constants = box_constants
    actions = range(env.action_space.n)
    state_shape = [env.observation_space.shape[0], STATE_WINDOW]
    state_preprocess = lambda state: state
    # (window_size, dim) -> (dim, window_size)
    phi = lambda state: np.transpose(state, [1, 0])
# atari environment
else:
    # constants = atari_constants
    actions = get_action_space(args.env)
    state_shape = [84, 84, STATE_WINDOW]
    def state_preprocess(state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (210, 160))
        state = cv2.resize(state, (84, 110))
        state = state[18:102, :]
        return np.array(state, dtype=np.float32) / 255.0
    # (window_size, H, W) -> (H, W, window_size)
    phi = lambda state: np.transpose(state, [1, 2, 0])


# ----------
# wrap gym environment
env = EnvWrapper(
    env,
    s_preprocess=state_preprocess,
    r_preprocess=lambda r: np.clip(r, -1, 1)
)


# ----------
# save constant variables
# dump_constants(constants, os.path.join(outdir, 'constants.json'))


# ----------
# exploration
if EXPLORATION_TYPE == 'linear':
    duration = EXPLORATION_DURATION
    explorer = LinearDecayExplorer(final_exploration_step=duration)
else:
    explorer = ConstantExplorer(EXPLORATION_EPSILON)


# ----------
# optimizer
if OPTIMIZER == 'adam':
    optimizer = tf.train.AdamOptimizer(LR)
else:
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=LR, momentum=MOMENTUM,
        epsilon=EPSILON)


# ----------
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

sess = tf.Session()
sess.__enter__()

model = make_cnn(convs=CONVS, hiddens=FCS)

agent = Agent(
    model,
    actions,
    state_shape,
    replay_buffer,
    explorer,
    optimizer,
    gamma=GAMMA,
    grad_norm_clipping=GRAD_CLIPPING,
    phi=phi,
    learning_starts=LEARNING_START_STEP,
    batch_size=BATCH_SIZE,
    train_freq=UPDATE_INTERVAL,
    target_network_update_freq=TARGET_UPDATE_INTERVAL
)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
if args.load is not None:
    saver.restore(sess, args.load)

train_writer = tf.summary.FileWriter(logdir, sess.graph)
tflogger = TfBoardLogger(train_writer)
tflogger.register('reward', dtype=tf.float32)
tflogger.register('eval_reward', dtype=tf.float32)
jsonlogger = JsonLogger(os.path.join(outdir, 'reward.json'))


# callback on the end of episode
def end_episode(reward, step, episode):
    tflogger.plot('reward', reward, step)
    jsonlogger.plot(reward=reward, step=step, episode=episode)


def after_action(state, reward, global_step, local_step):
    if global_step > 0 and global_step % MODEL_SAVE_INTERVAL == 0:
        path = os.path.join(outdir, 'model.ckpt')
        saver.save(sess, path, global_step=global_step)


evaluator = Evaluator(
    env=copy.deepcopy(env),
    state_shape=state_shape[:-1],
    state_window=STATE_WINDOW,
    eval_episodes=EVAL_EPISODES,
    recorder=Recorder(outdir) if args.record else None,
    record_episodes=RECORD_EPISODES
)

should_eval = lambda step, episode: step > 0 and step % EVAL_INTERVAL == 0

end_eval = lambda s, e, r: tflogger.plot('eval_reward', np.mean(r), s)

trainer = Trainer(
    env=env,
    agent=agent,
    render=args.render,
    state_shape=state_shape[:-1],
    state_window=STATE_WINDOW,
    final_step=FINAL_STEP,
    after_action=after_action,
    end_episode=end_episode,
    training=not args.demo,
    evaluator=evaluator,
    should_eval=should_eval,
    end_eval=end_eval
)

trainer.start()


# ----------
sess.close()

cv2.destroyAllWindows()
