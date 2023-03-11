import os, sys
import time
import numpy as np
import collections

import gym
import ptan

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
import torch.multiprocessing as mp


##############################################################################################
# --------------------------------------------------------------------------------------------
# RewardTracker
# --------------------------------------------------------------------------------------------

# This handles the full episode undiscounted reward, writes it into TensorBoard,
# and checks for the game solved condition.
class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


# --------------------------------------------------------------------------------------------
# AtariPGN
# --------------------------------------------------------------------------------------------

class AtariPGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariPGN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


# --------------------------------------------------------------------------------------------
# AtariP2C
# --------------------------------------------------------------------------------------------

class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # ----------
        conv_out_size = self._get_conv_out(input_shape)

        # ----------
        # return the policy with the probability distribution over our actions
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # return one single number, which will approximate the state's value.
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        # shared body
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        # return two heads
        return self.policy(conv_out), self.value(conv_out)


# --------------------------------------------------------------------------------------------
# unpack_batch
# --------------------------------------------------------------------------------------------

def unpack_batch(batch, net, last_val_gamma, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))

        # reward value already contains the discounted reward for REWARD_STEPS, as we use the ptan.ExperienceSourceFirstLast class.
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    # extra call to np.array() might look redundant, but without it, the performance of tensor creation
    # degrades 5-10x
    states_v = torch.FloatTensor(
        np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)

        # get V(state) appximation.
        last_vals_v = net(last_states_v)[1]

        # discounted and added to immediate rewards
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_t, ref_vals_v


##############################################################################################
# --------------------------------------------------------------------------------------------
# data_func:  executed in the child process.
# --------------------------------------------------------------------------------------------

ENV_NAME = "PongNoFrameskip-v4"
REWARD_BOUND = 18

# ENV_NAME = "BreakoutNoFrameskip-v4"
# REWARD_BOUND = 400


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))


TotalReward = collections.namedtuple('TotalReward', field_names='reward')

GAMMA = 0.99
REWARD_STEPS = 4

# NUM_ENVS: number of environemnts every child process will use to gather data.
# NUM_ENVS = 8 and PROCESSES_COUNT = 4 means that 32 parallel environemnts that we will get our training data from.
NUM_ENVS = 8
# MICRO_BATCH_SIZE: number of training samples that every child process needs to obtain
# before transferring those samples to the main process. 
MICRO_BATCH_SIZE = 32

def data_func(net, device, train_queue):

    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(
        lambda x: net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    micro_batch = []

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            data = TotalReward(reward=np.mean(new_rewards))
            train_queue.put(data)

        micro_batch.append(exp)
        if len(micro_batch) < MICRO_BATCH_SIZE:
            continue

        data = unpack_batch(
            micro_batch, net, device=device,
            last_val_gamma = GAMMA ** REWARD_STEPS)
        train_queue.put(data)
        micro_batch.clear()


##############################################################################################
# --------------------------------------------------------------------------------------------
# Pong Agent Learning:  Asynchronous Advantage Actor-Critic (A3C)
# --------------------------------------------------------------------------------------------

if __name__ == "__main__":


    # ----------
    # instruct the multiprocessing module about the kind of parallelism we want to use.
    # due to PyTorch multiprocessing limitations, spawn is the only option if you want to use GPU.
    mp.set_start_method('spawn')

    # instructing OpenMP library about the number of threads it can start.
    # We are implementing our own parallelism, by launching several processes,
    # extra threads overload the cores with frequent context switches, which negatively impacts
    # performance. To avoid this, we explicitly set the maximum number of threads OpenMP
    # can start with a single thread.
    # Since OpenMP is heavily used by the Gym and OpenCV libraries,
    # without this explicit specification, 3-4x performance drop.
    os.environ['OMP_NUM_THREADS'] = "1"


    # ----------
    # device (not torch.device("cuda"))
    device = "cuda"


    # ----------
    # environment
    env = make_env()


    # ----------
    # net:  Atari A2C
    net = AtariA2C(env.observation_space.shape, env.action_space.n).to(device)

    # The network is shared between all processes using PyTorch built-in capabilities,
    # allowing us to use the same nn.Module instance with all its weights in different processes
    # by calling the share_memory() method on NN creation.
    # Under the hood, this method has zero overhead for CUDA (as GPU memory is shared among all the host's processes),
    # or shared memory inter-process communication (IPC) in the case of CPU computation.
    net.share_memory()

    print(net)


    # ----------
    # optimizer
    # if 0.004 or larger --> does not converge ...
    LEARNING_RATE = 0.001

    # Normally eps, which is added to denominator to prevent zero division situation,
    # is set to some small number such as 1e-8 or 1e-10,
    # but in our case, these values turned out to be too small and the method does not converge at all.
    # so we set large eps such as 1e-3
    EPS_VAL = 1e-3
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=EPS_VAL)



    # ----------
    # mp.Queue:
    #  A concurrent multi-producer, multi-consumer FIFO queue with transparent serialization
    #  and deserialization of objects placed in the queue.

    # PROCESSES_COUNT:
    #   number of child processes that will gather training data for us.
    PROCESSES_COUNT = 4
    # queue we will use to send data from the child process to our master process, which will perform training.
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)

    data_proc_list = []

    # mp.Process:
    #  A piece of code that is run in the child process and methods to control it from the parent process.
    # The child processes have the responsibility to build mini-batches of data,
    # copy them to GPU, and then pass tensors via the queue.
    # Then, in the main process, we concatenate those tensors and perform the training.
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func,
                                args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)


    # ----------
    batch_states = []
    batch_actions = []
    batch_vals_ref = []
    step_idx = 0
    batch_size = 0
    BATCH_SIZE = 128

    ENTROPY_BETA = 0.01
    CLIP_GRAD = 0.1


    args_name = 'trial'
    writer = SummaryWriter(comment=f"-a3c-data_pong_{args_name}")

    try:
        with RewardTracker(writer, REWARD_BOUND) as tracker:
            with ptan.common.utils.TBMeanTracker(writer, 100) as tb_tracker:
                while True:
                    train_entry = train_queue.get()
                    if isinstance(train_entry, TotalReward):
                        if tracker.reward(train_entry.reward, step_idx):
                            break
                        continue

                    states_t, actions_t, vals_ref_t = train_entry
                    batch_states.append(states_t)
                    batch_actions.append(actions_t)
                    batch_vals_ref.append(vals_ref_t)
                    step_idx += states_t.size()[0]
                    batch_size += states_t.size()[0]
                    if batch_size < BATCH_SIZE:
                        continue

                    states_v = torch.cat(batch_states)
                    actions_t = torch.cat(batch_actions)
                    vals_ref_v = torch.cat(batch_vals_ref)
                    batch_states.clear()
                    batch_actions.clear()
                    batch_vals_ref.clear()
                    batch_size = 0

                    # ----------
                    optimizer.zero_grad()
                    logits_v, value_v = net(states_v)

                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.detach()
                    size = states_v.size()[0]
                    log_p_a = log_prob_v[range(size), actions_t]
                    log_prob_actions_v = adv_v * log_p_a
                    loss_policy_v = -log_prob_actions_v.mean()

                    prob_v = F.softmax(logits_v, dim=1)
                    ent = (prob_v * log_prob_v).sum(dim=1).mean()
                    entropy_loss_v = ENTROPY_BETA * ent

                    loss_v = entropy_loss_v + loss_value_v + loss_policy_v
                    loss_v.backward()
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    optimizer.step()
                    # ----------

                    # ----------
                    tb_tracker.track("advantage", adv_v, step_idx)
                    tb_tracker.track("values", value_v, step_idx)
                    tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                    tb_tracker.track("loss_value", loss_value_v, step_idx)
                    tb_tracker.track("loss_total", loss_v, step_idx)

    # finally block can be executed due to an exception (Ctrl+C, for example) or the game solved condition,
    # we terminate the child processes and wait for them.
    # This is required to make sure taht there are no leftover processes. 
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
