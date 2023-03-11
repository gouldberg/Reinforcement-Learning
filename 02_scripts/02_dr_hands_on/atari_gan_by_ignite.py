import os, sys
import random
# import argparse
import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils


from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage

from ignite.contrib.handlers import tensorboard_logger as tb_logger

import gym
import gym.spaces


##############################################################################################
# --------------------------------------------------------------------------------------------
# InputWrapper
# --------------------------------------------------------------------------------------------

# Wrapper
# - Resize the input image from 210 * 160 (the standard Atari resolution) to a square size IMAGE_SIZE * IMAGE_SIZE
# - Apply np.moveaxis
# - Cast the image from bytes to float

# dimension input image will be rescaled
IMAGE_SIZE = 64

class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32)

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(
            observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


##############################################################################################
# --------------------------------------------------------------------------------------------
# Discriminator and Generator
# --------------------------------------------------------------------------------------------

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64

# Discriminator takes our scaled color image as input and,
# by applying 5 layers of convolutions, converts it into a single number
# passed through a Sigmoid nonlinearity.
# The output from Sigmoid is interpreted as the probability that Discriminator thinks our input image is from the real dataset.

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


# Generator takes as input a vector of random numbers (latent vector) and
# using the "transposed convolution" operation (deconvolution) converts this vector into a color image
# of the original resolution.

class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


##############################################################################################
# --------------------------------------------------------------------------------------------
# iterate_batches
# --------------------------------------------------------------------------------------------

BATCH_SIZE = 16

def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, _ = e.step(e.action_space.sample())
        
        # Check for non-zero mean of the observation is required
        # due to a bug in one of the games to prevent the flickering of an image 
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()


# --------------------------------------------------------------------------------------------
# process_batch
# --------------------------------------------------------------------------------------------

SAVE_IMAGE_EVERY_ITER = 1000

def process_batch(trainer, batch):
    gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
    gen_input_v.normal_(0, 1)
    gen_input_v = gen_input_v.to(device)
    batch_v = batch.to(device)
    gen_output_v = net_gener(gen_input_v)

    # train discriminator
    dis_optimizer.zero_grad()
    dis_output_true_v = net_discr(batch_v)
    dis_output_fake_v = net_discr(gen_output_v.detach())
    dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
    dis_loss.backward()
    dis_optimizer.step()

    # train generator
    gen_optimizer.zero_grad()
    dis_output_v = net_discr(gen_output_v)
    gen_loss = objective(dis_output_v, true_labels_v)
    gen_loss.backward()
    gen_optimizer.step()

    if trainer.state.iteration % SAVE_IMAGE_EVERY_ITER == 0:
        fake_img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
        trainer.tb.writer.add_image("fake", fake_img, trainer.state.iteration)
        real_img = vutils.make_grid(batch_v.data[:64], normalize=True)
        trainer.tb.writer.add_image("real", real_img, trainer.state.iteration)
        trainer.tb.writer.flush()
    return dis_loss.item(), gen_loss.item()


##############################################################################################
# --------------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------------

log = gym.logger
log.set_level(gym.logger.INFO)


# ----------
# device
device = torch.device("cuda")


# ----------
# environment

envs = [
    InputWrapper(gym.make(name))
    for name in ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')
]


# ----------
base_path = '/home/kswada/kw/reinforcement_learning'

args_name = 'trial'
writer = SummaryWriter(comment="-atari_gan_" + args_name)

save_path = os.path.join(base_path, '04_output/02_deep_reinforcement_learning_hands_on/halfcheetah/d4pg_trial/model')
os.makedirs(save_path, exist_ok=True)


# ----------
# discriminator and generator
input_shape = envs[0].observation_space.shape

net_discr = Discriminator(input_shape=input_shape).to(device)

net_gener = Generator(output_shape=input_shape).to(device)


# ----------
# objective:  basic Cross-Entropy Loss
objective = nn.BCELoss()


# ----------
# optimizer

LEARNING_RATE = 0.0001

# betas: coefficients used for computing running averages of gradient and
# its square (default: (0.9, 0.999))

gen_optimizer = optim.Adam(
    params=net_gener.parameters(), lr=LEARNING_RATE,
    betas=(0.5, 0.999))

dis_optimizer = optim.Adam(
    params=net_discr.parameters(), lr=LEARNING_RATE,
    betas=(0.5, 0.999))


# ----------
REPORT_EVERY_ITER = 100

true_labels_v = torch.ones(BATCH_SIZE, device=device)
fake_labels_v = torch.zeros(BATCH_SIZE, device=device)


# ----------
engine = Engine(process_batch)

tb = tb_logger.TensorboardLogger(log_dir=None)

engine.tb = tb

RunningAverage(output_transform=lambda out: out[1]).attach(engine, "avg_loss_gen")

RunningAverage(output_transform=lambda out: out[0]).attach(engine, "avg_loss_dis")

handler = tb_logger.OutputHandler(tag="train",
    metric_names=['avg_loss_gen', 'avg_loss_dis'])

tb.attach(engine, log_handler=handler,
            event_name=Events.ITERATION_COMPLETED)


# ----------
@engine.on(Events.ITERATION_COMPLETED)
def log_losses(trainer):
    if trainer.state.iteration % REPORT_EVERY_ITER == 0:
        log.info("%d: gen_loss=%f, dis_loss=%f",
                    trainer.state.iteration,
                    trainer.state.metrics['avg_loss_gen'],
                    trainer.state.metrics['avg_loss_dis'])

engine.run(data=iterate_batches(envs))
