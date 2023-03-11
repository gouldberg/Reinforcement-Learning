
import gym
import pybullet_envs


# REFERENCE:
# https://github.com/bulletphysics/bullet3


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# check environemnt
# https://gymnasium.farama.org/environments/mujoco/half_cheetah/  --> here information also for v0
# -----------------------------------------------------------------------------------------------------------

# This environment is based on the work by P. Wawrzyński in “A Cat-Like Robot Real-Time Learning to Run”. 
# The HalfCheetah is a 2-dimensional robot consisting of 9 links and 8 joints connecting them
# (including two paws). 

# The goal is to apply a torque on the joints to make the cheetah run forward (right)
# as fast as possible, with a positive reward allocated based on the distance moved forward and
# a negative reward allocated for moving backward. 
# The torso and head of the cheetah are fixed, and the torque can only be applied
# on the other 6 joints over the front and back thighs (connecting to the torso), 
# shins (connecting to the thighs) and feet (connecting to the shins).

ENV_ID = "HalfCheetahBulletEnv-v0"
# ENV_ID = "RoboschoolHalfCheetah-v1"
RENDER = True


spec = gym.envs.registry.spec(ENV_ID)

spec._kwargs['render'] = RENDER

env = gym.make(ENV_ID)


# ----------
print(dir(env))
print(dir(spec))


# ----------
# Box(26,)
print("Observation space:", env.observation_space)

# Box(6,)
print("Action space:", env.action_space)

# 1000
print("Max Episode Steps:", spec.max_episode_steps)

# (-inf, inf), 3000.0
print("Reward Range:", env.reward_range)
print("Reward Threshold:", spec.reward_threshold)


# ----------
print(env)

print(env.reset())

input("Press any key to exit\n")
env.close()
