
import gym
import pybullet_envs


# REFERENCE:
# https://github.com/bulletphysics/bullet3


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# check environemnt
# https://gymnasium.farama.org/environments/mujoco/ant/#starting-state
# -----------------------------------------------------------------------------------------------------------

ENV_ID = "AntBulletEnv-v0"
RENDER = True


spec = gym.envs.registry.spec(ENV_ID)

spec._kwargs['render'] = RENDER

env = gym.make(ENV_ID)


# ----------
print(dir(env))
print(dir(spec))


# ----------
# Box(28,)
print("Observation space:", env.observation_space)

# Box(8,)
print("Action space:", env.action_space)


print("Max Episode Steps:", spec.max_episode_steps)


print("Reward Range:", env.reward_range)
print("Reward Threshold:", spec.reward_threshold)


# ----------
print(env)

print(env.reset())

input("Press any key to exit\n")
env.close()
