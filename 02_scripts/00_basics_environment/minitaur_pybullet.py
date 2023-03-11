
import gym
import pybullet_envs


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# check environemnt
# -----------------------------------------------------------------------------------------------------------

ENV_ID = "MinitaurBulletEnv-v0"
RENDER = True


spec = gym.envs.registry.spec(ENV_ID)

spec._kwargs['render'] = RENDER

env = gym.make(ENV_ID)


# ----------
# 28 numbers corresponding to different physical parametrs of the robot:
# velocity, position, and acceleration.
print("Observation space:", env.observation_space)

# 8 numbers for parameters of the motors, 2 in every leg (1 in every knee)
print("Action space:", env.action_space)


# reward of this environment is the distance traveled by the robot minus the enegy spent.

print(env)

print(env.reset())

input("Press any key to exit\n")
env.close()
