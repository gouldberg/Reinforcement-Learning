import gym


#############################################################################################################
# -----------------------------------------------------------------------------------------------------------
# check environemnt
# -----------------------------------------------------------------------------------------------------------

ENV_ID = "MountainCar-v0"

env = gym.make(ENV_ID)
spec = gym.envs.registry.spec(ENV_ID)


# ----------
print(dir(env))
print(dir(spec))


# ----------
# Box(2,)
# horizontal position of the car, and 
# car's velocity
print("Observation space:", env.observation_space)

# Discrete(3)
# 0: pushing the car to the left
# 1: no force
# 2: pushes the cat to the right
print("Action space:", env.action_space)


# 200
print("Max Episode Steps:", spec.max_episode_steps)


# Reward Range: (-inf, inf)
# Reward Threshold: -110.0
print("Reward Range:", env.reward_range)
print("Reward Threshold:", spec.reward_threshold)


# ----------
print(env.reset())


# ----------
action = 2

for _ in range(5):
    print(env.step(action))

