import gym

env = gym.make('MountainCar-v0')
print(env)
env.reset()
# env.step()
env.render()
