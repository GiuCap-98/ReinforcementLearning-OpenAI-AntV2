import gym
import matplotlib.pyplot as plt
from stable_baselines3 import TD3

env = gym.make("Ant-v2")

#TD3_SB3
# Nota : total_timesteps=35000
#model = TD3("MlpPolicy", env, tensorboard_log="./td3/{}_tensorboard/", verbose=1)

#TD3_2
# Nota : total_timesteps=35000
#model = TD3("MlpPolicy", env, buffer_size=900000, learning_starts=8192, batch_size=128, tensorboard_log="./td3/{}_tensorboard/", verbose=1)

# TD3__3
model = TD3("MlpPolicy", env, buffer_size=900000, learning_starts=8000, batch_size=128, train_freq=1024, tensorboard_log="./td3/{}_tensorboard/", verbose=1)


model.learn(total_timesteps=28000)
obs = env.reset()
for i in range(5000):
    
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    env.render()
    
    if done:
      obs = env.reset()
      
env.close()