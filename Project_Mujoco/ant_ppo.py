import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

env = gym.make("Ant-v2")

# PPO_SB3
#model = PPO("MlpPolicy", env,  verbose=1, tensorboard_log="./ppo/{}_tensorboard/")

# PPO_2
#model = PPO("MlpPolicy", env, n_steps=1000, batch_size = 128, verbose=1, clip_range = 0.1,  tensorboard_log="./ppo/{}_tensorboard/")

# PPO_3
model = PPO("MlpPolicy", env, learning_rate = 0.0003,  n_steps=200, batch_size = 128,gamma=0.99, gae_lambda = 0.95, verbose=1, clip_range = 0.2,  tensorboard_log="./ppo/{}_tensorboard/")

model.learn(total_timesteps=35000)
obs = env.reset()
for i in range(10000):
    
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
      obs = env.reset()
env.close()






