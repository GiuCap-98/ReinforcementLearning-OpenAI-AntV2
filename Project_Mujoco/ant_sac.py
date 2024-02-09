import gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

env = gym.make("Ant-v2")

# SAC_SB3
#model = SAC("MlpPolicy", env, verbose = 1 , tensorboard_log="./sac/{}_tensorboard/")

# SAC_2
#model = SAC("MlpPolicy", env, buffer_size=2000, learning_starts=1000, batch_size=256, tau=0.001, ent_coef = 'auto_0.1',  verbose=1, target_update_interval=2, tensorboard_log="./sac/{}_tensorboard/")


# SAC_3
model = SAC("MlpPolicy", env, buffer_size=2000, learning_starts=1000, batch_size=128, tau=0.005, ent_coef = 'auto_0.1',  verbose=1, target_update_interval=2, tensorboard_log="./sac/{}_tensorboard/")


model.learn(total_timesteps=30000)
obs = env.reset()
for i in range(10000):
    
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
      obs = env.reset()
env.close()