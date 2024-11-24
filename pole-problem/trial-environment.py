import gym
import numpy as np

# CartPole の読み込み
env = gym.make("CartPole-v0", render_mode="human")
observation = env.reset()

#実行
for i in range(50):
    action = 1 #右に押す
    # K = env.step(action) #情報を取得
    observation, reward, done, info, _ = env.step(action) #情報を取得
    print("observation = " + str(observation))
    print("reward = " + str(reward))

env.render() #ビジュアライズ
#終了
env.close()