# https://github.com/Fumio-eisan/RL20200527/blob/master/CartPole_20200526.ipynb
import gym
import numpy as np
import time
import math
from statistics import mean, median
import matplotlib.pyplot as plt

class Agent:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        # 観察値の等差数列
        self.bins = {
            'cart_position': np.linspace(-2.4, 2.4, 3)[1:-1],
            'cart_velocity': np.linspace(-3, 3, 5)[1:-1],
            'pole_angle':    np.linspace(-0.5, 0.5, 5)[1:-1],
            'pole_velocity': np.linspace(-2, 2, 5)[1:-1]
        }
        # 状態の設定
        num_state = (
            (len(self.bins['cart_position']) + 1)
            * (len(self.bins['cart_velocity']) + 1)
            * (len(self.bins['pole_angle']) + 1)
            * (len(self.bins['pole_velocity']) + 1)
        )
        self.q_table = np.random.uniform(low=-1, high=1, size=(num_state, self.env.action_space.n))
        # qテーブルの更新履歴
        self.q_table_update = [[math.floor(time.time()), 0]] * num_state

        

    # 状態値の取得
    def __digitize_state(self, observation):
        """
        状態のデジタル化をする
        :param observation: 観察値
        :return: デジタル化した数値
        """
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        state = (
            np.digitize(cart_position, bins=self.bins['cart_position'])
            + (
                np.digitize(cart_velocity, bins=self.bins['cart_velocity'])
                * (len(self.bins['cart_position']) + 1)
            )
            + (
                np.digitize(pole_angle, bins=self.bins['pole_angle'])
                * (len(self.bins['cart_position']) + 1)
                * (len(self.bins['cart_velocity']) + 1)
            )
            + (
                np.digitize(pole_velocity, bins=self.bins['pole_velocity'])
                * (len(self.bins['cart_position']) + 1)
                * (len(self.bins['cart_velocity']) + 1)
                * (len(self.bins['pole_angle']) + 1)
            )
        )
        return state


    def get_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        # ε-greedy法で行動を選択
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.choice([0, 1])
        return action

    def update_q_table(self, state, action, reward, state_next):
        alpha = 0.2
        gamma = 0.99
        q_value_max = max(self.q_table[state_next])
        q_value_current = self.q_table[state, action]
        # qテーブルの更新
        self.q_table[state, action] = q_value_current \
            + alpha * (reward + gamma * q_value_max - q_value_current)
        self.q_table_update[state] = [math.floor(time.time()), self.q_table_update[state][1] + 1]

    def train(self, num_episode, num_step_max):
        num_solved = 0
        num_solved_max = 0

        # 履歴用
        self.episodes = []
        self.episode_rewards = []
        self.episode_rewards_mean = []
        self.episode_rewards_median = []

        for episode in range(num_episode):
            observation = self.env.reset()
            state = self.__digitize_state(observation)
            action = np.argmax(self.q_table[state])
            episode_reward = 0

            


env = gym.make("CartPole-v0")

goal_average_steps = 195
max_number_of_steps = 200
num_consecutive_iterations = 100
num_episodes = 200
last_time_steps = np.zeros(num_consecutive_iterations)

q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))

def free_action():
    for episode in range(num_episodes):
        # 環境の初期化
        observation = env.reset()

        episode_reward = 0
        for t in range(max_number_of_steps):
            # CartPoleの描画
            env.render()
            # ランダムで行動の選択
            action = np.random.choice([0, 1])

            # 行動の実行とフィードバックの取得
            observation, reward, done, info, _ = env.step(action)
            episode_reward += reward

            if done:
                print('%d Episode finished after %d time steps / mean %f' % (episode, t + 1,
                    last_time_steps.mean()))
                last_time_steps = np.hstack((last_time_steps[1:], [episode_reward]))
                break

        if (last_time_steps.mean() >= goal_average_steps): # 直近の100エピソードが195以上であれば成功
            print('Episode %d train agent successfuly!' % episode)
            break


