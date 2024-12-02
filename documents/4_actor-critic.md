## 背景

方策勾配はQ関数による重み付け平均として定義

Q関数による方策評価が方策の学習に取り込まれていた。

状態行動空間が、状態数と自由度ともに小さいならばモデル化の必要なし。

方策ベース手法が高次元かつ連続な行動空間に対してい派、Q関数もモデル化=パラメータ化が必要となる。

エージェントが担う方策評価と方策評価の機能を分離して、個々にモデル化する手法が有効となる。

## Actor Critic法

Actor Critic法はValueベースとPolicyベースを組み合わせた手法

[Sarsa、Actor-Critic法の解説および倒立振子問題を解く #Python - Qiita](https://qiita.com/KokiSakano/items/0ab412ae9f69865f26d2)

Sarsaでは行動決定と状態更新は同一のQテーブルで行われていた

Actor Critic法では行動決定Actorと状態更新Criticを異なるテーブルで行い、これらを相互に更新し学習

更新式は以下。更新する場合のベースが価値関数であることが注意。

![1733088552762](image/4_actor-critic/1733088552762.png)

因みにQ学習とSarsaの更新式とも比較を行う。

![1733172594221](image/4_actor-critic/1733172594221.png)

Q学習では状態st+1における最大の価値を採用。

Sarsaでは次の行動at+1を決めて、価値Q(st+1, at+1)を採用

#### 前回までの確認

Q学習では次の行動をQ値の更新後に決定し、Sarsaでは次の行動をQ値の更新前に決定

Q学習は価値が最大となるような行動で状態更新する(Valueベース)ことから方策オフ型のアルゴリズム

Sarsaは戦略に基づいた行動で状態更新する(Policy)ことから方策オン型のアルゴリズム

実装での差分は以下

SARSAでは、次の行動 `next_action` を選択してからQ値を更新、Q学習では次の状態での最大のQ値を使用してQ値を更新。

Sarsa

```
class Env():
    def __init__(self, env, sarsa_class):
        self.env = env
        self.sarsa_class = sarsa_class

    def run(self):
        # 状態数を取得
        num_states = self.env.observation_space.shape[0]
        # 行動数を取得
        num_actions = self.env.action_space.n
        sarsa = self.sarsa_class(num_states, num_actions)

        step_list = []
        mean_list = []
        std_list = []
        for episode in range(NUM_EPISODES):
            observation = self.env.reset()  # 環境の初期化
            frames = []
            # 初期行動を求める
            action = sarsa.decide_action(observation, 0)
            for step in range(MAX_STEPS):
                if episode == NUM_EPISODES-1: frames.append(self.env.render(mode='rgb_array'))
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, _, done, _ = self.env.step(action)
                # 初期行動を求める
                action_next = sarsa.decide_action(observation_next, episode+1)
                # 報酬を与える
                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    if step < 195:
                        reward = -1  # 失敗したので-1の報酬を与える
                    else:
                        reward = 1  # 成功したので+1の報酬を与える
                else:
                    reward = 0
                # Qテーブル, Vを更新する
                sarsa.update_Qtable(observation, action, reward, observation_next, action_next)
                # 観測値を更新する
                observation = observation_next
                # 行動を更新する
                action = action_next

                # 終了時の処理
                if done:
                    step_list.append(step+1)
                    print('{}回目の試行は{}秒持ちました。(max:200秒)'.format(episode, step + 1))
                    break

            if episode == NUM_EPISODES-1:
                plt.figure()
                patch = plt.imshow(frames[0])
                plt.axis('off')

                def animate(i):
                    patch.set_data(frames[i])

                anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
                anim.save('movie_cartpole_v0_{}.gif'.format(episode+1), "ffmpeg")

        es = np.arange(0, len(step_list))
        plt.clf()
        plt.plot(es, step_list)
        plt.savefig("reward.png")
        for i, s in enumerate(step_list):
            if i < 100:
                mean_list.append(np.average(step_list[:i+1]))
                std_list.append(np.std(step_list[:i+1]))
            else:
                mean_list.append(np.average(step_list[i-100:i+1]))
                std_list.append(np.std(step_list[i-100:i+1]))
        mean_list = np.array(mean_list)
        std_list = np.array(std_list)
        plt.clf()
        plt.plot(es, mean_list)
        plt.fill_between(es, mean_list-std_list, mean_list+std_list, alpha=0.2)
        plt.savefig("mean_var.png")

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = Env(env, Sarsa)
    env.run()


```


Q学習

```
import gym
import numpy as np

# 環境の初期化
env = gym.make('CartPole-v1')

# Qテーブルの初期化
state_space = [20, 20, 20, 20]  # 状態空間の分割数
q_table = np.random.uniform(low=-1, high=1, size=(state_space + [env.action_space.n]))

# ハイパーパラメータの設定
alpha = 0.1  # 学習率
gamma = 0.99  # 割引率
epsilon = 1.0  # 探索率
epsilon_decay = 0.995  # 探索率の減衰
min_epsilon = 0.01  # 最小探索率
episodes = 10000  # エピソード数

# 状態を離散化する関数
def discretize_state(state):
    state_adj = (state - env.observation_space.low) * np.array(state_space) / (env.observation_space.high - env.observation_space.low)
    return tuple(state_adj.astype(int))

# Q学習のメインループ
for episode in range(episodes):
    state = discretize_state(env.reset())
    done = False
    total_reward = 0

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # 探索
        else:
            action = np.argmax(q_table[state])  # 利用

        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Q値の更新
        q_value_current = q_table[state + (action,)]
        q_value_max = np.max(q_table[next_state])
        q_table[state + (action,)] = q_value_current + alpha * (reward + gamma * q_value_max - q_value_current)

        state = next_state
        total_reward += reward

    # 探索率の減衰
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

env.close()

```
