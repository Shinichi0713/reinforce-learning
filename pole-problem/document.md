## 目的

オフポリシーの学習が可能なQ学習の練習台としてgymより提供されているpole問題を解いてみる。



##### 参考

[Pythonで始める強化学習！無料で試せる方法から具体的な実装例まで解説 | AI研究所](https://ai-kenkyujo.com/programming/language/python/python-3/)

[第3回 今更だけど基礎から強化学習を勉強する 価値推定編(TD法、モンテカルロ法、GAE) #Python - Qiita](https://qiita.com/pocokhc/items/312c817f9ddb0c5615da)

## 使う環境のスペック

[CartPole v0 · openai/gym Wiki](https://github.com/openai/gym/wiki/CartPole-v0)

| Num | Observation          | Min                  | Max                |
| --- | -------------------- | -------------------- | ------------------ |
| 0   | Cart Position        | -2.4                 | 2.4                |
| 1   | Cart Velocity        | -Inf                 | Inf                |
| 2   | Pole Angle           | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3   | Pole Velocity At Tip | -Inf                 | Inf                |

各種観察値を元に状態を数値で表します。
イメージとしては、

* 1の位が「cart_position」
* 10の位が「cart_velocity」
* 100の位が「pole_angle」
* 1000の位が「pole_velocity」

という風に桁で分けているイメージです。

## Action

| Num | Action                 |
| --- | ---------------------- |
| 0   | Push cart to the left  |
| 1   | Push cart to the right |



## 結論

Q学習により、pole問題を解くことが出来ることが確認できた。

この問題上では、poleを動かすエージェントは状態における、ベストな行動価値関数に従って行動を行う。


![1733034299131](image/document/1733034299131.png)
