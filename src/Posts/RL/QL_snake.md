---
title: QL 貪吃蛇實作
# icon: material-symbols:add-notes-outline
order: 2
date: 2023-08-19
category:
  - ML

tag:
  - python
  - Imp

---



Q learning 為一種強化學習。

## 強化學習

"強化學習"，並非指學習很強，而是指一步一步慢慢增"強"的學習法。

RL為透過「跟世界的互動」來學習的，不像監督式學習，已經有人告訴他答案，RL 沒有，RL 必須要自己先去做做看，然後這個世界回饋給他的也不是正確答案，世界回饋給他的只有分數，做了這個動作得了幾分之類的。

<!-- more -->

Richard S. Sutton 給出的定義為： “Reinforcement learning is learning what to do—how to map situations to actions—to maximize a numerical reward signal.

RL主要的算法有
- Q learning
- Sarsa
- Policy Gradients
- Actor-Critic
- Monte-carlo learning
- Deep-Q-Network

## 基礎名稱定義
- agent代理
- enritonment環境
- state狀態
- action動作
- reward獎勵
- value function價值函數


舉個例子
學生代表agent代理，升學和考試機制就代表著環境，考試題目代表著狀態，每次學生的考試作答(動作)，會得到一個成績(獎勵)，所以我們要根據得到的成績去做出最好的解題方式(RL目的)



## 定義
Q Learning 是一種無模型的強化學習算法，在基於價值的強化學習中，目標是優化一個價值函數，一個將最大未來獎勵映射到給定狀態的函數（可以被認為是一個簡單的查找表）。每個狀態的值是 RL 代理在實現目標之前可以期望獲得的獎勵總額。

## 算法思想

Q Learning是以value-based的算法，Q 即為 Q(s,a)，就是在某個s狀態下，採取了動作a，能夠獲得的收益期望，環境會根據 agent 的動作反饋回報 reward r，所以算法主要思想就是將 State 與 Action 建構一張 Q-table 來儲存Q值，然後根據Q值來選取能夠獲得收益的動作。

### 更新公式


$$
Q(s,a)\colon=Q(s,a) + \alpha(r + \gamma maxQ(s',a') - Q(s,a))
$$


- Q(s,a) 在s狀態執行a動作的價值
- α 學習率
- r 獎勵 (可能有正有負)
- γ 衰變值

(會發現 Q-table 很大，所以狀態不能把全部環境丟進去，只能擷取一小部分)

另外這和馬可夫決策有一點點不同

$$q_*(s,a) = R_s^a + \gamma \sum_{s'\in S}P^a_{s,s'}\space max \space q'_*(s',a')$$

馬可夫決策是全域，所以需要對於所有可能的 s' 進行期望計算，而 QL 則是基於 "單步經驗樣本更新"，是直接針對當前去估算。 

## 應用 (貪吃蛇)

### 貪吃蛇是什麼？

- 在遊戲中，玩家操控一條細長的直線（俗稱蛇）
- 玩家只能操控蛇的頭部朝向（上下左右）
- 路上會有很多"食物"
- 每次貪食蛇吃掉一件食物，它的身體便增長一些
- 不能碰觸到牆壁跟自身的身體
- 目的: 讓自己的身體越長越好

### 環境規劃

- 簡單化(處理資料比較方便，效率比較好)
- 12*12
- 蛇身藍色
- 蛇頭綠色
- 食物紅色
- 背景白色

![](https://hackmd.io/_uploads/rJ-KnVcNh.png)

## 流程

### 貪食蛇移動

- 定義前後左右為 0右 1左 2下 3上
- 頭移動到下一個位置，其餘位置替換成前面

### 貪食蛇吃到食物

- 身體不動，把頭換成食物的位置
- 讓食物隨機出生(不要碰到蛇)

### 死亡條件

- 頭的位置碰到自己的身體
- 頭的位置碰到牆壁

### Qtable

Qtable[(x,y),(l,r,u,d)] = [右, 左 ,下 ,上]

### 設定reward

- 撞到身體(-150)
- 撞到牆壁(-150)
- 拿到獎勵(200)
- 每走一步(-1)

### 更新 Qtable


$$
Q(s,a) \colon= Q(s,a) + \alpha(r + \gamma maxQ(s',a') - Q(s,a))
$$

只需要根據當前的state，查詢Qtable 例如: 當前的state為[(1,2),(0,0,0,1)] 也就是和食物距離 (1,2) 並且上面有障礙物，所以查詢Qtable，會得到選擇[右, 左 ,下 ,上]的值，只要選擇最大值即為當前最好的選擇。

最後依照公式更改 Qtable

結果

![image](https://hackmd.io/_uploads/SJxByRmQJl.png)

## Deep-Q-Network

Deep Q-Network (DQN) 是一種強化學習演算法，它結合了 Q-Learning 和 深度神經網路，用於解決狀態空間非常大或連續的問題。DQN 的誕生解決了表格型 Q-Learning 無法應對高維度問題的限制。

### 核心

在 Q-Learning 中，我們學習的是狀態-行動價值函數 Q(s, a)，但是，當狀態空間非常大時，表格形式無法處理。因此，DQN 使用 MLP 來逼近 Q(s, a)


### 流程

1. 初始化 Q Network: Q(s, a; $\theta$)
2. loss: $L(\theta) = E[(y-Q(s, a; \theta))^2]$
    其中 $y = r + \gamma max Q(s', a', \theta')$ 

### 實作

除了貪吃蛇的上下左右外，我還放了與食物 x 差距 y 貪吃蛇長度，跟上下左右共 7 個 state

Network : 7 -> 128 -> 128 -> 4

![image](https://hackmd.io/_uploads/SytF4CQXyg.png)




<h1 style = "color:gray">參考資料</h1>

[<強化學習>Q learning算法詳解](https://blog.csdn.net/qq_30615903/article/details/80739243)
