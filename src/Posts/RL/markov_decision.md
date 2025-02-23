---
title: 馬可夫決策
# icon: material-symbols:add-notes-outline
order: 1
date: 2023-08-14
category:
  - ML

tag:
  - probability
  - theorem

---

## 馬可夫決策 Markov decision process

馬可夫決策(MDP)是對完全可觀測的環境進行描述的，也就是觀測到的狀態內容完整地決定了決策需要的特徵，幾乎所有強化學習問題都可以轉化為MDP。



### 馬可夫性 Markov Property
某一狀態信息包含了所有相關的歷史，只要當前狀態可知，所有的歷史信息都不再需要，當前狀態就可以決定未來，則認為該狀態具有馬可夫性。

可以用下面的狀態轉移概率公式來描述馬可夫性： (狀態從s -> s'的機率)
$$P_{ss^{'}} = P \space (S_{t+1} = s^{'} | S_t = s)$$ 

<!-- more -->

下面狀態轉移矩陣定義了所有狀態的轉移概率： (每一列相加為1)
$$P=\begin{bmatrix}
 P_{11} & P_{12} & ... & P_{1n}\\
 ... & ... & ... &...\\
 P_{n1} & P_{n2} & ... & P_{nn} \\
\end{bmatrix}$$


### 馬可夫鏈 Markov Chain

它是一個無記憶的隨機過程，可以用<S,P>表示，其中S是有限數量的狀態集，P是狀態轉移概率矩陣

![](https://i.imgur.com/lP9uaiC.png)

圈圈表示學生所處的狀態，Sleep是一個終止狀態，或者說自己循環，(Sleep下一個狀態100%還是Sleep)，箭頭表示狀態之間的轉移，上面的數字就是當前轉移的機率。

:::info
舉例來說： 當學生在Class1時候，他有50%機率上Class2，同時有50%機率不聽課，跑去滑FB，在瀏覽FB時，又有90%在下一個狀態還在滑FB，不過有10%機率回到課堂上。以此類推，這些可能性都稱為Sample Episodes(樣本回合或說一輪遊戲)

以下四種Episodes都是有可能的
- C1 - C2 - C3 - Pass - Sleep
- C1 - FB - FB - C1 - C2 - Sleep
- C1 - C2 - C3 - Pub - C2 - C3 - Pass - Sleep
- C1 - FB  - C1 - C2 - C3 - Pub - C1 - FB - FB - C1 - C2 - C3 - Pub - C2 - Sleep


所以就可以把該學生的馬可夫的轉移矩陣寫出來
左到右(上到下) 為 C1、C2、C3、Pass、Pub、FB、Sleep


$$P =  \begin{bmatrix}
  .& 0.5 & . & . & . & 0.5 & . \\
  .& . & 0.8 & . & . & . & 0.2 \\
  .& . & . & 0.6 & 0.4 & . & . \\
  0.2& 0.4 & 0.4 & . & . & . & . \\
  0.2 & . & . &. & 0.8 &.&.\\
  0.1& . & . & . & . & 0.9 & . \\
  .& . & . & . & . & . & 1 \\
\end{bmatrix}$$
:::

### 補充

- 週期性馬可夫鏈:是指馬可夫鏈中每個狀態都具有固定的周期性行為，也就是說，從某一狀態出發，回到該狀態的時間間隔固定。 
- 馬可夫如果有無法離開的狀態，稱為吸收態。

特徵值關係: 

- 馬可夫鏈最大特徵值 = 1
- 週期性馬可夫鏈最大特徵值 = 1 (只有一個)，其餘小於 1
- 馬可夫鏈若存在多個獨立子空間，或存在吸收態，可以有不只一個特徵值 = 1 

譜半徑 = 1

###馬可夫獎勵過程Markov Reward Process

馬可夫獎勵過程，就是在馬可夫過程<S,P>基礎上，增加了獎勵R(獎勵函數)，和衰減係數$\gamma$，也就是<S,P,R,$\gamma$>
S狀態下的獎勵是某一時刻(t)處在狀態s下，在下一個時刻(t+1)能獲的的獎勵期望:

$$R_s = E(R_{t+1}|S_t = s)$$

你可能會好奇，為什麼獎勵是t+1時刻的，相當於離開這個狀態才能獲得獎勵，而不是進入該狀態獲得獎勵。但這其實只是約定好的，為了再描述RL問題比較方便，如果把獎勵改成$R_t$的話，本質上還是相同的，所以在表述上可以把獎勵描述為"當進入某個狀態會獲得相應的獎勵"。

***衰減係數 Discount Factor $\gamma \in [0,1]$***，他的引入目的為數學表達方便、避免無線循環、遠期利益有不確定性、符合人類對於眼前利益的追求、符合金融上獲得的利益產生新的利益更有價值等等。

下圖為獎勵過程的例子，加上了針對每個狀態的獎勵，但不涉及衰減係數的計算。
![](https://i.imgur.com/LgOMHwu.png)

###收穫/回報/收益 return


$G_t$為在一個馬可夫獎勵鏈上從t時刻開始往後所有獎勵的衰減總和，公式如下:
$$G_t = R_{t+1} + \gamma R_{t+2} + \space ... \space = \sum_{k=0}^\infty{\gamma ^k R_{t+k+1}}$$

其中，衰減係數體現了未來獎勵在當前的價值比例，$\gamma$ 接近0 代表注重 "較近的利益"，$\gamma$ 接近1 代表注重"遠期的利益"


###價值函數 Value function


價值函數給出了某一狀態或行為的長期價值，一個馬可夫獎勵過程的某一狀態的價值函數為"從該狀態開始"的馬可夫鏈收穫的期望:
$$v(s) = E(G_t|S_t = s)$$

把學生的馬可夫獎勵過程用下列圖表示。第三行以下為機率，第二行為reward值
![](https://i.imgur.com/jLaumqY.png)

那我們就可以計算當衰變$\gamma$ = $\frac{1}{2}$，在t=1時刻($S_1 = C_1$)時狀態$S_1$的reward
![](https://i.imgur.com/eceBEMq.png)

可以理解到，reward是針對"某一個狀態"，RL很多問題可以歸結為求狀態的價值問題。


## 馬可夫決策價值函數推導


### 公式 Bellman equation

:::info
嘗試用價值公式來推導看看能得到什麼
$v(s)$ 
$= E[G_t|S_t = s]$ 
$= E[R_{t+1} + \gamma R_{t+2} + \gamma ^ 2 R_{t+3} + \space ...|S_t = s]$ (根據$G_t公式$) 
$= E[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \space ...) | S_t = s]$ 
$= E[R_{t+1} + \gamma G_{t+1} | S_t = s]$
其中 $E[V(S_{t+1})|S_t] = E[E[G_{t+1}|S_{t+1}]|S_t] = E[G_{t+1}|S_t]$
$= E[R_{t+1} + \gamma v(S_{t+1}) | S_t = s]$
:::

在導出最後一行時，將$G_{t+1}$ 變成了 $v(S_{t+1})$
所以下面就是bellman的方程:
$\space E[R_{t+1} + \gamma v(S_{t+1}) | S_t = s]$


通過方程可以看出v(s)由 該狀態的即時期望和下一個狀態的價值期望。可以根據下一個狀態的機率分布得到其期望，如果用 s' 表示s狀態下一個任一可能的狀態，那麼可以改寫成
$R_{t+1} + \gamma \sum_{s' \in S} P_{ss'} v(s')$


### 解釋

![](https://i.imgur.com/7zrQjhE.png)

圖上有 $\gamma$ = 1 的各種狀態價值，狀態$C_3$的價值可以通過Pub和Pass價值以及他們之間的狀態轉移機率來計算:
4.3 = -2 + 1.0 * (0.6 * 10 + 0.4 * 0.8)

### Bellman equation的矩陣形式和求解

=> $v = R + \gamma Pv$

用矩陣來表達(好理解):

$$\begin{bmatrix}
  v(1)\\
  ...\\
  v(n)\\
\end{bmatrix} = \begin{bmatrix}
  R(1)\\
  ...\\
  R(n)\\
\end{bmatrix} + \gamma \begin{bmatrix} P_{11} & P_{12} & ... & P_{1n}\\
 ... & ... & ... &...\\
 P_{n1} & P_{n2} & ... & P_{nn} \\
\end{bmatrix}
 \begin{bmatrix}
  v(1)\\
  ...\\
  v(n)\\
\end{bmatrix}$$

bellman方程式一個線性方程式，因此理論上解可以直接求解:
$v = R + \gamma Pv$
$(I - \gamma P) v = R$
$v = (I - \gamma P)^{-1} R$

通過求解這個矩陣方程，我們能直接得到每個狀態在策略下的價值。

如果直接運算的話，複雜度很高，所以通常會用迭代法，例如
- 動態規劃DP，((想知道DP演算法概念嗎？--- [傳送門](https://hackmd.io/@HIPP0/B1cZdq0rs)
- 蒙地卡羅評估(Monte-Carlo evaluation)
- 時序差分學習(Temporal Diffrerence)




###馬可夫決策過程 Markov Decision Process

相較於馬可夫獎勵過程，馬可夫決定過程多了一個行為集合A，所以會是<S,A,P,R,$\gamma$>。看起來很類似獎勵過程，不過這裡的P和R都和具體的"行為A"對應，不像獎勵過程只對應"某個狀態"，A表示的是有限行為的集合，具體的表示如下:

$$P_{ss'}^a = P[S_{t+1} = s'\space | \space S_t = s, A_t = a]$$

$$P_s^a = E[R_{t+1} \space | \space S_t = s, A_t = a ]$$

:::info
舉個例子，圖上紅色的文字是採取的行動，而不是之前的狀態名，對比之前學生可以發現，及時獎勵與行為對應了，同一個狀態下採取不同的行為得到的獎勵是不一樣的，另外選擇"去查閱文獻"這個動作時，主動進入了臨時狀態(黑色小點)，隨後被動的按照環境分配到另外三個狀態，也就是說此時Agent沒有權利選擇去哪一個狀態。

![](https://i.imgur.com/Xmyunt5.png)

:::

###策略Policy

策略$\pi$是機率集合或分布，其元素$\pi (a|s)$為對過程中的某一狀態s採取其可能的行為a的機率：
$$\pi (a|s) = P[A_t = a \space | \space S_t = s]$$

一個策略完整了定義了整個個體的行為方式，也就是說定義了個體在各個狀態下可能的行為，以及其機率大小。Policy 僅和當前狀態有關，與之前無關。同時某一確定的Policy是靜態的，與時間無關，但是個體可以隨著時間更新策略。

當給定一個MDP : $M = <S,A,P,R,\gamma>$ 和一個策略 $\pi$，那麼狀態序列 $S_1,S_2,...$ 是一個馬可夫過程$<S,P^{\pi}>$。同樣，狀態和獎勵序列 $S_1,R_1,S_2,R_2,...$是一個馬可夫獎勵過程$<S,P^{\pi},R^{\pi},\gamma>$\
並且滿足下面兩個方程:

$$P_{s,s'}^{\pi} = \sum _{a \in A}\pi (a|s)P^a_{s,s'}$$

也就是在執行策略 $\pi$ 時，從狀態 s 轉移到狀態 s' 的機率等於一系列機率的和，這一系列機率指的是在當前策略時，執行某一行為機率與該行為能使狀態 s 轉移到 s' 的機率乘積。

以及滿足獎勵函數如下:

$$R_s^{\pi} = \sum _{a\in A} \pi(a|s)R_s^a$$

也就是在當前s狀態下，執行某一指定策略得到的即時獎勵是該策略下所有可能行為得到的獎勵與行為發生的機率乘積的和。

策略在MDP中相當於一個agent可以在某一個狀態時做出選擇，進而有形成各種馬可夫過程的可能，而且基於策略產生的每一個馬可夫過程是一個馬可夫獎勵過程，各過程之前的差別不同選擇產生的後續狀態以及對應不同的獎勵。

###基於π的價值函數

定義 $v_\pi(s)$是在基於策略$\pi$的"狀態價值函數"，表示從狀態s開始，遵循當前策略獲得的收穫期望值，表示如下:

$$v_\pi(s) = E_\pi (G_t \space | \space S_t = s)$$

注意: 策略是靜態的，不隨狀態改變而改變。

定義$q_\pi(s,a)$為"行為價值函數"，表示在執行策略$\pi$時，當前狀態s執行某一具體行為a所能得到的收穫期望值，行為價值函數一般都是和某一特定狀態相對應，表示如下:

$$q_\pi(s,a) = E_\pi (G_t \space | \space S_t = s,A_t = a)$$   

![](https://i.imgur.com/0YENMzF.png)

###Bellman期望方程

可以改用下一個時刻狀態價值函數或行為價值函數來表達，如下:
$$v_{\pi}(s) = E_{\pi}[R_{t+1}+\gamma v_{\pi}(S_{t+1})\space | \space S_t = s]$$
                                      
$$q_{\pi}(s,a) = E_{\pi}[R_{t+1} + \gamma q_{\pi}(S_{t+1},A_{t+1}) \space | \space S_t = s, A_t = a]$$
                               
##
                        
### <$v_\pi(s)和q_\pi(s,a)的關係$>
![](https://i.imgur.com/cBJkVkc.png)
空心的部分表示狀態，黑色表示動作本身，可以看出遵循$\pi$策略時，狀態s的價值為在該狀態下遵循某一策略採取所有可能的價值按行為發生機率的乘積和
                          
$$v_\pi(s) = \sum_{a\in A}\pi(a|s)q_\pi(s,a)$$
                              
也可以表示行為價值函數
![](https://i.imgur.com/IwYSjsf.png)
$$q_{\pi}(s,a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\pi(s')$$
                       
                      
組合起來可以得到
![](https://i.imgur.com/RQDHrsB.png)
$$v_\pi(s) = \sum_{a\in A}\pi(a|s)(R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\pi(s'))$$
                                                              
也可以得到
![](https://i.imgur.com/qFdsPmD.png)
$$q_{\pi}(s,a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a \sum_{a\in A}\pi(a'|s)q_\pi(s',a')$$
                                                           
:::info
下圖解釋了紅色圈圈的狀態價值計算過程，把所有可能的行為有相同的機率被選擇進行(0.5去Pub,0.5去Study)
7.4 = 0.5\*(1+0.2\*-1.3+0.4\*2.7+0.4\*7.4) + 0.5\*10
![](https://i.imgur.com/2ktNDXv.png)
:::                         
###Bellman期望方程矩陣形式                                                               
$v_\pi = R^\pi + \gamma P^\pi v_\pi$
$(I - \gamma P^\pi) v_\pi = R^\pi$
$v_\pi = (I- \gamma P^\pi)^{-1} R^\pi$
                                                           
## 最優函數
###最優價值函數
最優價值函數$v_*(s)$即為所有策略產生的狀態價值函數中，選取使狀態s價值最大的函數                                                                          
                                
$$v_* = max_\pi (v_\pi(s))$$

最優行為函數:
                      
$$q_*(s,a) = max_\pi (q_\pi(s,a))$$                     
                          

###最優策略
對於任何狀態s，遵循策略$\pi$的價值不小於遵循策略$\pi^{'}$，則策略$\pi$優於策略$\pi^{'}$
                                  
:::info             
### 定理                
對於任何MDP以下幾點成立
- 存在一個最優策略，比其他策略更好或至少相等
- 所有的最優策略有相同的最優價值函數
- 所有的最優策略具有相同的行為價值函數
:::                          

###尋找最優策略

可以通過最大化價值函數來找最優策略:

$$ \pi_*(a|s)\begin{cases}
 1 \space \space if \space a = argmax \space q_*(s,a)\space| \space a\in A\\
 0 \space \space otherwise\\
 \end{cases}$$                 
:::info                   
醉於任何MDP存在一個最優策略，同時如果我們知道最優行為價值函數，則表示找到了最優策略，以下圖中紅色箭頭為最優策略
![](https://i.imgur.com/1cyyQoa.png)
:::                             

###Bellman 最優方程                                             
                                
針對$v_*$ 一個狀態的最優價值等於從該狀態出發採取的所有行為所產生的行為價值中最大的那個行為價值，也就是:
                                    
$$v_*(s) = max \space q_*(s,a)$$

針對$q_*$ 在某個狀態s下，採取某個行為的最優價值由兩部分組成，一為離開狀態s的即時獎勵，另外一部分則是所有能達到狀態s'的最優狀態價值按照出現機率求和，也就是:

$$q_*(s,a) = R_s^a + \gamma \sum_{s'\in S}P^a_{s,s'}\space v_*(s')$$
                   
組合起來針對$v_*$，有:

$$v_*(s) = max \space R_s^a + \gamma \sum_{s'\in S}P^a_{s,s'}v_*(s')$$

針對$q_*$，有:

$$q_*(s,a) = R_s^a + \gamma \sum_{s'\in S}P^a_{s,s'}\space max \space q'_*(s',a')$$

:::info
bellman最優方程示例
![](https://i.imgur.com/y3ELD9r.png)
:::

###求解bellman最優方程

Bellman最優方程是非線性的，沒有固定的解決方案，通過一些迭代方法來解決例如:
- Q learning-----[傳送門](https://hackmd.io/@HIPP0/S1sKnyQm3)
- 價值迭代
- 策略迭代
- Sarsa
等等


<h1 style = "color:gray">參考資料</h1>

[<強化學習>馬可夫決策](https://zhuanlan.zhihu.com/p/28084942)
https://librariestoner.com/archives/rlnote1

