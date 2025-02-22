---
title: 從0開始的機器學習_2
# icon: material-symbols:add-notes-outline
order: 2
date: 2023-08-02
category:
  - ML

tag:
  - note
  - theorem
---


影片參考[吴恩达机器学习系列课程](https://www.bilibili.com/video/BV164411b7dx/?p=2&spm_id_from=pageDriver&vd_source=1163f03eb192d949135cb83df54fce2c) 
課程講義[github](https://github.com/TheisTrue/MLofAndrew-Ng)
  
本篇以梯度學習以及正規方程為軸，擬合資料。

<!-- more -->

## 模型描述 

課程中會用到一些符號代表某些事情
 
![](https://i.imgur.com/1wweOlQ.png)

(x,y) -> one training example 
($x^{(i)}$, $y^{(i)}$)  the ith training example

###  給訓練集下定義


![](https://i.imgur.com/IG11XAk.png)

- 提供訓練集
- 學習算法目的為輸出一個函數 -> h 表示 (h input x and output y)
    - 簡單說，h 是從 x 到 y 的映射(mapping)
    - $\theta$稱為模型參數(Parameters)

並且會從簡單的線性擬和到最終處理更加複雜的模型，以及更複雜的學習算法
而這個例子中，會稱他為"線性回歸(linear regression)"

## 損失函數(loss function) 


或叫代價函數(cost function)。能讓我們把最有可能的直線與我們的數據相擬合

![](https://i.imgur.com/ugB95EO.png)

假設我們有個資料，有個預測函數h(x)，然後m = 47個資料。那我們要如何選擇參數$\theta$

根據不同的假設，我們會得到不一樣的假設函數。我們的目的就是希望線能最接近我們的數據

![](https://i.imgur.com/dMKu9dt.png)

所以會希望取 min($(h(x)-y)^2$)  也就是方均根誤差，展開後

=>$\sum_{i=1}^m(h(x^{(i)}) - y^{(i)})^2$

=>$\frac{1}{2m}$$\sum_{i=1}^m(h(x^{(i)}) - y^{(i)})^2$

盡量減少平均誤差，所以在除以m，為了導數後抵銷剛剛的平方，所以在除以二，但其實對結果來說無影響因為是對所有資料一起除。因此，簡單的說，我們正把這個問題變成找到能使我們訓練集中預測的值和真實值的差平方和的1/2m最小的 $\theta_0$ , $\theta_1$，這就是我們線性回歸的目標函數。

為了使其更為明確，我們改寫這個函數，定義一個代價函數J

$J( \theta_0 , \theta_1)$ = $\frac{1}{2m}$$\sum_{i=1}^m(h(x^{(i)}) - y^{(i)})^2$ ， and minimize $J(\theta_0 , \theta_1)$，同時我們也稱他為損失函數 

PS：分母的2是為了之後微分可以跟平方抵銷

## 梯度下降法(gradient descent) 

PS: 微積分下會教

![](https://hackmd.io/_uploads/ryGwkcJwn.png)

把上面這張圖用等高線的方式畫成右下角的圖
同時想要求 minimize $J(\theta_0 , \theta_1)$，使得 $h(\theta)$ 擬合數據

![](https://hackmd.io/_uploads/rJ9oJ91wh.png)


###  梯度下降運作方式

![](https://hackmd.io/_uploads/rJa7-9kP3.png)

從這張圖來看，我們想要最小化，也就是想要找到低窪點，梯度下降法就是讓你找到斜率最大的那個方向爬下山，(不要掉進河裡哦X)

所以更改的方式為:

$\theta_ j := \theta_j - \alpha\frac{\partial}{\partial \theta_j}J(\theta_0,\theta_1) \space \space for \space j = 0, 1$ 

$\alpha$ 是學習率，如果學習率太高，可能會導致不收斂，

![](https://hackmd.io/_uploads/S1odo_-vn.png)

因為 $\alpha$ 高，所以那怕方向是往低點，但太超過就會跑到另外一邊更高的點，導致最後無法收斂。

如果 $\alpha$ 太小的話，又會導致收斂速度過慢，所以 $\alpha$ 取的很重要。

在程式實作的時候要注意，這很像兩數交換的概念，不能直接 a = b, b = a 

也就是說我們應該這樣寫
$temp0 := \theta_0 - \alpha \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)$
$temp1 := \theta_0 - \alpha \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)$
$\theta_0 := temp0$
$\theta_1 := temp1$

###  n維度 

$\theta_ j := \theta_j - \alpha\frac{\partial}{\partial \theta_j}J(\theta_0, ... ,\theta_n)$

然後帶回 $J( \theta_0 , \theta_1)$ = $\frac{1}{2m}$$\sum_{i=1}^m(h(x^{(i)}) - y^{(i)})^2$

所以偏微分完之後
$\theta_ j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})\space x^{(i)}_j$

(稍微推一下就出來了沒有很難~)

###  正規化 

也就是所謂特徵縮放，想像一下，如果是想預測房價，資料會有 房子大小(平方公尺)，跟價錢。

但很明顯這兩個量級差太多了，在梯度下降就會效率很差，因為每次修改但是下降的幅度都不高(下圖還不明顯，很容易長的比這個更尖，很密稠)

![](https://hackmd.io/_uploads/rkVSYCeDh.png)


所以希望能夠把數據給縮小化，希望它圓一點，一個好處就是能夠讓我們在做梯度下降時更快收斂，另一方面也可以避免產生偏誤。

那方法也很多，像是高中會學的標準化，或者歸一化，不過主要還是得看原資料的型態。
 
## 正規方程(normal equation) 

PS: 這個方法我在上線性代數課的時候，老師有教~ 好課值得一修在修!!

實際上，我們在求 $min \space J(\theta)$ 時，從微積分中可以知道，求的會是偏微為0的點，這個偏微分可能最終會很複雜，
所以介紹第二種方式，正規方程。

來用之前那個房價預測的例子，可以看到總共有四個變數，加上一個 價錢 y

![](https://hackmd.io/_uploads/BJlnNkWPh.png)

那我們加上一行 $X_0$ 並且把它設為1 (也就是常數的部分 1\*b = b)

![](https://hackmd.io/_uploads/SJbXHy-D2.png)

那麼矩陣就可以表達為

![](https://hackmd.io/_uploads/BkVurkbPh.png)

(以下線性代數會教)
則$X^TX\theta = X^Ty$  
所以$\theta = (X^TX)^{-1}X^Ty$ 

複數的話 
則$X^*X\theta = X^*y$  
所以$\theta = (X^*X)^{-1}X^*y$ (\*為conjugate共扼)

(在實係數 \* 在矩陣就是轉置就好，在複數才需要轉置完再取共扼)

### 推導(如果你有興趣的話)
:::info
(以下 讓 $\theta^T$ 以 v 表示，也就是下面都是用transpose去看 例如 $X^T\theta$ = vX)

$lemma$
$(a)$ $rank(X) = rank(XX^*)$
$(b)$ $R(X^*) = R(XX^*)$

##

$proof$ $(a)$
$To \space show \space N(XX^*) = N(X)$
($\supseteq$) $clear$

($\subseteq$) $Let \space v$ $\in$ $N(XX^*)$
$<vX,vX>\space = \space<v,vXX^*> \space = \space <v,0> = 0$ $\because v \in$ $N(XX^*)$
$\therefore vX = 0 \space , v \in N(XX^*)$

## 

$proof$ $(b)$
$R(XX^*) \subseteq R(X^*)$
$by (a) \space one\space  has \space rank(XX^*) = rank(X) = rank(X^*)$ $(by \space dimension \space theorem)$
$\therefore R(X^*) = R(XX^*)$

$so \space \forall \space y \in F^n$
$vXX^* = yX^*$
$v = yX^*(XX^*)^{-1}$ 

## 
因為X要transpose回來(看最上面假設)，所以

$v^T = \theta = (yX^*(XX^*)^{-1})^T = (X^*X)^{-1}X^*y$ 

如果 $y$ 是實數

$\theta = (X^TX)^{-1}X^Ty$ 

得證

:::

### 

而 $\theta = (X^*X)^{-1}X^*y$ 這個方程稱為正規方程


## 優缺點 

![](https://hackmd.io/_uploads/Skq5ggbw3.png)

而在梯度下降中的學習率 $\alpha$ 老師他說會用3倍方法，
像是一開始設 0.03 -> 0.1 -> 0.3 -> 1 這樣

然而現在矩陣的那些運算也不需要硬刻了。
python 有 numpy 
(另外我目前沒打算用C去寫，有點麻煩，我想先學好再說)

另外 正規方程也有可能會遇到不可逆的情況。


###  正規方程不可逆時 

$\theta = (X^TX)^{-1}X^Ty$ ， 當 $(X^TX)^{-1}$ 不可逆時，怎麼辦。

可以用偽逆的方式，


不過，不可逆的情況少之又少，不應該專注在不可逆的情況！

## 後記: 

我後來在做梯度下降法練習，用三次方程產生資料，去擬合原函數，但很容易python會浮點數overflow，alpha到0.03才可以，重點是我還正規化過了。同時我也測試了 $\theta$ 一輪才改一次還是當下馬上改，如同老師說的一樣，也是會成功的，只是那算另一種方法。 

後來問學長，他說是用純線性方程式擬合不太好，像是3次方程式，在做梯度下降的同時，稍微修改一下$\theta$就會很容易爆掉。

下一篇筆記，我打算先做python numpy 的介紹，讓大家可以實做練習一下，順便把我在python練習的程式碼放上來給大家參考。
