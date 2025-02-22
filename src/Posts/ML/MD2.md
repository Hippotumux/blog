---
title: 從0開始的機器學習_3
# icon: material-symbols:add-notes-outline
order: 3
date: 2023-08-03
category:
  - ML

tag:
  - note
  - theorem
---


影片參考[吴恩达机器学习系列课程](https://www.bilibili.com/video/BV164411b7dx/?p=2&spm_id_from=pageDriver&vd_source=1163f03eb192d949135cb83df54fce2c) 
課程講義[github](https://github.com/TheisTrue/MLofAndrew-Ng)


Python 是資料科學領域中非常熱門的程式語言。
有幾個常用的套件，例如：Numpy、Pandas、Matplotlib、Scipy 以及 scikit-learn，本篇將教大家 python 中 numpy 的使用方式，以及回顧上一篇的梯度下降和正規方程的應用。

<!-- more -->

## Numpy 

Numpy 是許多 Python 資料科學套件的基礎，讓使用者可以很容易建立向量（Vector）、矩陣（Matrix）等進行高效率的大量資料運算。

## 下載
 
```
pip install numpy 
```

## 引用 

```
import numpy as np
```

命名為 np 是因為 numpy 太長，如果要引用函式的話要打比較多字。

## 陣列 

我們會稱一維陣列為向量 vector ， 二維陣列為矩陣 matrix ， 而axis則是各軸的標示0(特定操作上會指定要哪個軸)

並請注意，在程式中，陣列是從 0 開始，一直到 N - 1
例如我們有個大小 N 為 4 的陣列 arr 
則 arr 裡面的值有 arr\[0], arr\[1], arr\[2], arr\[3]，並不會到arr\[4] 請記得。

## 建立一維陣列(vector) 

```python
import numpy as np

arr = np.array([0, 1, 2, 3])
```

除了指定元素外，也可以透過 np.zeros() , np.ones() 建立指定大小的陣列 (都存放 0 or 1)

```python
import numpy as np

arr0 = np.zeros(25)
arr1 = np.ones(15)
```

## 取值 

```python
import numpy as np 

arr0 = np.zeros(14) # 建立 大小為 14 全為 0 的vector
arr1 = np.ones(15) # 建立 大小為 15 全為 1 的vector

arr0[3] = 3 # 改變第四個位置的值
arr0[4] = 4 # 改變第五個位置的值

print(arr0) # 輸出 arr0 全部的值
print(arr0[2:5]) # 輸出 arr0 從 2 到 5（不含）的元素
print(arr0[::-1]) # 逆向輸出
```


## 加減法 

挺簡單的，它比照矩陣加減法，但請注意，和矩陣一樣，大小不一樣的話，會報錯的 !!

```python
import numpy as np
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
C = A + B
D = A - B
print(C)
print(D)
```



## 元素乘法(非矩陣乘法) 以及除法 

因為一維的沒辦法做到矩陣乘法，後面二維就會提到了。所謂元素乘法就是跟加減法一樣，對應的位置相加，所以規定也是需要矩陣大小一樣，那除法也是一樣的意思。

```python
import numpy as np
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
C = A * B
D = A / B
print(C)
print(D)
```


## 建立二維陣列 

```python
import numpy as np
A = np.array([[1,2,3],[4,5,6]])
B = np.array([[7,8,9],[0,1,2]])
```


$$A = \begin{bmatrix}
 1 & 2 & 3 \\
 4 & 5 & 6 \\
\end{bmatrix}$$

$$B = \begin{bmatrix}
 7 & 8 & 9 \\
 0 & 1 & 2 \\
\end{bmatrix}$$

當然也可以用 np.zeros(), np.ones() 快速建立 0,1 陣列

```python
import numpy as np
A = np.zeros([2,3]) # 2 X 3 矩陣
B = np.ones([2,3]) # 2 X 3 矩陣

# 補 多維的話 => np.zeros([2,2,2,2]) 2x2x2x2矩陣
```



## 建立單位矩陣 

當然也可以自己建立一個矩陣，然後寫個 for 賦予值即可

np.eye() 

```python
import numpy as np
A = np.eye(3) # 3X3單位矩陣
B = np.eye(3,5) # 3X5 從0的位置斜對角為1
C = np.eye(3,5,2) # 3X5 從2的位置斜對角為1
print(A)
print(B)
print(C)
```


## 轉置矩陣 

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.T)
```

## 點積 (dot product) 

注意 : 
在一維的時候，它會執行點積。
而二維的時候會執行矩陣乘法，需要第一個欄數等於第二個矩陣列數

```python
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
C = A.dot(B)
print(C)
```


## 內積 (inner product) 

跟 dot 很多操作類似。 
在一維時候操作是一樣的。
二維也是，不過要注意，內積的話 $A \cdot B$ = $AB^T$

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
C1 = np.dot(A,B)
C2 = np.inner(A,B.T)
print(C1)
print()
print(C2)
```


## dot VS inner 

多維的時候，  dot 會比 inner 靈活一點。
(運算方式不同，所以會導致不一樣的結果)

如果你看的很茫沒關係，解釋給你聽。

有兩個2X2矩陣 A B


在 dot 時候，跟矩陣乘法一樣
c11 = a11 \* b11 \+ a12 \* b21
c12 = a11 \* b12 \+ a12 \* b22
c21 = a21 \* b11 \+ a21 \* b21
c22 = a21 \* b12 \+ a21 \* b22


在 inner 的時候
c11 = a11 \* b11 \+ a12 \* b12
c12 = a11 \* b21 \+ a12 \* b22
c21 = a21 \* b11 \+ a22 \* b12
c22 = a21 \* b21 \+ a22 \* b22

發現哪裡不同了對吧，很清楚就知道他們乘的方式了。

## 外積 張量積 (outer product)

```python
import numpy as np

A = np.array([2,3,4])
B = np.array([1,2,3])
C = np.outer(A,B)
```

摁 ...，怎麼跟我想的外積不太一樣呢. 哦哦哦! 外積有兩個名稱一樣的，cross product 跟 outer product。

outer product 的運算方式為，把 A 陣列每個單值跟 B 陣列每個值相乘。

也就是說，矩陣大小為 len(A) X len(B)

例如 A = [1,2], B = [[1,2],[4,5]]
乘出來為 2 \* 4 的大小

也可以很清楚知道，乘的方式就是拿 A 的一個值 跟 B 所有值一一相乘得到 [1 , 2 , 4 , 5]，再拿 A 的第二個值乘上 B 的所有值一一相乘得到 [2 , 4 , 8 , 10]



## 外積 向量積 (cross product)

PS: 我一開始以為它的 outer 是向量積，想說怎麼怪怪的，原來是我英文不好QQ

對於兩個向量 A,B，他們的向量積寫作 A X B , 得出的為和AB互相垂直的向量，也就是 dot 完後 = 0
特別的是，方向依賴右手定則 (歐幾里得空間)

```python
import numpy as np

A = np.array([1,2,3])
B = np.array([-2,-5,7])
C = np.cross(A,B)
print(C)
```


可以看到和 A 內積為 0 和 B 內積也為 0
請注意，cross 中 dimension 最多為 3 ，不能超過。 
例如 A = [1,2,3,4] 這樣大小為 4 超過了

## 建立間距\等差一維陣列 

這有兩種方式，一種方式為 建立範圍內指定距離的陣列，另一種方式為 指定範圍，跟需要的資料量大小，自己生成等距陣列。

第一種 np\.arange(): 

```python
import numpy as np

a = np.arange(10) # 建立 建立 0～9，間隔為 1 的陣列
b = np.arange(1,10) # 建立 1～9，間隔為 1 的陣列
c = np.arange(1,5,0.5) # 建立 1～5，間隔為 0.5 的陣列
```

第二種 np\.linspace(): 

```python
import numpy as np

# 產生 1~10之間，100個資料(包含 1 10)
a = np.linspace(1,10,100) 
```

這兩個都是非常好用的函式，不用自己刻，也很清楚明白的告訴其他人，你的程式在做什麼。

## 更多常用功能 

```python
# 行列式值
detval = numpy.linalg.det(arr)

# 逆矩陣
arr_inv = np.linalg.inv(arr)

# 偽逆
arr_pinv = np.linalg.pinv(arr)
```

##  梯度下降實作  

numpy 常用介紹差不多就到這邊，如果有需要更多的使用方式可以另外去查，例如 fft, svd 等等

(另外這邊偷懶不正規化操作，只是簡單實作)

## 實作 

假設從 $2 - x + x^2 = 0$ 生成資料 ， $x$ 範圍在 [0~1]，生成20筆資料

```python
import numpy as np
data = 20
x = np.linspace(0,1,data)
y = 2 - x + x*x
```

此時我們就有了一筆資料，然後我們也偷懶知道它是二次曲線
可以假設 $\theta$ = [1,1,1] (代表 $\theta_0$ + $\theta_1x$ + $\theta_2x^2$ ) 這邊值隨便都可以。

```python
v = [1,1,1] # theta難打 我用 v 代替 請見諒
```

快樂偏微

$\theta_ j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})\space x^{(i)}_j$

$x^{(i)}_j$，其實很好理解，由$\theta_0$ + $\theta_1x$ + $\theta_2x^2$，可以知道 $\theta$ 對應一個資料 $x$

例如 $\theta_0$ 對應 $x^0 = 1$ 
$\theta_1$ 對應 $x$ 
$\theta_2$ 對應 $x^2$
所以 $\theta_j$ 對應 $x_j$ 上標就是第幾個資料
所以可以看到程式碼中 有 x\*\*j

```python
alpha = 0.1 # 大家可以修改這個值試試
v2 = [1,1,1] # 還記得是做完每一輪才跟改 v 的吧
for i in range(10000): # 一萬次迭代
    for j in range(3):
        jt = sum((v[0] + v[1]*x + v[2]*(x**2) - y) * (x**j) / data)
        v2[j] = v[j] - alpha * jt
    v = v2.copy() 
```

另外我們需要畫圖

```python
from matplotlib import pyplot as plt 

#把剛剛處理好的弄成第二個 y_after，因為要畫線，所以我們的點多一點
x_after = np.linspace(0,1,1000)
y_after = v[0] + v[1]*x_after + v[2]*(x_after**2) 

plt.title("gradient practice")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,"ob") # 畫原圖的資料(20個)
plt.plot(x_after,y_after,color='red') # 把處理好的方程式畫成紅線
plt.show()
print(v) # 最後輸出擬合的結果
```

可以發現非常的擬合原方程式，另外大家可以試試把 alpha 值改很小，或很大，會出現什麼事情!!

##  正規方程式實作  

一樣，從 $2 - x + x^2 = 0$ 生成資料 ， $x$ 範圍在 [0~1]，生成20筆資料


然後正規方程: $\theta = (X^TX)^{-1}X^Ty$ 
所以得先刻劃出 X = [1,-x,x^2] 然後對應20個資料

```python
data = 20
x = np.linspace(0,1,data)
X = np.zeros([data,3])
for i in range(data):
    X[i][0] = 1
    X[i][1] = x[i]
    X[i][2] = x[i] * x[i]
y = 2 - x + x*x
```

根據正規方程:

```python
v = np.linalg.inv(np.dot(X.T,X))
v = np.dot(v,X.T)
v = np.dot(v,y)
```

後續和gradient一樣

```python
#把剛剛處理好的弄成第二個 y_after，因為要畫線，所以我們的點多一點
x_after = np.linspace(0,1,1000)
y_after = v[0] + v[1]*x_after + v[2]*(x_after**2) 

plt.title("normal equation practice")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y,"ob") # 畫原圖的資料(20個)
plt.plot(x_after,y_after,color='red') # 把處理好的方程式畫成紅線
plt.show()
print(v) # 最後輸出擬合的結果
```


## 不可逆 

意外發現當 data = 15時，會不可逆，運氣蠻好的XDD

所以此時 inv 就用偽逆 pinv 即可。

## 

後記:

學會了 梯度下降 以及 正規方程 的實作~ 
希望能繼續看下去，看更多的方法來解決問題，因為現在的問題都還很簡單，後續也會拿一些例子來實戰~


下一篇 我們將來討論分類的問題 !
