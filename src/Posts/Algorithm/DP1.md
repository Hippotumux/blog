---
title: 動態規劃
# icon: material-symbols:add-notes-outline
order: 7
date: 2023-01-02
category:
  - Algorithm

tag:
  - note
  - lecture
---



## 簡介   
動態規劃(Dynamic Programming, DP)。
動態規劃是一種通過把原問題分解為簡單的子問題來求解最終問題的方法，然而動態規劃並不是某種具體的算法，而是一種解決特定問題的方法，所以種類也更為繁雜。
<!--more-->
## 基本思維與步驟

DP 與分治有個相同之處，都是將問題劃分成子問題，再由子問題的解合併成最後的解，其中子問題是指相同問題比較小的輸入資料。所以，設計 DP 的方法從找出遞迴式開始，設計 DP 的演算法通常包含下面幾個步驟：

1. 定義子問題。
2. 找出問題與子問題之間的(遞迴)關係。
3. 找出子問題的計算順序來避免以遞迴的方式進行計算。

這邊做一個簡單的例子:

費波那契數 定義為:
$F_{0}=0$
$F_{1}=1$
${\displaystyle F_{n}=F_{n-1}+F_{n-2}}（n≧2）$

很簡單明瞭的一個遞迴式子，用程式則是:

```cpp
int f(int n){
    if(n == 0) return 0;
    if(n == 1) return 1;
    return f(n-1) + f(n-2);
}
```

可以發現，其實會有很多步驟重複到了，一個遞迴呼叫兩個遞迴，當數量一多，很容易就爆掉了。時間複雜度為$O(2^n)$。所以需要找出子問題的計算順序來避免以遞迴的方式進行計算。那什麼樣的計算順序會讓我們可以不需要遞迴呢？

原則上「只要讓遞迴式等號右邊出現的在左邊的之前先算，並且用表格把算過的記下來，就可以不需要遞迴了。」而這個就稱為DP動態規劃。

以程式碼表示:
```cpp
int main(void){
    int n ;
    long long F[10000];
    cin >> n;
    F[0] = 0 , F[1] = 1;
    for(int i = 2 ; i <= n ; i++){
        F[i] = F[i-1] + F[i-2];
    }
    cout << "ans = " << F[n];
    return 0 ;
}
```

回憶一下剛才走過的思維步驟：定義子問題 -> 找遞迴關係 -> 最後找計算順序與定義表格來避免遞迴。所以 DP 從遞迴開始，但以去除遞迴來完成，看來似乎有點難，但以這題來說又很簡單。而其實真正難的是遞迴關係式的尋找。

## 狀態轉移
DP子問題的遞迴關係式，也有人稱「狀態轉移」。

也就是說：子問題可以看成一個狀態，目前的狀態是根據之前的那些狀態，這樣的轉換關係就稱作狀態轉移，設計 DP 時，最重要的就是找出狀態轉移。

所以可以把每個狀態想像每一個點，若 A 是 B 的前置狀態，則由 A 劃一個箭頭指到 B，會得到一個所謂的圖(graph)，這個圖就是這個 DP 的狀態轉換圖，在圖論上來說，它必然是個有方向但沒有環路的圖(directed acyclic graph, dag)，所謂環路是指從某點出發，沿著箭頭前進，最後又走到出發點。遞迴關係不會無限遞迴，所以必然沒有環路。

以費波那契數來看 : f(n) = f(n-1) + f(n-2)
可以看出每一個狀態是經由前兩個狀態轉移而來

![](https://hackmd.io/_uploads/rkeCQDl_h.png)

 
## 分類與複雜度

一個 DP 如果狀態有 $O(n^x)$ 而轉移式涉及 $O(n^y)$ 個狀態，一般可稱為 xDyD 的 DP

那費波那契數就是1D0D的，因為有 n 個要算，每次算 f(n)只涉及 2 個 f(n-1)與 (n-2) 簡單說就是從n個狀態轉成1個狀態

DP，如果沒有優化，很明顯的，時間複雜度就是 $O(n^{x+y})$ ，一般來說若 y=0 時比較簡單，因為遞迴式單純，而且遞迴式找出來時，程式就已經差不多寫完了。因為套個迴圈依序計算就是了。y=1 的 DP 題目通常存在某種優化或需要某些資料結構幫忙，所以會比較難。

## bottom up / top down 
最常用的方式是bottom up的作法(就像是上述例子)，基本上就是先找出遞迴式，然後找出計算順序來避免遞迴。這是很標準的DP。

另外有一種實現DP的方式是不找計算順序，直接依照遞迴式寫，不過為了避免重複計算，採取了一些步驟，這種實現方式稱為top down 

### 步驟:

1. 開一個表格紀錄是否算過，可以初始為不可能的值(或者另外存是否算過)，用來辨識是否曾算過
2. 在遞迴時，如果表格上曾計算過，則直接回傳答案
3. 如果沒有計算過，則呼叫遞迴，算完的時候紀錄在表格中。

以上述的費波那契數來改成top down的寫法:

```cpp
long long F[10000]={0}; // 因為不可能是0，所以只要是0代表沒跑過
F[0] = 1, F[2] = 2;

int Fab(int n){
    if(F[n]>0) return F[n];
    F[n] = Fab(n-1) + Fab(n-2);
    return F[n];
}
```
這個遞迴的程式跑的並不會很慢，基本上和bottom up一樣快，只差在判斷，不過差異並不大。

## 經典LCS例子(最長共同序列)
因為1D0D太過於直覺，所以我們來看2D0D的題目

### 題目
輸入兩字串，輸出最長共同序列(不用連續)

範例輸入: 
ABCDEFGHIJK
ACDAGHAJKLM

範例輸出(一個整數):
8  

....(ACDGHJK)

### 思路
如果我們直接暴力搜索的話，一個長度n的序列有$2^n$個子序列。因為每個位置可以刪除或不刪除，時間複雜度(worst case)=$O(n*2^n)$，超級可怕。

那我們從DP的想法來看，我們要設法定義子問題，比對兩個序列最簡單就是前往後比對，所以我們可以定義lcs[i][j] 為 s1的前i個字元，與s2前j個字元的LCS長度。可以分成兩種情況:

- 如果s1[i]!=s2[j]，兩者不相等，所以只是有一個對找LCS是沒有幫助的，所以我們可以刪掉s1最後一個字母s1[i]或s2最後一個字母s2[j]，因為要最常，所以取最大的。 也就是
$lcs[i][j] = max ( lcs[i-1][j] ,lcs[i][j-1] )$

- 如果s1[i]==s2[j]: 因為對中了，所以直接加上去就好
$lcs[i][j] = lcs[i-1][j-1]+1$

我們已經完成了遞迴的設計，那要注意需要定義邊界條件。根據我們的定義可以得到
lcs[i][0] = 0 , lcs[0][j] = 0;
整理出來即是:
$$ lcs[i][j]\begin{cases}
 0\hspace{14em} if\hspace{1em} i\hspace{1em} or\hspace{1em} j =0 \\ 
 lcs[i-1][j-1]+1 \hspace{6em} if\hspace{1em} s1[i]  =  s2[j] \\
 max(lcs[i-1][j],lcs[i][j-1]) \hspace{2em} otherwise\\
 \end{cases}$$

那如果把lcs[][]看成一個表格，就可以知道他跟左方，上方，左上方的格子有關，當然還要看s1[i]，s2[j]是否相等。時間複雜度就會降到$O(mn)$

### 實作
```cpp 
int main (void){
    string s1,s2;
    int lcs[N][N]={0};
    cin >> s1 >> s2;
    int n = s1.length() , m = s2.length();
    for (int i = 1 ; i <= m ; i++) {
        for (int j = 1 ; j <= n ; j++) {
            if (s2[i-1]==s1[j-1]) {
                lcs[i][j] = lcs[i-1][j-1]+1;
            }else{
                lcs[i][j] = max(lcs[i-1][j],lcs[i][j-1]);
            }
        }
    }
}
```

## 經典背包問題

背包問題就是經典的NP完全問題~ ， 問題通常都是，有很多重量的物品，有其價值，在有限的重量下，希望能裝到最多價值的物品。

## 01背包

### 問題
這是最基本最經典的背包問題，特點是，物品只能放一次，可放可不放。題目：有很多的寶物，重量為$w_i$，價值為$p_i$，請問當你有一個負重為K的背包，請問最多能裝多少價值的寶物(每個寶物只能裝一次)。
輸入格式為N，K，第二行為每件寶物的重量，第三行為每件寶物的價值
範例輸入：
5 10
2 2 6 5 4
6 3 5 4 6

範例輸出：
15


### 思路
可以根據所有重量對每個物品做DP，也就是若當前的 i 重量 可以裝下 j 物品，就裝進去算出剩下的重量在之前的最大價值為多少。
$DP[i] = max(DP[i],DP[i-w_j]+p[j])$

用圖表來看 (從左到右在換下一行看)

![](https://hackmd.io/_uploads/ry_14vgOh.png)

### 實作
```cpp
#include <bits/stdc++.h>
using namespace std;

vector<pair<int,int>> box; //定義first為重量，second為價值

int main(void){
    ios::sync_with_stdio(false),cin.tie(0);
    int n,k;
    cin >> n >> k;
    box.resize(n+1);
    for (int i = 1 ; i <= n ; i++) cin >> box[i].first;
    for (int i = 1 ; i <= n ; i++) cin >> box[i].second;
    
    int dp[1000][1000] = {0};
    for (int j = 1 ; j <= n ; j++){ //每個寶物
        for (int i = 0 ; i < box[j].first ; i++){
            dp[i][j] = dp[i][j-1];
        }
        for (int i = box[j].first ; i <= k ; i++){ //背包重量
            dp[i][j] = max(dp[i][j-1],dp[i-box[j].first][j-1]+box[j].second);    
        }
    }
    cout << dp[k][n] << '\n';
    
}
```
需要注意的有，因為物品一次只能買一件，DP需要開二維存取，不然不知道之前的到底裝過了目前的物品沒。

### 優化空間

剛剛說到，如果不開二維存取，會因為轉移式$DP[i] = max(DP[i],DP[i-w_j]+p[j])$ 而發生DP[i-w]不知道當前的寶物裝過了沒。但是當寶物還是重量空間需要1e5以上，二維空間就放不下了，那就需要優化空間了。

有兩種方式可以優化空間：
1. 滾動數組：在深入DP的時候會詳細講，會發現我們只需要Nx2的空間就好了，做完一次在將DP[][1]的結果放進DP[][2]，或是用0101交替。
2. 將主循環的 1 ~ k 逆向，變成從k~1就能解決了。

滾動數組之後會說，這邊放第二種的程式碼

```cpp
int dp[1000005] = {0};
for (int j = 1 ; j <= n ; j++){ //每個寶物
    for (int i = k ; i >= box[j].first ; i--){ //背包重量
        dp[i] = max(dp[i],dp[i-box[j].first]+box[j].second);    
    }
}
cout << dp[k] << '\n';
```

## 完全背包

### 問題
跟剛剛不同的是，物品有無限個，放幾次都可以。題目：有很多的寶物，重量為$w_i$，價值為$p_i$，請問當你有一個負重為K的背包，請問最多能裝多少價值的寶物(寶物無限多個)。
輸入格式為N，K，第二行為每件寶物的重量，第三行為每件寶物的價值
範例輸入：
5 10
2 3 6 5 4
2 4 5 4 6

範例輸出：
14

### 思路

類似於01背包問題，但已經不是對每件寶物「取或不取」，而是取0件，取1件，取2件....等很多種，那想法也跟01背包差不多，自己可以裝很多次，所以剛剛提到的問題就不存在了。
可以得到$dp[i][v] = max(dp[i-1][v - k * c[i]] + k * w[i]\space | \space 0 <= k * c[i]<= v)$

不過這樣太複雜了，上面我們讓重量從k~1是為了避免裝進自己，不過現在沒差了，所以可以簡單寫
$DP[i] = max(DP[i],DP[i-w_j]+p[j])，i從1到k$ 

### 實作
```cpp
#include <bits/stdc++.h>
using namespace std;

vector<pair<int,int>> box; //定義first為重量，second為價值

int main(void){
    ios::sync_with_stdio(false),cin.tie(0);
    int n,k;
    cin >> n >> k;
    box.resize(n+1);
    for (int i = 1 ; i <= n ; i++) cin >> box[i].first;
    for (int i = 1 ; i <= n ; i++) cin >> box[i].second;
    
    int dp[1000000] = {0};
    for (int j = 1 ; j <= n ; j++){ //每個寶物
        for (int i = box[j].first ; i <= k ; i++){ //背包重量
            dp[i] = max(dp[i],dp[i-box[j].first]+box[j].second);     
        }
    }
    cout << dp[k] << '\n';
    
}
```

## 多重背包

### 問題
其實也差不多，只差在寶物不是無限裝，而是有限的次數。題目：有很多的寶物，重量為$w_i$，價值為$p_i$，數量為$m_i$，請問當你有一個負重為K的背包，請問最多能裝多少價值的寶物。
輸入格式為N，K，第二行為每件寶物的重量，第三行為每件寶物的價值，第四行為每件物品的數量
範例輸入：
5 10
2 3 6 5 4
3 4 5 4 5
3 1 3 4 2

範例輸出：
14

### 思路
因為每件物品的數量有限，所以需要再多一層迴圈來判斷裝幾個進去。所以DP轉移式就變成
$dp[i][v] = max(dp[i - 1][v - k * c[i]] + w[i]\space | \space 0 <=k <= n[i])$

### 實作

```cpp
#include <bits/stdc++.h>
using namespace std;

struct info {
    int w,p,m;
};

vector<info> box; //定義first為重量，second為價值

int main(void){
    ios::sync_with_stdio(false),cin.tie(0);
    int n,k;
    cin >> n >> k;
    box.resize(n+1);
    for (int i = 1 ; i <= n ; i++) cin >> box[i].w;
    for (int i = 1 ; i <= n ; i++) cin >> box[i].p;
    for (int i = 1 ; i <= n ; i++) cin >> box[i].m;
    
    int dp[1000][1000] = {0};
    for (int j = 1 ; j <= n ; j++){ //每個寶物 
        for (int i = box[j].w ; i <= k ; i++){ //背包重量
            for (int l = 1 ; l <= box[j].m && l*box[j].w <= i ; l++){ //數量且數量*重量可以裝進去
                dp[i][j] = max({dp[i][j],dp[i][j-1],dp[i-l*box[j].w][j-1]+l*box[j].p});
            }       
        }
    }
    cout << dp[k][n] << '\n';
    
}
```


這邊分享一下滾動數組的作法，而且是有效率的。透過01交換

```cpp
int dp[1000000][2] = {0};
for (int j = 1 ; j <= n ; j++){ //每個寶物 
    for (int i = box[j].w ; i <= k ; i++){ //背包重量
        for (int l = 1 ; l <= box[j].m && l*box[j].w <= i ; l++){ //數量且數量*重量可以裝進去
            dp[i][j&1] = max({dp[i][j],dp[i][j!=(j&1)],dp[i-l*box[j].w][j!=(j&1)]+l*box[j].p});
        }       
    }
}
cout << dp[k][n&1] << '\n';
```
