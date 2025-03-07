---
title: 貪婪法
# icon: material-symbols:add-notes-outline
order: 5
date: 2023-01-02
category:
  - Algorithm

tag:
  - note
  - lecture
---

## 簡介

貪婪演算法（英語：greedy algorithm），又稱貪心演算法，是一種在每一步選擇中都採取在當前狀態下最好或最佳（即最有利）的選擇，類似於模擬一個「貪心」的人，這個人目光短淺，只看眼前利益，不考慮可能後果，所以可想而知，並不是所有時候都可以得到最優解，使用貪心演算法時，請要確保自己可以「證明」其正確性
<!--more-->
## 核心原理

在每一個階段選擇當前最佳解，並以此達到整個問題的最佳解。


### 最優子結構

問題能夠分解成子問題解決，子問題的最優解可以推到最終問題的最優解，而貪婪演算法在最優子結構的問題很有效果。

### 證明

一般有兩種證明方式，反證法或歸納法。

- 反證法：如果任意交換兩個元素後，答案不會變的更好，可以確定目前解是最好的。
- 歸納法：先算出邊界情況的最優解(例如 n = 1)，在證明對於每個 n，$F_{n+1}$ 都可以由 $F_n$ 推出。

### 題型

常見的有兩種

- 將XXX按照某種方式排序，然後按照某種順序選擇
- 每次從XXX取最大/小的東西，更新XXX

兩者區別在於，第一個是離線的，先處理完再選擇，另一種是在線，邊處理邊選擇。

### 後悔解法

把當前選擇都當成最優解來看，然後進行比較，如果之後選擇不是最優就反悔，捨棄，如果一樣是最優，就接受，以此類推。


## 換硬幣

### 問題
假設能換代幣的幣值有 [1,10,100,1000] ， 今天河馬拿了x元去換，則最少能拿到多少個代幣。

### 思路
因為代幣的數量希望要最少,直覺貪心的想法就是要盡量使用面額大的硬幣,代幣由大
面額的往小的考慮,能換盡量換,剩下不足的再考慮比較小的面額。

### 實作

```cpp
int coin[4] = {1,10,100,1000};
int x, total = 0;
cin >> x;
for (int i = 3 ; i > 0 ; i--) {
    total += x / coin[i];
    x %= coin[i];
}  
cout << total << '\n';
```

### 這種寫法是正確的嗎？      

程式寫起來很簡單，簡單到如果考試碰到這種題目，會懷疑是不是有陷阱。這個方法很直覺，但怎麼知道它是對的呢？ 你可能會說:就很直覺啊，路邊隨便找個小學生也會這樣做。    

那麼請看看下面的狀況 : 假設代幣的面額是 [1, 4, 5] 三種，上面的貪心方法會找到正確答案嗎？ 假設 x = 8，貪心的方法會先換一個 5 元，然後剩下 3 元，在換 3 個 1 元，所以總共換4個硬幣，不過若是全部換成 4 元的只需要 2 個硬幣。   

所以並不是應用在任何場合，必須證明，那如何證明呢？
這邊使用歸納法的方法。

在代幣是 [1, 10, 100, 1000] 的情形下，如果 1 元的代幣總額不小於 10 ， 一定可以拿出 10 個換成一個 10 元，所以 1 元代幣的個數不會超過 9 個，同樣的你可以證明 10 元， 100元不會超過9個。也就是說，對任意第 i 種代幣 coin[i]，小於 coin[i] 的代幣總額不會超過 coin[i] , 所以 Greedy 是對的。 在[1, 4, 5]的狀況，就沒有這個性質了。

## 排班問題

[排班問題](https://hackmd.io/@5568KE/排班問題)

> 待修

## 單欄位資料排序

### 題目 
輸入格式:第一行是一個正整數 n ，第二行有 n 個非負整數是對方的戰力，第三行有 n 個非負整數是我方的戰力，同行數字間以空白間格， n 與戰力值不超過 $10^5$，輸出 : 輸出最多可能的勝場數。

範例輸入:
5
3 1 7 0 2
8 3 5 0 0
輸出 
3

### 思路

我們只要關心可以戰勝的那些場次需要誰去出戰，其他場次贏不了，就在剩下的人裡面隨便挑選去應戰。假設我方目前最弱的戰力是 a 而對方最弱的是 b，如果 a <= b，則 a對誰都不會贏，是沒用的，可以忽略。否則 a > b，則「一定有一個最佳解是派 a 去出戰 b」。 證明如下(反證法):

假設在一個最佳解中，a 對戰 x 而 y 對戰 b，我們將其交換改成「a 對 b 而 y 對x」。根據我們的假設 y >= a ， 交換後 a 可以贏 b 而 y 對 x 的戰績不會比 a 對 x差，所以交換後勝場數不會變少。

所以可以將雙方戰力先進行一次排序，然後依序由小往大考慮即可。
當然也可以由大到小考慮，基本原則是能贏的贏一點就好，也可以用優先佇列(PQ)來維護。

### 實作
```cpp
int n, enemy[100000], our[100000];
cin >> n ;
for (int i = 0 ; i < n ; i++) 
    cin >> enemy[i];
for (int i = 0 ; i < n ; i++)     
    cin >> our[i];

sort(enemy,enemy+n);
sort(our,our+n);
int win = 0;
for (int i = 0, j = 0 ; i < n && j < n ; i++) { 
    if (ours[i]> enemy[j]) { 
        win++;
        j++;
    }
}
cout << win << '\n'; 
```
## 結構資料排序
在上面的題目中，每個資料都是一個數字，接下來我們來看需要使用struct的情況(或pair)，我們來看一個經典題目

### 題目

員工想用機台，已知每台機台有可以使用的時間(開始時間，結束時間)，使用時必須全程在場，也就是不能同時使用兩台機檯，河馬希望能使用越多機台越好，可以賺更多錢，請幫忙計算最多可以參加幾場，注意，使用機台時間必須在開始時間使用否則會被其他員工搶走，以及在使用結束時，當下不可能瞬間使用另一台，也就是前一個機台的結束時間需小於下一個機台的開始時間。

範例輸入
4
1 4
0 3
3 4
4 6
輸出
2

### 思路
這個題目可以想成，數線上有若干線段，要挑選出最多的不重疊線段。 可能會想，就盡量挑短的，但是看範例就知道，挑短的不是正確答案。所以轉個方向想，若 [x,y] 是所有現存線段中右端點最小的，那麼一定有個最佳解是挑選 [x,y]。

證明: 假設最佳解沒有挑[x,y]，令 [a,b] 是最佳解中右端點最小的。 根據我們的假設， y <= b,因此將[x,y] 取代 [a,b] 不會重疊到任何最佳解中的其他線段，所以就得到另外一個最佳解包含[x,y]。

根據此性質，我們可以不斷地將目前最小右端挑選進答案，當然要忽略有重疊的線段，所以只需要依照右端由小排到大，從頭到尾掃描一次就可以得到答案。用struct做排序，要定義一個比較函數，不熟的話記得回去看~

### 實作
```cpp
#include <bits/stdc++.h>
using namespace std;

struct se{
    int f;
    int e;
};

bool cmp (se a, se b){
    return a.e < b.e;
}

int main(void){
    int n;
    cin >> n;
    vector<se>men(n);
    
    for (int i = 0 ; i < n; i++) cin >> men[i].f >> men[i].e;
    sort(men.begin(),men.end(),cmp);
    
    int endtime = -1, total = 0;
    for (int i = 0 ; i < n ; i++) {
        if (men[i].f >= endtime) {
            total++;
            endtime = men[i].e;
        }
    }
    cout << total << '\n';
}
```

## 貪心二分搜


### 題目

直線上有 N 個要服務的點，每架設一座基地台可以涵蓋直徑 R 範圍以內的服務點。輸入服務點的座標位置以及一個正整數 K ，請問：在架設 K 座基地台以及每個基地台的直徑皆相同的條件下，基地台最小直徑 R 為多少? 
(座標範圍不超過 1e9,1≤ K < N ≤ 5e4。)

輸入第一行為N，K
接下來N行為座標

範例輸入: 
6 2
5 2 1 7 5 8

範例結果:
3

### 思路
服務點可以看成數線上的點，基地台可以看成數線上的線段，所以我們要計算以 K 根長度相同的線段蓋住所有的點，求最小的線段長度。

要解決這個問題，先看以下問題：
輸入給數線上 N 個點以及 R，請問最少要用幾根長度 R 的線段才能蓋住所有輸入點。

我們以貪心法則來思考。考慮座標值最小的點 P，如果有一個解的最左端點小於 P，我們可以將此線段右移到左端點對齊 P，這樣的移動不會影響解的合法性，因此，可以得到以下結論：「一定有一個最佳解(最少的 K)是將一根線段的左端放在最小座標點上。」因此，我們可以放上一個線段在 [P,P+R] ,然後略過所有以被蓋住的點，對剩下的點繼續此步驟。重複執行到所有的點被蓋住,就可以找到最小的 K 值。

回到我們的原來問題。假設 $f(R)$ 是給定長度 R 的最少線段數，那麼上述的演算法可以求得 $f(R)$。 所以，當 R 增加時，$f(R)$必然只會相同或減少，因為用更長的線段去蓋相同的點，不會需要更多線段。所以我們可以二分搜來找出最小的 R，滿足 $f(R)$ <= K。

### 實作

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> sit;

//greedy
bool check (int n, int k, int r) {
    int cnt = k, endline = -1 ;
    for (int i = 0 ; i < n ; i++) {
        if (endline >= sit[i]) continue;
        if (cnt == 0) return false;
        
        cnt--;
        endline = sit[i] + r ;
    }
    return true;
}

int main()
{
    ios::sync_with_stdio(false),cin.tie(0);
    int n, k;
    cin >> n >> k;
    sit.resize(n);
    for (int i = 0 ; i < n ; i++) {
        cin >> sit[i];
    }
    sort(sit.begin(),sit.end());
    
    //二分搜的部分 找出最小不符合的答案
    int ans = 0, L = sit[n-1] - sit[0];
    for (int jump = L/2 ; jump > 0 ; jump /= 2) {
        while (ans + jump < L && !check(n,k,ans+jump)) 
            ans += jump;
    }
    cout << ans + 1 << '\n';
    return 0;
}

```

## 掃描線演算法 sweep-line
### 介紹
掃描線演算法的名字來自在解某些幾何問題時，想像用一條掃描線沿著某個方向掃描，過程中記錄某些資料或維護某些結構。因此與貪心演算法類似，往往第一步就是排序。常見的題目有數線上的點或線段，以及平面上的點或直線。

### 題目

輸入數線上的 N 個線段，計算線段聯集的總長度。N 不超過 1e5，座標絕對值皆不超過 1e8

範例輸入：
5
10 20
20 20
30 75
5 15
40 80

範例結果：
65

### 簡單思路

簡單想法(不好)： 在每個線段的左界到右界開個大陣列，標記為1，最後在算出整個陣列中有幾個為1，此方法只適用於小範圍，因為時間複雜度跟空間複雜度都看線段總長度。

### 實作 (假設題目N，以及座標很小)
```cpp
#include <bits/stdc++.h>
using namespace std;

int box[1005];

int main()
{
    ios::sync_with_stdio(false),cin.tie(0);
    int n;
    cin >> n;
    for (int i = 0 ; i < n ; i++) {
        cin >> l >> r ;
        for (int j = l ; j < r ; j++) box[j] = 1;
    }
    int ans = 0;
    for (int i = 1 ; i <= 1000 ; i++) {
        if (box[i]) ans ++;
    }
    cout << ans << '\n';
    return 0;
}

```
### 掃描線思路

想像有跟掃描線，從左掃到右邊，碰到線的左端時，代表要加入新的線段進來，碰到右端，代表一個線段的離開，所以我們只需將線段左端做排序，逐一將線段加入，並且維護好線段的最後一根就好了。

### 實作
實作的部分我分成了使用struct跟pair，讓大家多熟悉這兩者的用法，並且在sort的部分，struct需要獨立寫一個function，而pair不用。

```cpp
// struct 做法
#include <bits/stdc++.h>
using namespace std;

struct info {
    int left,right;  
};

vector<info> box;

bool cmp (info a, info b) {
    return a.left < b.left ;
}

int main()
{ 
    ios::sync_with_stdio(false),cin.tie(0);
    int n;
    cin >> n;
    box.resize(n);
    for (int i = 0 ; i < n ; i++) {
        cin >> box[i].left >> box[i].right;
    }
    sort(box.begin(),box.end(),cmp);
    //下面開始掃描
    int ans = 0;
    info last = box[0];
    for (int i = 1 ; i < n ; i++) {
        if (box[i].left > last.right) {
            ans += last.right - last.left;
            last = box[i];
            continue;
        }
        last.right = max(last.right,box[i].right);
    }
    ans += last.right - last.left; //最後一筆會出來，不要忘記
    cout << ans << '\n';
    return 0;
}
```

```cpp
// pair 做法
#include <bits/stdc++.h>
using namespace std;

vector<pair<int,int>> box;


int main()
{  
    ios::sync_with_stdio(false),cin.tie(0);
    int n;
    cin >> n;
    box.resize(n);
    for (int i = 0 ; i < n ; i++) {
        cin >> box[i].first >> box[i].second;
    }
    
    sort(box.begin(),box.end());
    
    int ans = 0;
    pair<int,int> last = box[0];
    for (int i = 1 ; i < n ; i++) {
        if (box[i].first > last.second) {
            ans += last.second - last.first;
            last = box[i];
            continue;
        }
        last.second = max(last.second,box[i].second);
    }
    ans += last.second - last.first; //最後一筆會出來，不要忘記
    cout << ans << '\n';
    return 0;
}
```

