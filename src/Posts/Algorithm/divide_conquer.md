---
title: 分治法
# icon: material-symbols:add-notes-outline
order: 6
date: 2023-01-02
category:
  - Algorithm

tag:
  - note
  - lecture
---


# Divide and Conquer (分治法)

## 簡介
分治演算法(Divide-and-Conquer algorithm)，又稱分而治之演算法，也有人稱為各個擊破法。分治是一種非常重要的演算法思維模式與策略，有很多重要的演算法都是根據分治的思維模式，例如快速排序法、合併排序法、快速傅立葉轉換(FFT)、矩陣乘法、整數乘法以及在一些在計算幾何的知名演算法都是分治的策略。相較於它的重要性，分治出現在題目中的比例卻不是那麼高。

<!--more-->

主要原因有二:
1. 很多分治算法的問題都有別的解法的版本，樹狀圖的分治也往往歸類到動態規劃。
2. 一些重要的分治算法已經被納入在庫存函數中，例如排序，還有一些則因為太難而不適合出題。

即使如此，當碰到一個不曾見過的問題，分治往往是第一個思考的重點，因為分治的架構讓我們很容易找到比暴力要好的方法。

## 核心想法

分治法主要有三個步驟，切割，解決，合併。

- 切割：把問題以「相同問題」切割成更多小的子問題。
- 解決：把切割完的子問題解決。
- 合併：把解決完的子問題合併起來成為問題的答案。

注意，把問題切割並不是將問題切成不同的問題，而是相同的問題，化成很多比較小的輸入資料。

## 最大值與最小值問題

### 問題
在一組數據中，找到其最大值與最小值

### 思路
- 切割：把區間分成好幾個部分
- 解決：分別求出每個分割出來的最大最小值
- 合併：把每個區間的最大最小值回傳，並且比較誰最大(小)

PS：這個問題太簡單了，直接解就可以了，這邊用分治只是為了更清楚分治的流程。

### 實作
(這邊使用struct/pair讓大家多熟練一下)
```cpp
//struct 實作
#include <bits/stdc++.h>
using namespace std;

struct info {
    int mx;
    int mn;
};

vector<int> ls;

info solve (int left, int right) {
    info ans ;
    if (right - left <= 1) {
        ans.mx = max(ls[left],ls[right]);
        ans.mn = min(ls[left],ls[right]);
        return ans;
    }
    ans.mx = max(solve(left,(left+right)/2).mx , solve((left+right)/2 + 1,right).mx);
    ans.mn = min(solve(left,(left+right)/2).mn , solve((left+right)/2 + 1,right).mn);
    return ans;
}

int main()
{
    int n;
    cin >> n;
    ls.resize(n);
    for (int i = 0 ; i < n ; i++) {
        cin >> ls[i];
    }
    info ans = solve(0,n-1);
    cout << "max = " << ans.mx << " , min = " << ans.mn << '\n';
}

```
```cpp
//pair實作
#include <bits/stdc++.h>
using namespace std;

vector<int> ls;
//假設pair的第一個是最大值，第二個是最小值
pair<int,int> solve (int left, int right) {
    pair<int,int> ans ;
    if (right - left <= 1) {
        ans.first = max(ls[left],ls[right]);
        ans.second = min(ls[left],ls[right]);
        return ans;
    }
    ans.first = max(solve(left,(left+right)/2).first , solve((left+right)/2 + 1,right).first);
    ans.second = min(solve(left,(left+right)/2).second , solve((left+right)/2 + 1,right).second);
    return ans;
}

int main()
{
    int n;
    cin >> n;
    ls.resize(n);
    for (int i = 0 ; i < n ; i++) {
        cin >> ls[i];
    }
    pair<int,int> ans = solve(0,n-1);
    cout << "max = " << ans.first << " , min = " << ans.second << '\n';
}

```
## 回顧合併排序法(待補)

### 實作

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> ls;

vector<int> merge_sort (int left, int right) {
    if (right - left <= 1) {
        if (left == right) {
            return {ls[left]};
        }else {
            if (ls[left] < ls[right]) return {ls[left],ls[right]};
            else return {ls[right],ls[left]};
        }
    }
    vector<int> m1 = merge_sort(left,(left+right)/2);
    vector<int> m2 = merge_sort((left+right)/2 + 1,right);
    vector<int> merge_list(right - left + 1);
    
    int l1 = 0 , l2 = 0, now = 0;
    while (l1 != m1.size() || l2 != m2.size()) {
        if (l1 == m1.size()) {
            merge_list[now++] = m2[l2++];
        }else if (l2 == m2.size()) {
            merge_list[now++] = m1[l1++];
        }else {
            if (m1[l1] < m2[l2])
                merge_list[now++] = m1[l1++];
            else 
                merge_list[now++] = m2[l2++];
        }
    }
    return merge_list;
}

int main()
{
    ios::sync_with_stdio(false),cin.tie(0);
    int n ;
    cin >> n;
    ls.resize(n);
    for (int i = 0 ; i < n ; i++) cin >> ls[i];
    ls = merge_sort(0,n-1);
    for (int i = 0 ; i < n ; i++) cout << ls[i] << ' ';
    cout << '\n';
    return 0;
}

```

## 回顧快速排序法(待補)

### 實作

```cpp
#include<bits/stdc++.h>
using namespace std;

vector<int> vec;

void quicksort (int first, int end) {
    if (first >= end) return;
        int j = first+1;
        for (int i = first+1 ; i <= end ; i++) {
        if (vec[i]<vec[first]) {
            swap(vec[j],vec[i]);
            j++;
        }
    }

    swap(vec[first],vec[j-1]);
    int mid=j-1;
    quicksort(first,mid);
    quicksort(mid+1,end);
}

int main(){
    ios::sync_with_stdio(false),cin.tie(0);
    vector<int> temp;
    int n;
    cin >> n;
    temp.resize(n);
    for (int i = 0 ; i < n ; i++) cin >> temp[i];
    quicksort(0,n-1);
    for (int i = 0 ; i < n ; i++) cout << temp[i] << " ";
}
```

## 分治的複雜度
分治是一個遞迴的演算法，不像迴圈的複雜度可以用加總的方法或是乘法計算，遞迴的複雜度是由遞迴關係式(recurrence relation)所表達。計算複雜度必須解遞迴關係式。遞迴關係又稱為差分方程式(difference equation)，解遞迴關係是個複雜的事情。

分治演算法的常見形式是將一個大小為 n 的問題切成 a 個大小為 b 的子問題此外必須做一些分割與合併的工作。假設大小為 n 的問題的複雜度是 T(n)，而分割合併需要的時間是 f(n)，我們可以得到以下遞迴關係:

T(n) = a * T(n/b) + f(n)，這裡我們省略不整除的處理，因為只計算 big-O，通常不會造成影響。這個遞迴式子幾乎是絕大部分分治時間複雜度的共同形式，所以也有人稱為 divide-and-conquer recurrences，對於絕大部分會碰到的狀況 (a,b, f(n))，這個遞迴式都有公式解，有興趣的請上網查詢 [Master theorem](https://en.wikipedia.org/wiki/Master_theorem_(analysis_of_algorithms))。

分治的複雜度算法，主要有兩種方式，對數學有興趣的可以自行查詢。
1. Recursion-Tree Method搭配Substitution Method(「數學歸納法」)
2. Master Method

## 經典例子 反序問題

### 題目
考慮一個數列 A[1:n]。如果 A 中兩個數字 A[i]和 A[j]滿足 i<j 且 A[i]>A[j]，也就是在前面的比較大，則我們說(a[i],a[j])是一個反序對(inversion)。定義W(A)為數列 A 中反序對數量。

例如，在數列 A=(3,1,9,8,9,2)中，一共有(3,1)、(3,2)、(9,8)、(9,2)、(8,2)、(9,2)一共 6 個反序對，所以 W(A)=6。請注意到序列中有兩個 9 都在 2 之前，因此有兩個(9,2)反序對，也就是說，不同位置的反序對都要計算，不管兩對的內容是否一樣。請撰寫一個程式，計算一個數列 A 的反序數量W(A) 。

輸入格式:第一行是一個正整數 n，代表數列長度，第二行有 n 個非負整數，是依序數列內容，數字間以空白隔開。 n 不超過 1e5 數列內容不超過 1e6。

範例輸入:
6
3 1 9 8 9 2
範例結果:
6

### 思路

可能會想，就每個都判斷他後面的所有數字有沒有吻合就好了，但這樣的時間複雜度為$O(n^2)$，很明顯不夠好。用分治法的思路來看，。要計算一個區間的反序數量，將區間平分為兩段，一個反序對的兩個數可能都在左邊或都在右邊，否則就是跨在左右兩邊。都在同一邊的可以遞迴解，我們只要會計算跨左右兩邊的就行了，也就是對每一個左邊的元素 x，要算出右邊比 x 小的有幾個。假設對左邊的每一個，去檢查右邊的每一個，那會花 $O(n^2)$，不行，我們聚焦的目標是在不到 $O(n^2)$的時間內完成跨左右的反序數量計算。

記得之前學過的排序應用，假設我們將右邊排序，花 $O(nlog(n))$，然後，每個左邊的 x 可以用二分搜就可算出右邊有多少個小於x，這樣只要花$O(nlog(n))$的時間就可以完成合併，那麼，根據分治複雜度 $T(n) = 2\space T(n/2)+O(nlog(n))$ ， $T(n)$的結果是 $O(nlog^2(n))$。

### 實作
這邊在二分搜的部分使用stl的lower_bound。

```cpp
#include <bits/stdc++.h>
using namespace std;

int divide (vector<int> ls, int left, int right) {
    int mid = (left+right)/2;
    if (left+1 >= right) return 0 ;
    long long ans = divide(ls,left,mid) + divide(ls,mid,right);
    
    sort(ls.begin() + mid , ls.begin() + right);
    for (int i = left ; i < mid ; i++) {
        ans += lower_bound(ls.begin()+mid,ls.begin()+right,ls[i]) - (ls.begin()+mid);
    }
    return ans ;
}

int main()
{
    int n;
    cin >> n;
    vector<int> ls(n);
    for (int i = 0 ; i < n ; i++) {
        cin >> ls[i];
    }
    cout << divide(ls,0,n) << '\n';
    return 0;
}
```

也可以自己寫一個bsearch()，這邊一樣用一般的二分搜

```cpp
#include <bits/stdc++.h>
using namespace std;

int bsearch (vector<int> ls, int left, int right, int val) {
    while (left < right){
        int mid = (left + right)/2;
        if (ls[mid] < val) left = mid + 1;
        else right = mid;
    }
    return (left+right)/2;
}

int divide (vector<int> ls, int left, int right) {
    int mid = (left+right)/2;
    if (left+1 >= right) return 0 ;
    long long ans = divide(ls,left,mid) + divide(ls,mid,right);
    sort(ls.begin() + mid , ls.begin() + right);
    for (int i = left ; i < mid ; i++) {
        ans += bsearch(ls,mid,right,ls[i]) - mid;
    }
    return ans ;
}

int main()
{
    int n;
    cin >> n;
    vector<int> ls(n);
    for (int i = 0 ; i < n ; i++) {
        cin >> ls[i];
    }
    cout << divide(ls,0,n) << '\n';
    return 0;
}             
```

### 還能更快？                     
上面的寫法是$O(nlog^2(n))$，雖然已經比$O(n^2)$好的多了。不過我們可以注意到，我們只需要再回傳的時候sort就好了，不需要每一次都進行單邊排序。               
         
也就是說在分治的部分可以改寫成
```cpp
int divide (vector<int> &ls, int left, int right) {
    int mid = (left+right)/2;
    if (left+1 >= right) return 0 ;
    long long ans = divide(ls,left,mid) + divide(ls,mid,right);
    
    for (int i = left; i < mid ; i++) {
        ans += lower_bound (ls.begin()+mid,ls.begin()+right,ls[i]) - (ls.begin()+mid);
    }                     
    
    sort(ls.begin()+left,ls.begin()+right);
    return ans ;
}
```

在搜尋的時候其實跟一路慢慢逛過去的時間複雜度差不多，所以其實不需要多用二分。

```cpp
int divide (vector<int> &ls, int left, int right) {
    int mid = (left+right)/2;
    if (left+1 >= right) return 0 ;
    long long ans = divide(ls,left,mid) + divide(ls,mid,right);
    
    for (int i = left , j = mid; i < mid ; i++) {
        while (j<right && ls[j]<ls[i]) 
            j++;
        ans += j - mid;
    }                                   
    sort(ls.begin()+left,ls.begin()+right);
    return ans ;
}
```

不過這樣會因為排序所以複雜度還是$O(nlog^2(n))$啊，不過上面的這種寫法，是不是很像用merge sort呢，所以我們在一個很像merge sort裡面還用sort()，很多此一舉，只要配合merge sort的寫法，我們就可以將複雜度降到$O(nlog(n))$了。也就是在兩個排序好的資料中，本身就已經是反序數了。

所以就會改寫成

```cpp
int divide (vector<int> &ls, int left, int right) {
    int mid = (left+right)/2;
    if (left+1 >= right) return 0 ;
    long long ans = divide(ls,left,mid) + divide(ls,mid,right);
    int temp[right-left], k=0, j = mid;
    for (int i = left ; i < mid ; i++) {
        while (j<right && ls[j]<ls[i]) 
            temp[k++] = ls[j++];
        temp[k++] = ls[i];
        ans += j - mid;
    }
    
    for (k=left; k < j; k++)
        ls[k] = temp[k-left];
    return ans ;
}
```

這樣時間複雜度就從$O(nlog^2(n))$降為$O(nlog(n))$了