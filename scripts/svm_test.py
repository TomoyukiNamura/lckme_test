#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### 初期処理 =======================================================

## ディレクトリ変更
import os
os.chdir('/Users/tomoyuki/python_workspace/lckme_test')

## パッケージ読み込み
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import pandas as pd
from copy import deepcopy
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import kernel_mean as km
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


plt.style.use('ggplot') 



### データ用意=========================================================

# アヤメデータセットを用いる
iris = datasets.load_iris()

#クラスラベルを取得
y = iris.target
tmp_boolean = y!=0
y = y[tmp_boolean]

y[y==2] = 0

# 例として、3,4番目の特徴量の2次元データで使用
X = iris.data[:, [2,3]]
X = X[tmp_boolean,:]

# 正規誤差
e = np.ones((y.shape[0], 2))
e[:,0] = np.random.randn(y.shape[0])/1
e[:,1] = np.random.randn(y.shape[0])/1

X = X + e


id_data = np.arange(0,y.shape[0],1)


# プロット
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])



### oneleaveout=========================================================
pred = []
truth = []

for i in range(y.shape[0]):
    # 訓練・評価データ
    is_train = id_data!=i
    
    X_train = deepcopy(X[is_train,:])
    y_train = deepcopy(y[is_train])
    
    X_test = deepcopy(X[is_train==False,:])
    y_test = deepcopy(y[is_train==False])
    
    # データの標準化処理
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # ペアワイズ距離計算
    pair_dist = km.calcPairwiseDist(X_train_std, method='method1')
    
    # svm
#    model = SVC(kernel='rbf', gamma=1.0/np.median(pair_dist),C=100/np.sqrt(y_train.shape[0]))
    model = SVC(kernel='rbf',C=10/np.sqrt(y_train.shape[0]))

    #モデルの学習。fit関数で行う。
    model.fit(X_train_std, y_train)
    
#    # トレーニングデータに対する精度
#    pred_train = model.predict(X_train_std)
#    accuracy_train = accuracy_score(y_train, pred_train)
#    print('トレーニングデータに対する正解率： %.2f' % accuracy_train)
#    
#    #分類結果を図示する    
#    fig = plt.figure(figsize=(7,4))
#    plot_decision_regions(X_train_std, y_train, clf=model,  res=0.02)
#    plt.show()
    
    # 予測
    pred.append(model.predict(X_test_std)[0])
    truth.append(y_test[0])


pred = np.array(pred)
truth = np.array(truth)

accuracy_test = accuracy_score(truth, pred)
print('評価データに対する正解率： %.2f' % accuracy_test)


