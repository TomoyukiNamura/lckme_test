#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

## ディレクトリ変更
os.chdir('/Users/tomoyuki/python_workspace/kernelmean')


## データ生成(x:1次元,y:1次元)===============================================
n = 50
N = 1000
n_test = 10

# 学習データ
x = np.linspace(-3, 3, n)
pix = np.pi * x
y = np.sin(pix) / pix + 0.1 * x + 0.2 * np.random.randn(n)

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

plt.scatter(x,y)


# 評価データ
x_test = np.linspace(-3, 3, n_test)
pix_test = np.pi * x_test
y_test = np.sin(pix_test) / pix_test + 0.1 * x_test + 0.2 * np.random.randn(n_test)

x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

plt.scatter(x_test,y_test)


# plot用x軸データ
X = np.linspace(-3, 3, N)
X = X.reshape(-1, 1)




## データ生成(x:4次元,y:1次元)===============================================
n = 50
n_test = 10

# インプットを適当に生成
X1 = np.sort(5 * np.random.rand(n+n_test, 1).reshape(n+n_test), axis=0)
X2 = np.sort(3 * np.random.rand(n+n_test, 1).reshape(n+n_test), axis=0)
X3 = np.sort(9 * np.random.rand(n+n_test, 1).reshape(n+n_test), axis=0)
X4 = np.sort(4 * np.random.rand(n+n_test, 1).reshape(n+n_test), axis=0)

# インプットの配列を一つに統合
tmp_x = np.c_[X1, X2, X3, X4]

# アウトプットを算出
tmp_y = np.sin(X1).ravel() + np.cos(X2).ravel() + np.sin(X3).ravel() - np.cos(X4).ravel()
y_o = tmp_y.copy()

# ノイズを加える
tmp_y[::5] += 3 * (0.5 - np.random.rand(tmp_y[::5].shape[0]))

# 訓練データ，評価データに分割
y = tmp_y[0:n]
y_test = tmp_y[(n+1):(n+n_test)]

x = tmp_x[0:n]
x_test = tmp_x[(n+1):(n+n_test)]


x.shape
y.shape

plt.scatter(x[:,1],y)
plt.scatter(x[:,2],y)



## kernel ridge regression =======================================
import sklearn
from sklearn.kernel_ridge import KernelRidge

# KRR
clf = KernelRidge(alpha=1.0, kernel='rbf',gamma=1)
clf.fit(x, y)

# prediction
y_pred_clf = clf.predict(x_test)

# plot & score
plt.scatter(x, y)# 訓練データ
plt.plot(X, clf.predict(X))# 予測曲線の表示
plt.scatter(x_test, y_pred_clf)# 予測値
plt.scatter(x_test, y_test)# 正解値
#print(clf.score(x, y))





## Conditional Kernel Mean =======================================
from sklearn import kernel_mean as km
#from sklearn.kernel_mean import KernelMean
#from sklearn.kernel_mean import ConditionalKernelMean

# kernel mean
emb_x = km.KernelMean(x,kernel='rbf',gamma=1)
emb_y = km.KernelMean(y,kernel='rbf',gamma=1)

# 内積，ノルム，距離
km.innerProduct(emb_x,emb_y)
km.innerProduct(emb_y,emb_x)
km.RKHSnorm(emb_x)
km.RKHSdist(emb_x, emb_y)
km.RKHSdist(emb_y, emb_x)
km.RKHSdist(emb_x, emb_x)

# conditional kernel mean
ckm = km.ConditionalKernelMean(alpha=1.0)
ckm.fit(emb_x, emb_y)
emb_y_pred = ckm.predict(x_test)


GramMatrix = ckm.emb_X._get_kernel(x_test[0:3,:],x_test[0:4,:])

GramMatrix = np.round(ckm.emb_X._get_kernel(y,y),2)




# 重み付き和で予測値計算
y_pred_ckm = []
for i in range(len(emb_y_pred)):
    y_pred_ckm.append(emb_y_pred[i].weighted_sum())

y_pred_ckm = np.array(y_pred_ckm)


# plot & score
plt.scatter(x, y)# 訓練データ
plt.plot(X, clf.predict(X))# 予測曲線の表示
plt.scatter(x_test, y_pred_ckm)# 予測値
plt.scatter(x_test, y_test)# 正解値





## Local Conditional Kernel Mean =======================================
from sklearn.kernel_mean import KernelMean
from sklearn.kernel_mean import LocalConditionalKernelMean

# kernel mean
emb_x = KernelMean(x,kernel='rbf',gamma=1)
emb_y = KernelMean(y,kernel='rbf',gamma=1)

# local conditional kernel mean
knn_k=5
lckm = LocalConditionalKernelMean(alpha=1.0)
lckm.fit(emb_x, emb_y)
emb_y_pred = lckm.predict(x_test,knn_k)


# 重み付き和
y_pred_lckm = []
for i in range(len(emb_y_pred)):
    y_pred_lckm.append(emb_y_pred[i].weighted_sum())

y_pred_lckm = np.array(y_pred_lckm)



# plot & score
plt.scatter(x, y)# 訓練データ
plt.plot(X, clf.predict(X))# 予測曲線の表示
plt.scatter(x_test, y_pred_lckm)# 予測値
plt.scatter(x_test, y_test)# 正解値



knn_id =np.argsort(emb_x._get_kernel(emb_x.X, x_test)[:,1])[::-1][0:5]
emb_y.X[knn_id,:]
emb_y.weights




## Nystroem Conditional Kernel Mean =======================================
from sklearn.kernel_mean import KernelMean
from sklearn.kernel_mean import NystroemConditionalKernelMean

# kernel mean
emb_x = KernelMean(x,kernel='rbf',gamma=1)
emb_y = KernelMean(y,kernel='rbf',gamma=1)

# conditional kernel mean
nckm = NystroemConditionalKernelMean(alpha=1.0, m_subsample=5)
nckm.fit(emb_x, emb_y)
emb_y_pred = nckm.predict(x_test)


K_X = nckm.emb_X._get_kernel(nckm.emb_X.X, x_test)
K_X.shape
nckm.inv_gram_.shape


np.dot(nckm.inv_gram_, K_X).shape


# 重み付き和で予測値計算
y_pred_nckm = []
for i in range(len(emb_y_pred)):
    y_pred_nckm.append(emb_y_pred[i].weighted_sum())

y_pred_nckm = np.array(y_pred_nckm)



# plot & score
plt.scatter(x, y)# 訓練データ
plt.plot(X, clf.predict(X))# 予測曲線の表示
plt.scatter(x_test, y_pred_nckm)# 予測値
plt.scatter(x_test, y_test)# 正解値