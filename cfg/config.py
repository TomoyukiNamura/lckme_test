#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# 訓練データ数
n_train = 1000

# 評価データ数
n_test  = 100

# 説明変数の次元数
dim_x = 3

# 目的変数の次元数
dim_y = 1

# 実験データのタイプ
#data_type = 'linear'
data_type = 'nonlinear'



## data_type = 'linear'のパラメータ
## 平均ベクトル
#mean_lm = [0.0, 0.0]
#
## 分散今日分散行列
#cov_lm = [[1.0, 0.8], 
#          [0.8, 1.0]]

#coef_lm = np.array([0.0,0.03])
coef_lm = np.append(np.array([0.0]),np.ones(dim_x))

## data_type = 'nonlinear'のパラメータ
nc_nlm = 1000
sigma_u_nlm = 0.1

## 共通のパラメータ
var_eps = 0.001



## 各モデルのパラメータ
# カーネルパラメータ・正則化パラメータ(全手法共通)
lv = -5
uv = 3

sigma_x_list = 10.0**np.arange(lv,uv+1)
sigma_y_list = np.array([0.01])
alpha_list   = 10.0**np.arange(lv,uv+1)


# サブサンプルの訓練データ数に対する割合(rss,rff,nys,dc,lckm)
r_subsample = 0.1

# 弱学習器の個数(dc)
n_weaklearner = 10
