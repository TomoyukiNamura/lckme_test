#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# 実験回数
#n_experiment = 5
n_experiment = 1

# 訓練データ数
#n_train = 1000
n_train = 300

# 評価データ数
#n_test  = 100
n_test  = 10

# 説明変数の次元数
#dim_x = 1
dim_x_list = [1,3,5,7,10]
#dim_x_list = [7,10]

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
#coef_lm = np.append(np.array([0.0]),np.ones(dim_x))

## data_type = 'nonlinear'のパラメータ
nc_nlm = 3
sigma_u_nlm =1# 0.3

## 共通のパラメータ
var_eps = 0.001



## 各モデルのパラメータ
# カーネルパラメータ・正則化パラメータ(全手法共通)
sigma_y_list = np.array([0.01])

lv_sigma = -3
uv_sigma = 5
sigma_x_list = 10.0**np.arange(lv_sigma,uv_sigma+1)

lv_alpha = -10
uv_alpha = -3
alpha_list   = 10.0**np.arange(lv_alpha,uv_alpha+1)


# サブサンプルの訓練データ数に対する割合(rss,rff,nys,dc,lckm)
r_subsample = 0.1

# 弱学習器の個数(dc)
n_weaklearner = 10


# 比較対象モデルリスト
#model_name_list = ['ckm', 'icf', 'rss', 'nw', 'rff', 'nys', 'dc', 'lckm']
model_name_list = ['ckm', 'nw', 'rff', 'lckm']
