#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# 訓練データ数
max_n_train = 5000
#max_n_train = 1000

# 評価データ数
#n_test  = 1000
n_test  = 100

# 説明変数の次元数
dim_x = 10

# 目的変数の次元数
dim_y = 1

# 実験データのタイプ
#data_type = 'linear'
data_type = 'nonlinear'


## data_type = 'nonlinear'のパラメータ
nc_nlm = 3
sigma_u_nlm = 0.3

## 共通のパラメータ
var_eps = 0.001



## 各モデルのパラメータ
# カーネルパラメータ・正則化パラメータ(全手法共通)
lv = -3
uv = 3


sigma_x_list = 10.0**np.arange(lv,uv+1)
sigma_y_list = np.array([0.01])
alpha_list   = 10.0**np.arange(lv,uv+1)


# サブサンプルの訓練データ数に対する割合(rss,rff,nys,dc,lckm)
r_subsample = 0.1

# 弱学習器の個数(dc)
n_weaklearner = 10


# 比較対象モデルリスト
#model_name_list = ['ckm', 'icf', 'rss', 'nw', 'rff', 'nys', 'dc', 'lckm']
model_name_list = ['ckm', 'rss', 'nw', 'rff', 'nys', 'dc', 'lckm']



# 部分サンプルによる実験回数
n_exp = 10

# 限界サンプル数
limit_sumple_num = 1500
#limit_sumple_num = 300



