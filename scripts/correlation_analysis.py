#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

相関分析

"""



### 初期処理 =======================================================

## ディレクトリ変更
import os
os.chdir('/Users/tomoyuki/python_workspace/lckme_test')

## パッケージ読み込み
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

## 設定ファイル，関数ファイル読み込み
from cfg import config as cfg
from functions import function as func
from functions import data_generator
from functions import plotter


### 結果フォルダ作成=========================================================
output_pass = f'output/correlation_analysis'
func.makeNewFolder(output_pass)


## 実験用データ作成=========================================================
from cfg import config as cfg

if cfg.data_type=='linear':
    train,test,truth_line = data_generator.generateLinearModelData(n_train=cfg.n_train,n_test=cfg.n_test,dim_x=cfg.dim_x,dim_y=cfg.dim_y,coef_lm=cfg.coef_lm,var_eps=cfg.var_eps)
    
elif cfg.data_type=='nonlinear':
    train,test,truth_line = data_generator.generateNonLinearModelData(cfg.n_train,cfg.n_test,cfg.dim_x,cfg.dim_y,cfg.nc_nlm,cfg.sigma_u_nlm,cfg.var_eps)
  
plotter.plotTrainTest(train, test, truth_line)


# ペアワイズ距離=========================================================
from sklearn import kernel_mean as km

tmp_X = train['y']
emb_tmp_X = km.KernelMean(tmp_X,kernel='rbf',gamma=1.0/np.median(km.calcPairwiseDist(tmp_X, method='method1')))
#emb_tmp_X = km.KernelMean(tmp_X,kernel='rbf',gamma=1.0/1.0)
tmp_x_list = np.arange(min(tmp_X),max(tmp_X),0.0001).reshape(-1,1)

plt.hist(tmp_X,bins=50,density=True,color='orange',alpha=0.5)
plt.plot(tmp_x_list.reshape(-1,),emb_tmp_X.estimate(tmp_x_list),c='r')
plt.grid()



## peason vs hsic ======================================
#peason
pd.DataFrame(train['x']).corr()
print(np.corrcoef(train['x'][:,0],train['y'][:,0]))


#hsic
from sklearn import kernel_mean as km
emb_x = km.KernelMean(train['x'][:,0],kernel='rbf',gamma=1.0/np.median(km.calcPairwiseDist(train['x'], method='method1')))
emb_y = km.KernelMean(train['y'][:,0],kernel='rbf',gamma=1.0/np.median(km.calcPairwiseDist(train['y'], method='method1')))
print(km.hsic(emb_x,emb_y)/km.hsic(emb_x,emb_x))




# mic
from minepy import MINE

mine = MINE()
mine.compute_score(train['x'][:,0],train['y'][:,0])
mine.mic()



df = pd.DataFrame(train['x'])


import numpy as np
import pandas as pd
from minepy import MINE

def mic(df):
    result = np.eye(df.shape[1])
    
    for i in range(df.shape[1]):
        for j in range(df.shape[1]):
            mine = MINE()
            mine.compute_score(df.iloc[:,i],df.iloc[:,j])
            result[i,j] = mine.mic()
    
    result = pd.DataFrame(result)
    result.columns = df.columns
    result.index = df.columns
    
    return result


mic(df)

import seaborn as sns
corr_mat=mic(df)
sns.heatmap(corr_mat,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True, # True:格子の中に値を表示
            fmt='.2f',
            xticklabels=corr_mat.columns.values,
            yticklabels=corr_mat.columns.values
           )
plt.show()

## ======================================



