#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

局所カーネル平均の実験

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
from functions import do_experiment


## 実験用データ作成=========================================================
from cfg import config as cfg

if cfg.data_type=='linear':
    train,test,truth_line = data_generator.generateLinearModelData(n_train=cfg.n_train,n_test=cfg.n_test,dim_x=cfg.dim_x,dim_y=cfg.dim_y,coef_lm=cfg.coef_lm,var_eps=cfg.var_eps)
    
elif cfg.data_type=='nonlinear':
    train,test,truth_line = data_generator.generateNonLinearModelData(cfg.n_train,cfg.n_test,cfg.dim_x,cfg.dim_y,cfg.nc_nlm,cfg.sigma_u_nlm,cfg.var_eps)
  
plotter.plotTrainTest(train, test, truth_line)



### 結果フォルダ作成=========================================================
from cfg import config as cfg

output_pass = f'output/{datetime.now().strftime("%Y%m%d")}'
func.makeNewFolder(output_pass)

output_pass = f'output/{datetime.now().strftime("%Y%m%d")}/{cfg.data_type}'
func.makeNewFolder(output_pass)

output_pass = f'output/{datetime.now().strftime("%Y%m%d")}/{cfg.data_type}/r_subsample_{cfg.r_subsample}'
func.makeNewFolder(output_pass)


### 実験 =============================================================

# 実験実施
result = do_experiment.doExperiment(train, test, cfg.sigma_x_list, cfg.sigma_y_list, cfg.alpha_list, 
                                    test['y'], cfg.var_eps, output_pass,truth_line)



## 結果のプロット ====================================================================
# ヒートマップ
plotter.heatmapResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,j=0,func='mean',output_pass=output_pass)

plotter.heatmapResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,j=0,func='max',output_pass=output_pass)

# 2d result
plotter.doPlot2dResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,output_pass)


