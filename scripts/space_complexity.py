#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

空間計算量実験

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
from copy import deepcopy
import shutil
import time


## 設定ファイル，関数ファイル読み込み
from cfg import space_complexity_conf as cfg
from functions import function as func
from functions import data_generator
from functions import plotter
from functions import do_experiment



### 結果フォルダ作成=========================================================
from cfg import space_complexity_conf as cfg

output_pass = f'output/{datetime.now().strftime("%Y%m%d")}'
func.makeNewFolder(output_pass)

i=0
mk_folder = False
while(mk_folder==False):
    i+=1
    output_pass = f'output/{datetime.now().strftime("%Y%m%d")}/space_complexity_{i}'
    mk_folder = func.makeNewFolder(output_pass)
    

# cfgファイルをoutputにコピー
shutil.copyfile("cfg/space_complexity_conf.py", f"{output_pass}/space_complexity_conf.py")



## 人工データ発生=========================================================
if cfg.data_type=='linear':
    train,test,truth_line = data_generator.generateLinearModelData(n_train=cfg.max_n_train, n_test=cfg.n_test ,dim_x=cfg.dim_x ,dim_y=cfg.dim_y ,var_eps=cfg.var_eps)
    
elif cfg.data_type=='nonlinear':
    train,test,truth_line = data_generator.generateNonLinearModelData(n_train=cfg.max_n_train ,n_test=cfg.n_test ,dim_x=cfg.dim_x ,dim_y=cfg.dim_y ,nc_nlm=cfg.nc_nlm ,sigma_u_nlm=cfg.sigma_u_nlm ,var_eps=cfg.var_eps)

# 人工データプロット
plotter.plotTrainTest(train, test, truth_line, output=output_pass)



## 実験 =========================================================
start = time.time()

# 実験パラメータの初期化
experiment_params = {}
experiment_params['model_name_list'] = deepcopy(cfg.model_name_list)
experiment_params['r_subsample']     = cfg.r_subsample
experiment_params['n_weaklearner']   = cfg.n_weaklearner


# 結果格納先を初期化
best_norm_error_list = {}

for i_exp in range(cfg.n_exp):
    print(f"##############################")
    print(f"#  ")
    print(f"#  {i_exp+1}回目")
    print(f"#  ")
    print(f"##############################")
    
    # 部分訓練データ数を計算
    n_tmp_train = int(train["x"].shape[0] / float(cfg.n_exp) * (i_exp+1))
    
    # 
    if n_tmp_train > cfg.limit_sumple_num:
        func.removeElementFromList(experiment_params['model_name_list'], "ckm")
        #func.removeElementFromList(experiment_params['model_name_list'], "nw")
    
    if n_tmp_train * experiment_params['r_subsample'] > cfg.limit_sumple_num:
        func.removeElementFromList(experiment_params['model_name_list'], "rss")
        func.removeElementFromList(experiment_params['model_name_list'], "rff")
        func.removeElementFromList(experiment_params['model_name_list'], "nys")
        func.removeElementFromList(experiment_params['model_name_list'], "dc")
        func.removeElementFromList(experiment_params['model_name_list'], "lckm")
        
    # 部分訓練データを作成
    tmp_train = {}
    tmp_train["x"] = train["x"][0:n_tmp_train]
    tmp_train["y"] = train["y"][0:n_tmp_train]
    
    # 実験開始
    result = do_experiment.doExperiment(train=tmp_train, test=test, experiment_params=experiment_params , sigma_x_list=cfg.sigma_x_list, sigma_y_list=cfg.sigma_y_list, alpha_list=cfg.alpha_list, 
                                                mu_bar_list=test['y'], v_bar=cfg.var_eps, output_pass=output_pass, truth_line=truth_line)
    
    best_norm_error_list[n_tmp_train] = func.selectBestNormErrorInParams(result, cfg.sigma_x_list, cfg.sigma_y_list, cfg.alpha_list)


# プロット
plotter.plotNtrainVsError(best_norm_error_list, model_name_list=cfg.model_name_list, plot_target="best_RKHS_norm_error", output=output_pass)
plotter.plotNtrainVsError(best_norm_error_list, model_name_list=cfg.model_name_list, plot_target="best_sup_norm_error", output=output_pass)
        

process_time = time.time() - start
print(process_time)
