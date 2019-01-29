#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

局所カーネル平均の実験d

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
import shutil

## 設定ファイル，関数ファイル読み込み
from cfg import config as cfg
from functions import function as func
from functions import data_generator
from functions import plotter
from functions import do_experiment



### 結果フォルダ作成=========================================================
today = datetime.now().strftime("%Y%m%d")

output_pass = f'output/{today}'
func.makeNewFolder(output_pass)

output_pass = f'output/{today}/{cfg.data_type}'
func.makeNewFolder(output_pass)

#output_pass = f'output/{today}/{cfg.data_type}/r_subsample_0.1'
#func.makeNewFolder(output_pass)
#
#output_pass = f'output/{today}/{cfg.data_type}/r_subsample_0.5'
#func.makeNewFolder(output_pass)

# cfgファイルをoutputにコピー
shutil.copyfile("cfg/config.py", f"{output_pass}/config.py")


### 実験 =============================================================
# 実験用データ作成

for i_experiment in range(cfg.n_experiment):
    
#    output_pass_org = f'output/{today}/{cfg.data_type}/r_subsample_0.1/{i_experiment}'
    output_pass_org = f'output/{today}/{cfg.data_type}/{i_experiment}'
    func.makeNewFolder(output_pass_org)
    
    # プロット用結果格納場所
    best_norm_error_list = {}
    
    for id_dim_x in range(len(cfg.dim_x_list)):
        
        # 
        output_pass = f'{output_pass_org}/dim_x_{cfg.dim_x_list[id_dim_x]}'
        func.makeNewFolder(output_pass)
    
        # 人工データ発生
        if cfg.data_type=='linear':
            train,test,truth_line = data_generator.generateLinearModelData(n_train=cfg.n_train, n_test=cfg.n_test ,dim_x=cfg.dim_x_list[id_dim_x] ,dim_y=cfg.dim_y ,var_eps=cfg.var_eps)
            
        elif cfg.data_type=='nonlinear':
            train,test,truth_line = data_generator.generateNonLinearModelData(n_train=cfg.n_train ,n_test=cfg.n_test ,dim_x=cfg.dim_x_list[id_dim_x] ,dim_y=cfg.dim_y ,nc_nlm=cfg.nc_nlm ,sigma_u_nlm=cfg.sigma_u_nlm ,var_eps=cfg.var_eps)
            
        # 人工データプロット
        plotter.plotTrainTest(train, test, truth_line, output=output_pass)
        
        
        # 実験実施(サブサンプル率別)
        experiment_params = {}
        experiment_params['model_name_list'] = cfg.model_name_list
        experiment_params['r_subsample']     = cfg.r_subsample
        experiment_params['n_weaklearner']   = cfg.n_weaklearner
        
        result = do_experiment.doExperiment(train=train, test=test,experiment_params=experiment_params , sigma_x_list=cfg.sigma_x_list, sigma_y_list=cfg.sigma_y_list, alpha_list=cfg.alpha_list, 
                                            mu_bar_list=test['y'], v_bar=cfg.var_eps, output_pass=output_pass,truth_line=truth_line)
        
        # 結果プロット(ヒートマップ)
        plotter.heatmapResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,j=0,func='mean' ,output_pass=output_pass)
        plotter.heatmapResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,j=0,func='max' ,output_pass=output_pass)
    
        # 結果プロット(2d result)
        #plotter.doPlot2dResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,output_pass)
    
        # 最良のRKHSnorm/sup_norm_errorを取得
        best_norm_error_list[f'dim_x_{cfg.dim_x_list[id_dim_x]}'] = func.selectBestNormErrorInParams(result=result,sigma_x_list=cfg.sigma_x_list ,sigma_y_list=cfg.sigma_y_list ,alpha_list=cfg.alpha_list)
    
    # ベストErrorのプロット
    plotter.plotDimVsError(best_norm_error_list=best_norm_error_list, dim_x_list=cfg.dim_x_list, plot_target='best_RKHS_norm_error', output=output_pass_org)
    plotter.plotDimVsError(best_norm_error_list=best_norm_error_list, dim_x_list=cfg.dim_x_list, plot_target='best_sup_norm_error', output=output_pass_org)





#import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd
#
#color_map = ['r','g','b','orange','m','y','c']
#
#
#
#R_FIGSIZE = 2
#plt.rcParams['figure.figsize'] = [6.0*R_FIGSIZE, 4.0*R_FIGSIZE]
#plt.rcParams['font.size'] = 10.0*R_FIGSIZE
#
#tmp_df = pd.read_csv(f"{output_pass}/heatmap_RKHSnormError_max_ckm.csv", index_col=0)
#
#
#sns.heatmap(tmp_df, annot=True, fmt="1.2f", linewidths=.5, cmap="Reds", vmin=0.0,vmax=3.0)
#plt.title(f"RKHS norm error (ckm)")
#plt.xlabel("ε (10^x)")
#plt.xlabel("σx (10^x)")





    
