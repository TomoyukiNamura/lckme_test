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



### 結果フォルダ作成=========================================================
from cfg import config as cfg

output_pass = f'output/{datetime.now().strftime("%Y%m%d")}'
func.makeNewFolder(output_pass)

output_pass = f'output/{datetime.now().strftime("%Y%m%d")}/{cfg.data_type}'
func.makeNewFolder(output_pass)

output_pass = f'output/{datetime.now().strftime("%Y%m%d")}/{cfg.data_type}/r_subsample_0.1'
func.makeNewFolder(output_pass)

output_pass = f'output/{datetime.now().strftime("%Y%m%d")}/{cfg.data_type}/r_subsample_0.5'
func.makeNewFolder(output_pass)


### 実験 =============================================================
# 実験用データ作成
from cfg import config as cfg


for i_experiment in range(5):
    
    output_pass_org = f'output/{datetime.now().strftime("%Y%m%d")}/{cfg.data_type}/r_subsample_0.1/{i_experiment}'
    func.makeNewFolder(output_pass_org)
    
    # プロット用結果格納場所
    best_norm_error_list_subsample1 = {}
    best_norm_error_list_subsample2 = {}
    
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
        for r_subsample in [0.5]:
            
            experiment_params = {}
            experiment_params['model_name_list'] = cfg.model_name_list
            experiment_params['r_subsample']     = r_subsample
            experiment_params['n_weaklearner']   = cfg.n_weaklearner
            
            result = do_experiment.doExperiment(train=train, test=test,experiment_params=experiment_params , sigma_x_list=cfg.sigma_x_list, sigma_y_list=cfg.sigma_y_list, alpha_list=cfg.alpha_list, 
                                                mu_bar_list=test['y'], v_bar=cfg.var_eps, output_pass=output_pass,truth_line=truth_line)
            
            # 結果プロット(ヒートマップ)
            plotter.heatmapResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,j=0,func='mean' ,output_pass=output_pass)
            plotter.heatmapResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,j=0,func='max' ,output_pass=output_pass)
        
            # 結果プロット(2d result)
            #plotter.doPlot2dResults(result,cfg.sigma_x_list,cfg.sigma_y_list,cfg.alpha_list,output_pass)
        
            # 最良のRKHSnorm/sup_norm_errorを取得
            if r_subsample==0.1:
                best_norm_error_list_subsample1[f'dim_x_{cfg.dim_x_list[id_dim_x]}'] = func.selectBestNormErrorInParams(result,cfg.sigma_x_list ,cfg.sigma_y_list ,cfg.alpha_list)
    
            elif r_subsample==0.5:
                best_norm_error_list_subsample2[f'dim_x_{cfg.dim_x_list[id_dim_x]}'] = func.selectBestNormErrorInParams(result,cfg.sigma_x_list ,cfg.sigma_y_list ,cfg.alpha_list)
    
    # ベストErrorのプロット
    #output_pass = f'output/{datetime.now().strftime("%Y%m%d")}/{cfg.data_type}/r_subsample_0.1'
    plotter.plotDimVsError(best_norm_error_list=best_norm_error_list_subsample1, dim_x_list=cfg.dim_x_list, plot_target='best_RKHS_norm_error', output=output_pass_org)
    plotter.plotDimVsError(best_norm_error_list=best_norm_error_list_subsample1, dim_x_list=cfg.dim_x_list, plot_target='best_sup_norm_error', output=output_pass_org)


#output_pass = f'output/{datetime.now().strftime("%Y%m%d")}/{cfg.data_type}/r_subsample_0.5'
#plotter.plotDimVsError(best_norm_error_list=best_norm_error_list_subsample2, dim_x_list=cfg.dim_x_list, plot_target='best_RKHS_norm_error', output=output_pass)
#plotter.plotDimVsError(best_norm_error_list=best_norm_error_list_subsample2, dim_x_list=cfg.dim_x_list, plot_target='best_sup_norm_error', output=output_pass)
