#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from cfg import config as cfg
from functions import plotter
from functions import function as func
from sklearn import kernel_mean as km

from sklearn.kernel_mean import ConditionalKernelMean as ckm
from sklearn.kernel_mean import LocalConditionalKernelMean as lckm
from sklearn.kernel_mean import ApproximationConditionalKernelMean as rffckm
from sklearn.kernel_mean import NystroemConditionalKernelMean as nysckm
from sklearn.kernel_mean import DivideAndConquerCKME as dc


def doTrainAndPred(model, model_name, train, test, subsample_ID, mu_bar_list, v_bar, sigma_x, sigma_y, alpha, output_pass,truth_line):
    # 予測
    emb_y_pred = model.predict(test['x'])
    
    ## 結果を集計保存
    result = func.summarizeResult(emb_y_pred, test, sigma_y, mu_bar_list, v_bar)
    
#            np.mean(tmp_result['ckm']['RKHSnormError'])
#            np.sqrt(np.mean(tmp_result['ckm']['truth_predmean_Error'] ** 2 ))
#            np.sqrt(np.mean(tmp_result['ckm']['testy_predmean_Error'] ** 2 ))
    
    # フォルダ作成
    output_folder_name = f"{output_pass}/{model_name}"
    func.makeNewFolder(output_folder_name)
    
    # 結果のプロット
    if train['x'].shape[1]==1:
        plotter.plotModelResult(train,test,subsample_ID,model,model_name,result,sigma_x,sigma_y,alpha,output_folder_name,truth_line)
    
    return result




def doExperiment(train, test, sigma_x_list, sigma_y_list, alpha_list, mu_bar_list, v_bar, output_pass,truth_line):
    # nys,rff,rss用のサブサンプルIDを抽出
    train_id = np.arange(0,train['x'].shape[0])
    n_subsample = round(train['x'].shape[0]*cfg.r_subsample)
    subsample_id = np.random.choice(train_id,n_subsample)

    # 結果保存場所    
    result = [[[0 for i3 in range(len(alpha_list))] for i2 in range(len(sigma_y_list))] for i1 in range(len(sigma_x_list))]

    # 実験
    for i in range(len(sigma_x_list)):
        for j in range(len(sigma_y_list)):
            for k in range(len(alpha_list)):
                
                print(f'===============================================')
                print(f'sigma_x : {sigma_x_list[i]}')
                print(f'sigma_y : {sigma_y_list[j]}')
                print(f'alpha : {alpha_list[k]}')
                
                # 結果格納場所を定義
                tmp_result = {}
                
                # 訓練データのカーネル平均作成
                emb_X = km.KernelMean(train['x'],kernel='rbf',gamma=1.0/sigma_x_list[i])
                emb_y = km.KernelMean(train['y'],kernel='rbf',gamma=1.0/sigma_y_list[j])
                                    
                
                ### 条件つきカーネル平均=======================================
                model_name = 'ckm'
                
                # 初期化
                ckm_model = ckm(alpha=alpha_list[k])
                
                # 学習・予測
                ckm_model.fit(emb_X, emb_y)
                tmp_result[model_name] = doTrainAndPred(model=ckm_model, model_name=model_name, train=train , test=test, subsample_ID=None, mu_bar_list=mu_bar_list, v_bar=v_bar, 
                          sigma_x=sigma_x_list[i], sigma_y=sigma_y_list[j], alpha=alpha_list[k], output_pass=f"{output_pass}",truth_line=truth_line)
                    


                
                ### 4.1 Incomplete Cholesky Factorization (ICF)=======================================
                model_name = 'icf'
                
                
                
                ### 4.2 Random Sub-sampling=======================================
                model_name = 'rss'
                                
                # モデル初期化
                rss_model = ckm(alpha=alpha_list[k], method='rss')
                
                # 学習・予測
                rss_model.fit(emb_X, emb_y, subsample_id=subsample_id)
                tmp_result[model_name] = doTrainAndPred(model=rss_model, model_name=model_name, train=train, test=test, subsample_ID=subsample_id, mu_bar_list=mu_bar_list, v_bar=v_bar, 
                          sigma_x=sigma_x_list[i], sigma_y=sigma_y_list[j], alpha=alpha_list[k], output_pass=f"{output_pass}",truth_line=truth_line)
                
               
                
                
                
                ### 4.3 Nadaraya-Watson kernel regression=======================================
                model_name = 'nw'
                
                # モデル初期化
                nw_model = ckm(alpha=alpha_list[k], method = 'nw')
                
                # 学習・予測
                nw_model.fit(emb_X, emb_y)
                tmp_result[model_name] = doTrainAndPred(model=nw_model, model_name=model_name, train=train, test=test, subsample_ID=subsample_id, mu_bar_list=mu_bar_list, v_bar=v_bar, 
                          sigma_x=sigma_x_list[i], sigma_y=sigma_y_list[j], alpha=alpha_list[k], output_pass=f"{output_pass}",truth_line=truth_line)
                
                
                
                
                ### 4.4 Random Fourier Features (RFF)=======================================
                model_name = 'rff'
                                
                # モデル初期化
                rffckm_model = rffckm(alpha=alpha_list[k],n_components=n_subsample, method='RBFSampler')
#                rffckm_model = rffckm(alpha=alpha_list[k],n_components=n_subsample, method='Nystroem')
                                                
                # 学習・予測
                rffckm_model.fit(emb_X, emb_y)
                tmp_result[model_name] = doTrainAndPred(model=rffckm_model, model_name=model_name, train=train, test=test, subsample_ID=subsample_id, mu_bar_list=mu_bar_list, v_bar=v_bar, 
                          sigma_x=sigma_x_list[i], sigma_y=sigma_y_list[j], alpha=alpha_list[k], output_pass=f"{output_pass}",truth_line=truth_line)
               
                
                
                
                ### Nystroem=======================================
                model_name = 'nys'
                                
                # モデル初期化
                nysckm_model = nysckm(alpha=alpha_list[k],n_components=n_subsample)
                
                # 学習・予測
                nysckm_model.fit(emb_X, emb_y, subsample_id=subsample_id)
                tmp_result[model_name] = doTrainAndPred(model=nysckm_model, model_name=model_name, train=train, test=test, subsample_ID=subsample_id, mu_bar_list=mu_bar_list, v_bar=v_bar, 
                          sigma_x=sigma_x_list[i], sigma_y=sigma_y_list[j], alpha=alpha_list[k], output_pass=f"{output_pass}",truth_line=truth_line)
               
              

                
                
                ### 4.5 Divide and Conquer CKME=======================================
                model_name = 'dc'
                
                # モデル初期化
                dc_model = dc(alpha=alpha_list[k],n_components=n_subsample,n_weaklearners=cfg.n_weaklearner)
#                dc_model = dc(alpha=1,n_components=100,n_weaklearners=10)

                # 学習・予測
                dc_model.fit(emb_X, emb_y)
                tmp_result[model_name] = doTrainAndPred(model=dc_model, model_name=model_name, train=train, test=test, subsample_ID=subsample_id, mu_bar_list=mu_bar_list, v_bar=v_bar, 
                          sigma_x=sigma_x_list[i], sigma_y=sigma_y_list[j], alpha=alpha_list[k], output_pass=f"{output_pass}",truth_line=truth_line)
               
                
                
                ### 4.6 Fast Randomized CKME
                model_name = 'fr'
                
                
                
                ### 【提案法】局所条件つきカーネル平均=======================================
                model_name = 'lckm'
                
                # モデル初期化
                lckm_model = lckm(alpha=alpha_list[k],knn_k=n_subsample,random_sub_sampling=False)
                
                # 学習・予測
                lckm_model.fit(emb_X, emb_y)
                tmp_result[model_name] = doTrainAndPred(model=lckm_model, model_name=model_name, train=train, test=test, subsample_ID=subsample_id, mu_bar_list=mu_bar_list, v_bar=v_bar, 
                          sigma_x=sigma_x_list[i], sigma_y=sigma_y_list[j], alpha=alpha_list[k], output_pass=f"{output_pass}",truth_line=truth_line)
               
             
                
                
                #### 結果の保存=======================================
                result[i][j][k] = tmp_result
                
                print(f'===============================================')

    return result