#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from functions import plotter
from functions import function as func
from sklearn import kernel_mean as km

from sklearn.kernel_mean import ConditionalKernelMean as ckm
from sklearn.kernel_mean import LocalConditionalKernelMean as lckm
from sklearn.kernel_mean import ApproximationConditionalKernelMean as rffckm
from sklearn.kernel_mean import NystroemConditionalKernelMean as nysckm
from sklearn.kernel_mean import DivideAndConquerCKME as dc


def initAndTrainModel(emb_X,emb_y,model_name,params):
    
    n_subsample = round(emb_X.X.shape[0]*params['r_subsample'])
    subsample_id = np.arange(0,n_subsample)
    
    if model_name=='ckm':
        # init model
        model = ckm(alpha=params['alpha'])
        # train model
        model.fit(emb_X, emb_y)
        
    
    elif model_name=='icf':
        # init model
        model = ckm(alpha=params['alpha'], method='ic', ic_tol=0.00001)
        # train model
        model.fit(emb_X, emb_y)
        
    
    elif model_name=='rss':
        # init model
        model = ckm(alpha=params['alpha'], method='rss')
        # train model
        model.fit(emb_X, emb_y, subsample_id=subsample_id)
        
        
    elif model_name=='nw':
        # init model
        model = ckm(alpha=params['alpha'], method = 'nw')
        # train model
        model.fit(emb_X, emb_y)
        

    elif model_name=='rff':
        # init model
        model = rffckm(alpha=params['alpha'], n_components=n_subsample, method='RBFSampler')
        # train model
        model.fit(emb_X, emb_y)


    elif model_name=='nys':
        # init model
        model = nysckm(alpha=params['alpha'], n_components=n_subsample)
        # train model
        model.fit(emb_X, emb_y, subsample_id=subsample_id)
        
        
    elif model_name=='dc':
        # init model
        model = dc(alpha=params['alpha'], n_components=n_subsample, n_weaklearners=params['n_weaklearner'])
        # train model
        model.fit(emb_X, emb_y)
        

    elif model_name=='fr':
        print(f'model name {model_name} is not found')
        return None


    elif model_name=='lckm':
        # init model
        model = lckm(alpha=params['alpha'], knn_k=n_subsample, random_sub_sampling=False)
        # train model
        model.fit(emb_X, emb_y)
        

    else:
        print(f'model name {model_name} is not found')
        return None
    
    
    return model,subsample_id




def doExperiment(train, test, experiment_params, sigma_x_list, sigma_y_list, alpha_list, mu_bar_list, v_bar, output_pass,truth_line):    

    # 結果保存場所    
    result = [[[0 for i3 in range(len(alpha_list))] for i2 in range(len(sigma_y_list))] for i1 in range(len(sigma_x_list))]

    # 実験
    for i in range(len(sigma_x_list)):
        for j in range(len(sigma_y_list)):
            for k in range(len(alpha_list)):
                
                print(f' start ==============================================')
                
                tmp_dim_x_tmp = train['x'].shape[1]
                print(f'dim_x : {tmp_dim_x_tmp}')
                print(f'sigma_x : {sigma_x_list[i]}')
                print(f'sigma_y : {sigma_y_list[j]}')
                print(f'alpha : {alpha_list[k]}')
                
                # パラメータを設定
                params = {}
                params['alpha']         = alpha_list[k]
                params['r_subsample']   = experiment_params['r_subsample']
                params['n_weaklearner'] = experiment_params['n_weaklearner']
                
                # 訓練データのカーネル平均作成
                emb_X = km.KernelMean(train['x'],kernel='rbf',gamma=1.0/sigma_x_list[i])
                emb_y = km.KernelMean(train['y'],kernel='rbf',gamma=1.0/sigma_y_list[j])             
                
                # 結果格納場所を定義
                tmp_result = {}
                
                # 評価実施
                for model_name in experiment_params['model_name_list']:
                    print(f'== {model_name} ==')
                    
                    # 初期化・学習
                    model,subsample_id = initAndTrainModel(emb_X=emb_X, emb_y=emb_y, model_name=model_name, params=params)
                    
                    # 予測
                    pred = model.predict(test['x'])
                    
                    # 結果を集計保存
                    tmp_result[model_name] = func.summarizeResult(pred, test, sigma_y_list[j], mu_bar_list, v_bar)
                    
                    # 結果をプロット
#                    plotter.plotModelResult(train,test,subsample_id,model,model_name,tmp_result[model_name],
#                                            sigma_x_list[i],sigma_y_list[j],alpha_list[k],truth_line,f"{output_pass}/{model_name}")  
                
                
                # 結果を保存
                result[i][j][k] = tmp_result
                
                print(f'finish ==============================================')

    return result