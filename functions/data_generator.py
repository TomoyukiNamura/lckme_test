#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import kernel_mean as km


#def generateMultivariateNormalRand(n_train,n_test,dim_x,dim_y,mean,cov):
#    # 引数チェック
#    if ((dim_x+dim_y)==np.array(mean).shape[0]==np.array(cov).shape[0]==np.array(cov).shape[1])==False:
#        print("error: dimension isn't matched")
#        return np.nan,np.nan
#    
#    # 乱数発生
#    rand = np.random.multivariate_normal(mean, cov, (1, n_train+n_test))[0]
#    
#    # 訓練データ作成
#    train = {}
#    tmp_train = rand[0:n_train,:]
#    train['x'] = tmp_train[:,0:dim_x].reshape(-1, dim_x)
#    train['y'] = tmp_train[:,dim_x:(dim_x+dim_y)].reshape(-1, dim_y)
#    
##    # 評価データ作成
##    test = {}
##    tmp_test  = rand[n_train:(n_train+n_test),:]
##    test['x'] = tmp_test[:,0:dim_x].reshape(-1, dim_x)
##    test['y'] = tmp_test[:,dim_x:(dim_x+dim_y)].reshape(-1, dim_y)
#    
#    # 評価データ作成(等間隔)
#    test = {}
#    tmp_max_x = max(train['x'])[0]
#    tmp_min_x = min(train['x'])[0]
#    test['x'] = np.arange(tmp_min_x, tmp_max_x, (tmp_max_x-tmp_min_x)/n_test).reshape(-1, dim_x)
#    test['y'] = cov[0][1]/cov[0][0]*test['x'].reshape(-1, dim_y)
#    
#    # return
#    return train,test
#
#

#def generateLinearModelData(n_train,n_test,dim_x,dim_y,mean,cov):
#    ## 多変量正規分布N(0,V)から乱数を発生
#    # 乱数発生
#    train,test = generateMultivariateNormalRand(n_train,n_test,dim_x,dim_y,mean,cov)
#    
#    # RKHSnorm計算用の値生成
#    mu_bar_list = []
#    for i in range(len(test['x'])):
#        mu_bar_list.append(cov[0][1]/cov[0][0]*test['x'][i][0])
#    mu_bar_list = np.array(mu_bar_list)
#    v_bar  = cov[1][1] - (cov[0][1]**2)/cov[0][0]
#    
#    # 描画用
#    truth_line = {}
#    
#    tmp_x_min = min(train['x'])[0]
#    tmp_x_max = max(train['x'])[0]
#    truth_line['x'] = np.arange(tmp_x_min,tmp_x_max,(tmp_x_max-tmp_x_min)/1000)
#    
#    truth_line['y'] = []
#    for i in range(len(truth_line['x'])):
#        truth_line['y'].append(cov[0][1]/cov[0][0]*truth_line['x'][i])
#        
#    truth_line['y'] = np.array(truth_line['y'])
#    
#    return train,test,mu_bar_list,v_bar,truth_line
    


def generateLinearModelData(n_train,n_test,dim_x,dim_y,var_eps):   
    # U(-1,1)の乱数を発生
    tmp_rand = (np.random.rand((n_train + n_test)*dim_x)*2-1).reshape(-1, dim_x)
    
    # f(x)+εのε~N(0,var_eps)部分を発生
    epsilon = np.random.normal(size=n_train + n_test, loc=0, scale=np.sqrt(var_eps)).reshape(-1, dim_y)
    
    # 訓練データ作成
    train = {}
    train['x'] = tmp_rand[0:n_train,:]
    coef_lm = np.append(np.array([0.0]),np.ones(dim_x))
    train['y'] = coef_lm[0] + np.dot(train['x'], coef_lm[1:(dim_x+1)].reshape(-1,1)).reshape(-1, dim_y)
    train['y'] = train['y'] + epsilon[0:n_train,:]
    
    # 評価データ作成
    test = {}
    test['x'] = tmp_rand[n_train:(n_train+n_test),:]
    test['y'] = coef_lm[0] + np.dot(test['x'], coef_lm[1:(dim_x+1)].reshape(-1,1)).reshape(-1, dim_y)
#    test['y'] = test['y'] + epsilon[n_train:(n_train+n_test),:]
    
#    #  評価データ作成 (等間隔)
#    test = {}
#    tmp_max_x = max(train['x'])[0]
#    tmp_min_x = min(train['x'])[0]
#    test['x'] = np.arange(tmp_min_x, tmp_max_x, (tmp_max_x-tmp_min_x)/n_test).reshape(-1, dim_x)
#    test['y'] = coef_lm[0] + np.dot(test['x'], coef_lm[1:(dim_x+1)].reshape(-1,1)).reshape(-1, dim_y)

    
    # 描画用
    if dim_x==1 and dim_y==1:
        truth_line = {}
        tmp_x_min = min(train['x'])[0]
        tmp_x_max = max(train['x'])[0]
        truth_line['x'] = np.arange(tmp_x_min,tmp_x_max,(tmp_x_max-tmp_x_min)/1000).reshape(-1, dim_x)
        truth_line['y'] = coef_lm[0] + np.dot(truth_line['x'], coef_lm[1:(dim_x+1)].reshape(-1,1)).reshape(-1, dim_y)
        truth_line['y'] = np.array(truth_line['y'])
        
    else:
        truth_line = None
    
    return train,test,truth_line



def generateNonLinearModelData(n_train,n_test,dim_x,dim_y,nc_nlm,sigma_u_nlm,var_eps):
    # U(-1,1)の乱数を発生
    u = (np.random.rand(nc_nlm*dim_x)*2-1).reshape(-1, dim_x)
    
    # 各uの重み定義(全て等しい重み)
    #weights_u = np.ones(u.shape[0])/u.shape[0]
    weights_u = (np.random.rand(nc_nlm)*2-1)
    weights_u = weights_u.reshape(1, -1)
    
    # Σγk(x,Ui)　部分の定義
    emb_u = km.KernelMean(X=u, weights=weights_u, kernel='rbf',gamma=1.0/sigma_u_nlm)
    
    
    # U(-1,1)の乱数を発生
    tmp_rand = (np.random.rand((n_train + n_test)*dim_x)*2-1).reshape(-1, dim_x)
    
    # f(x)+εのε部分を発生
    epsilon = np.random.normal(size=n_train + n_test, loc=0, scale=np.sqrt(var_eps)).reshape(-1, dim_y)
    
    # 訓練データ作成
    train = {}
    train['x'] = tmp_rand[0:n_train,:]
    train['y'] = np.dot(emb_u.weights, emb_u._get_kernel(emb_u.X,train['x'])).reshape(-1, dim_y)
    train['y'] = train['y'] + epsilon[0:n_train,:]
    
    # 評価データ作成
    test = {}
    test['x'] = tmp_rand[n_train:(n_train+n_test),:]
    test['y'] = np.dot(emb_u.weights, emb_u._get_kernel(emb_u.X,test['x'])).reshape(-1, dim_y)
#    test['y'] = test['y'] + epsilon[n_train:(n_train+n_test),:]
    
#    #  評価データ作成 (等間隔)
#    test = {}
#    tmp_max_x = max(train['x'])[0]
#    tmp_min_x = min(train['x'])[0]
#    test['x'] = np.arange(tmp_min_x, tmp_max_x, (tmp_max_x-tmp_min_x)/n_test).reshape(-1, dim_x)
#    test['y'] = np.dot(emb_u.weights, emb_u._get_kernel(emb_u.X,test['x'])).reshape(-1, dim_y)

    
    # 描画用
    if dim_x==1 and dim_y==1:
        truth_line = {}
        tmp_x_min = min(train['x'])[0]
        tmp_x_max = max(train['x'])[0]
        truth_line['x'] = np.arange(tmp_x_min,tmp_x_max,(tmp_x_max-tmp_x_min)/1000).reshape(-1, dim_x)
        truth_line['y'] = np.dot(emb_u.weights, emb_u._get_kernel(emb_u.X,truth_line['x'])).reshape(-1, dim_y)
    
    else:
        truth_line = None
        
    return train,test,truth_line
    