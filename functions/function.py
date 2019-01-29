#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.stats import norm
from sklearn import kernel_mean as km


def makeNewFolder(folder_name):
    if os.path.exists(folder_name)==False:
        os.mkdir(folder_name)
        return True
    else:
        return False

def removeElementFromList(target_list, element):
    if element in target_list:
        target_list.remove(element)
        
def ckmeOfNorm1dim(y,mu_bar,v_bar,sigma_y):
    return norm.pdf(x=y, loc=mu_bar, scale=np.sqrt(v_bar+sigma_y))

def innerProductWithTrue(ckme_est,mu_bar,v_bar,sigma_y):
    return np.dot(ckme_est.weights, ckmeOfNorm1dim(ckme_est.X,mu_bar,v_bar,sigma_y))


def RKHSdistWithTrue(ckme_est, mu_bar, v_bar, sigma_y):    
    first_term  = norm.pdf(x=0, loc=0, scale=np.sqrt(2*v_bar+sigma_y))    
    second_term = -2 * innerProductWithTrue(ckme_est,mu_bar,v_bar,sigma_y)
    third_term  = km.innerProduct(ckme_est,ckme_est)
    
    return first_term+second_term+third_term

def summarizeResult(emb_y_pred,test,sigma_y,mu_bar_list,v_bar):
    result = {}
    
    truth_mean = []
    RKHSnormError = []
    pred_mean = []
    pred_mode = []
    
    for i in range(len(emb_y_pred)):
        # ground truth　の 平均値
        #truth_mean.append(cov[0][1]/cov[0][0]*test['x'][i])
        
        # RKHSnorm error
        #RKHSnormError.append(np.sqrt(RKHSdistWithTrue(emb_y_pred[i],test['x'][i],cov,sigma_y).reshape(1)))
        RKHSnormError.append(np.sqrt(RKHSdistWithTrue(emb_y_pred[i],mu_bar_list[i],v_bar,sigma_y).reshape(1)))
        
        # 予測値(mean)
        pred_mean.append(emb_y_pred[i].weighted_sum())
        
        # 予測値(mode)
        pred_mode.append(np.nan)    
        
    #truth_mean = np.array(truth_mean)
    truth_mean = mu_bar_list
    RKHSnormError = np.array(RKHSnormError)
    pred_mean = np.array(pred_mean)
    pred_mode = np.array(pred_mode)
    
    #result['truth_mean'] = truth_mean
    #result['pred_mean'] = pred_mean
    #result['pred_mode'] = pred_mode
    
    result['RKHSnormError']    = RKHSnormError
    #result['truth_predmean_Error'] = truth_mean - pred_mean
    #result['testy_predmean_Error'] = test['y']  - pred_mean
    #result['truth_predmode_Error'] = truth_mean - pred_mode
    #result['testy_predmode_Error'] = test['y']  - pred_mode
    
    result['mean_RKHSnormError'] = np.mean(RKHSnormError)
    result['max_RKHSnormError']  = np.max(RKHSnormError)
    
    return result


def selectBestNormErrorInParams(result,sigma_x_list,sigma_y_list,alpha_list):
    # 結果格納先
    best_norm_error_list = {}
    
    # モデル名取得
    model_name_list = list(result[0][0][0].keys())
        
    for model_name in model_name_list:
        best_norm_error_list[model_name] = {}
        
        tmp_list_mean_RKHSnormError = []
        tmp_list_max_RKHSnormError = []
    
        for id_sigma_x in range(len(sigma_x_list)):
            
            for id_sigma_y in range(len(sigma_y_list)):
                
                for id_alpha in range(len(alpha_list)):
                    
                    tmp_list_mean_RKHSnormError.append(result[id_sigma_x][id_sigma_y][id_alpha][model_name]['mean_RKHSnormError'])
                    tmp_list_max_RKHSnormError.append(result[id_sigma_x][id_sigma_y][id_alpha][model_name]['max_RKHSnormError'])
        
        # nanを除外
        tmp_list_mean_RKHSnormError = [x for x in tmp_list_mean_RKHSnormError if np.isnan(x)==False]
        tmp_list_max_RKHSnormError  = [x for x in tmp_list_max_RKHSnormError if np.isnan(x)==False]

        # 最良の結果を保存
        best_norm_error_list[model_name]['best_RKHS_norm_error'] = min(tmp_list_mean_RKHSnormError)
        best_norm_error_list[model_name]['best_sup_norm_error'] = min(tmp_list_max_RKHSnormError)

    return best_norm_error_list