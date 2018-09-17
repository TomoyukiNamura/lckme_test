#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

color_map = ['r','g','b','orange','m','y','c']

def plotTrainTest(train, test, truth_line):
    
    if train['x'].shape[1]>1:
        print('dim_x is not 1')
        return None
    
    if train['y'].shape[1]>1:
        print('dim_y is not 1')
        return None
    
    plt.scatter(train['x'], train['y'])
    plt.scatter(test['x'], test['y'])
    plt.plot(truth_line['x'], truth_line['y'],c='K')
    plt.grid()
    plt.show()
    
    
def plotModelResult(train,test,subsample_ID,model,model_name,result,sigma_x,sigma_y,alpha,output_pass,truth_line):
    # 予測曲線用のデータ作成
#    tmp_min = min(train['x'])
#    tmp_max = max(train['x'])
#    x_line = np.arange(tmp_min - 0.03*(tmp_max-tmp_min),
#                       tmp_max + 0.03*(tmp_max-tmp_min),
#                       0.01)
#    emb_y_line = model.predict(x_line)
    emb_y_line = model.predict(truth_line['x'])
    y_line = []
    for i in range(len(emb_y_line)):
        y_line.append(emb_y_line[i].weighted_sum())
    y_line = np.array(y_line)
    
    
    # plot & score
    plt.scatter(train['x'], train['y'],s=10)# 訓練データ
    
    # subsample プロット
    if model_name=='rss' or model_name=='nys':
        plt.scatter(train['x'][subsample_ID,:], train['y'][subsample_ID,:],s=15,c='orange')# 訓練データ
#    plt.scatter(test['x'], test['y'],s=20,c='orange')# 評価データ
#    plt.scatter(test['x'], result['truth_mean'],s=20,c='black')# 条件付き正規分布の平均(mode)
    plt.plot(truth_line['x'], y_line,c='r',linewidth = 2.5,  linestyle='solid')# 予測曲線の表示
    plt.plot(truth_line['x'], truth_line['y'],c='black',linewidth = 2.5,  linestyle='dashed')# 条件付き正規分布の平均
    #plt.scatter(test['x'], result[model_name]['pred_mean'],s=20,c='green')# 
    plt.grid()
    plt.title(f'{model_name} (σx={sigma_x},σy={sigma_y},ε={alpha})')
    plt.savefig(f'{output_pass}/plot_{model_name}_σx{sigma_x}_σy{sigma_y}_ε{alpha}.png')
    plt.show()
    
    
    
def heatmapResults(result,sigma_x_list,sigma_y_list,alpha_list,j,func,output_pass):
    # モデル名取得
    model_names = list(result[0][0][0].keys())
    
    # RKHSnormErrorに対応するsigma_xとalphaをまとめる
    vec_sigma_x = []
    vec_alpha = []
    
    for i in range(len(sigma_x_list)):
        for k in range(len(alpha_list)):
            vec_sigma_x.append(sigma_x_list[i])
            vec_alpha.append(alpha_list[k])
            
    vec_sigma_x = np.array(vec_sigma_x)
    vec_alpha   = np.array(vec_alpha)

    
    # モデルごとにRKHSnormErrorをまとめる
    for m in range(len(model_names)):
        tmp_list_RKHSnormError = []

        for i in range(len(sigma_x_list)):
            
            for k in range(len(alpha_list)):
                
                # RKHSnormErrorのarray取得
                tmp_array = result[i][j][k][model_names[m]]['RKHSnormError']
                
                if func=='mean':
                    # 平均
                    tmp_list_RKHSnormError.append(np.mean(tmp_array))
                    
                elif func=='max':
                    # 最大値
                    tmp_list_RKHSnormError.append(np.max(tmp_array))
                    
                else:
                    print('must choose func')
                    return None
                
        # RKHSnormErrorのdf作成
        df_RKHSnormError = pd.DataFrame({
                'σx':vec_sigma_x,
                'ε':vec_alpha,
                'RKHS norm error':np.array(tmp_list_RKHSnormError)
                })

        # RKHSnormErrorのピボットテーブル作成
        df_pivot = pd.pivot_table(data=df_RKHSnormError , values='RKHS norm error', 
                                          columns='σx', index='ε', aggfunc=np.mean)
            
        # ヒートマップ作成
        sns.heatmap(df_pivot, annot=True, fmt="1.2f", linewidths=.5, cmap="Reds",
                    vmin=0.0,vmax=3.0)
        plt.title(f"RKHS norm error ({model_names[m]})")
        plt.savefig(f'{output_pass}/heatmap_{model_names[m]}_RKHSnormError_{func}.png',bbox_inches="tight")
        plt.show()
        
        

def plot2dResults(result,sigma_x_list,sigma_y_list,alpha_list,xlabel,i,j,k,output_pass):
    
    # モデル名取得
    model_names = list(result[0][0][0].keys())
    
    ylim_RKHSnormError = [0.0,2.0]
#    ylim_truth_predmean_Error = [0.0,1.5]
#    ylim_testy_predmean_Error = [0.0,1.5]
    
    RKHSnormError = {}
    std_RKHSnormError = {}
#    truth_predmean_Error = {}
#    testy_predmean_Error = {}
    
    for m in range(len(model_names)):
        tmp_list_RKHSnormError = []
        std_tmp_list_RKHSnormError = []
#        tmp_list_truth_predmean_Error = []
#        tmp_list_testy_predmean_Error = []
        
        if xlabel == "σx":
            for i in range(len(sigma_x_list)):
                tmp_list_RKHSnormError.append(np.mean(result[i][j][k][model_names[m]]['RKHSnormError']))
                std_tmp_list_RKHSnormError.append(np.std(result[i][j][k][model_names[m]]['RKHSnormError']))
#                tmp_list_truth_predmean_Error.append(np.sqrt(np.mean(result[i][j][k][model_names[m]]['truth_predmean_Error'] ** 2 )))
#                tmp_list_testy_predmean_Error.append(np.sqrt(np.mean(result[i][j][k][model_names[m]]['testy_predmean_Error'] ** 2 )))
        
        elif xlabel == "σy":
            for j in range(len(sigma_y_list)):
                tmp_list_RKHSnormError.append(np.mean(result[i][j][k][model_names[m]]['RKHSnormError']))
                std_tmp_list_RKHSnormError.append(np.std(result[i][j][k][model_names[m]]['RKHSnormError']))
#                tmp_list_truth_predmean_Error.append(np.sqrt(np.mean(result[i][j][k][model_names[m]]['truth_predmean_Error'] ** 2 )))
#                tmp_list_testy_predmean_Error.append(np.sqrt(np.mean(result[i][j][k][model_names[m]]['testy_predmean_Error'] ** 2 )))
    
        elif xlabel == "ε":
            for k in range(len(alpha_list)):
                tmp_list_RKHSnormError.append(np.mean(result[i][j][k][model_names[m]]['RKHSnormError']))
                std_tmp_list_RKHSnormError.append(np.std(result[i][j][k][model_names[m]]['RKHSnormError']))
#                tmp_list_truth_predmean_Error.append(np.sqrt(np.mean(result[i][j][k][model_names[m]]['truth_predmean_Error'] ** 2 )))
#                tmp_list_testy_predmean_Error.append(np.sqrt(np.mean(result[i][j][k][model_names[m]]['testy_predmean_Error'] ** 2 )))
    
            
        RKHSnormError[model_names[m]] = np.array(tmp_list_RKHSnormError)
        std_RKHSnormError[model_names[m]] = np.array(std_tmp_list_RKHSnormError)
#        truth_predmean_Error[model_names[m]] = np.array(tmp_list_truth_predmean_Error)
#        testy_predmean_Error[model_names[m]] = np.array(tmp_list_testy_predmean_Error)
                
    
    # plot RKHSnormError
    for m in range(len(model_names)):
        # 平均値
#        plt.plot(np.log10(sigma_x_list), RKHSnormError[model_names[m]], 
#                 marker="o", label=model_names[m])
        #　標準偏差
        plt.errorbar(np.log10(sigma_x_list),RKHSnormError[model_names[m]],
                     yerr=std_RKHSnormError[model_names[m]],marker="o", 
                     color=color_map[m], ecolor=color_map[m], capsize = 4.0,
                     label=model_names[m],alpha=0.7)
    
    plt.grid() 
    plt.ylabel(f"RKHS norm error")
    plt.xlabel(f"log10({xlabel})")
    plt.legend()
#    plt.ylim(ylim_RKHSnormError)
    
    
    if xlabel == "σx":
        plt.title(f"RKHS norm error  (σy={sigma_y_list[j]} ε={alpha_list[k]})")
        plt.savefig(f'{output_pass}/RKHSnormError_{xlabel}_σy{sigma_y_list[j]}_ε{alpha_list[k]}.png')
        
    elif xlabel == "σy":
        plt.title(f"RKHS norm error  (σx={sigma_x_list[i]} ε={alpha_list[k]})")
        plt.savefig(f'{output_pass}/RKHSnormError_{xlabel}_σx{sigma_x_list[i]}_ε{alpha_list[k]}.png')
        
    elif xlabel == "ε":
        plt.title(f"RKHS norm error  (σx={sigma_x_list[i]} σy={sigma_y_list[j]})")
        plt.savefig(f'{output_pass}/RKHSnormError_{xlabel}_σx{sigma_x_list[i]}_σy{sigma_y_list[j]}.png')

    plt.show()
    
    
    
def doPlot2dResults(result,sigma_x_list,sigma_y_list,alpha_list,output_pass):
    # x軸:σx
    if len(sigma_x_list) > 1:
        i=0
        for j in range(len(sigma_y_list)):
            for k in range(len(alpha_list)):
                plot2dResults(result,sigma_x_list,sigma_y_list,alpha_list,
                                  xlabel="σx",i=i,j=j,k=k,output_pass=output_pass)
    
    # x軸:σy
    if len(sigma_y_list) > 1:
        j=0
        for i in range(len(sigma_x_list)):
            for k in range(len(alpha_list)):
                plot2dResults(result,sigma_x_list,sigma_y_list,alpha_list,
                                  xlabel="σy",i=i,j=j,k=k,output_pass=output_pass)
    
    # x軸:ε 
    if len(alpha_list) > 1:            
        k=0
        for i in range(len(sigma_x_list)):
            for j in range(len(sigma_y_list)):
                plot2dResults(result,sigma_x_list,sigma_y_list,alpha_list,
                                  xlabel="ε",i=i,j=j,k=k,output_pass=output_pass)
    