# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:05:20 2018

@author: zhengyang
"""

# -*- coding: utf-8 -*-
'''
Created on Tue Sep 4 13:09:38 2018
@author: zhengyang
'''

'''模块导入'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import datetime
# from sklearn import linear_model
from sklearn import decomposition  # decomposition函数，提供主成分分析功能(PCA)
from sklearn import svm  # svm支持向量机模块
from sklearn import metrics  # 矩阵处理函数
from sklearn.model_selection import train_test_split  # 拆分训练集函数

'''参数设置 define a class including all Parameters'''
class Para():
    method = 'SVM'                      # 'LR'或者'SGD'
    svm_kernel = 'linear'               # SVM的核函数类型，也可以：poly（多项式核），rbf(高斯核)
    svm_c = 0.01                        # 线性支持向量机的惩罚系数C
    month_in_sample = range(1,153+1)    # 样本内数据
    month_test = range(153,243+1)       # 样本外数据
    percent_select = [0.2,0.2]          # 正例和反例数据的比例
    percent_cv = 0.1                    # 试题比例
    seed = 45                           # 随机数种子点
    # logi_C = 0.0005                   # 置信度
    n_stock = 3625                      # 现今股票数
    n_stock_select = 10                 # 选出股票数
    parameters = ('EP','EPcut','BP','SP','NCFP','OCFP','DP','G/PE','Sales_G_q','Profit_G_q','OCF_G_q','ROE_G_q','ROE_q','ROE_ttm','ROA_q','ROA_ttm','grossprofitmargin_q','grossprofitmargin_ttm','profitmargin_q','profitmargin_ttm','assetturnover_q','assetturnover_ttm','operationcashflowratio_q','operationcashflowratio_ttm','financial_leverage','debtequityratio','cashratio','currentratio','ln_capital','HAlpha','return_1m','return_3m','return_6m','return_12m','wgt_return_1m','wgt_return_3m','wgt_return_6m','wgt_return_12m','exp_wgt_return_1m','exp_wgt_return_3m','exp_wgt_return_6m','exp_wgt_return_12m','std_FF3factor_1m','std_FF3factor_3m','std_FF3factor_6m','std_FF3factor_12m','std_1m','std_3m','std_6m','std_12m','ln_price','beta','turn_1m','turn_3m','turn_6m','turn_12m','bias_turn_1m','bias_turn_3m','bias_turn_6m','bias_turn_12m','rating_average','rating_change','rating_targetprice','holder_avgpctchange','macd','dea','dif','rsi','psy','bias')
                                        # 模型所使用的特征因子
para = Para()

'''清洗所有样本内csv中的数据，并对超额收益打标签，得到样本内数据集data_in_sample'''
for i_month in para.month_in_sample:
    
    data_curr_month = pd.read_csv('./data/csv_demo/'+str(i_month)+'.csv') 
    data_curr_month = data_curr_month.dropna()  # 删除数据有空值的股票
    data_curr_month['return_bin'] = np.nan      # 加入判定列，默认设为空
    data_curr_month = data_curr_month.sort_values(by='return',ascending=False) # 按return列排序，降序       
    
    n_stock = data_curr_month.shape[0]  # 样本内股票数
    n_stock_select = np.multiply(n_stock,para.percent_select)    #取样本前后比例的“显著”样本参与训练
    n_stock_select = np.around(n_stock_select)      # 取整
    n_stock_select = n_stock_select.astype(int)     # 转化为int
    
    data_curr_month.iloc[:n_stock_select[0],-1] = 1     # 打标签
    data_curr_month.iloc[-n_stock_select[0]:,-1] = 0    # 打标签
    data_curr_month = data_curr_month.dropna()          # 去除中间部分表现平平的股票
    
    if i_month==para.month_in_sample[0]:
        data_in_sample = data_curr_month
    else:
        data_in_sample = data_in_sample.append(data_curr_month)     # 矩阵拼接


'''将样本内数据集拆分：训练集+验证集，通过主成分分析模型去除相关变量'''
x_in_sample = data_in_sample.loc[:,para.parameters]
# 拆分所需的特征因子（列数据），定义为x
y_in_sample = data_in_sample.loc[:,'return_bin']
# 取return_bin为标签y
x_train, x_cv, y_train, y_cv = train_test_split(x_in_sample, y_in_sample,test_size = para.percent_cv,random_state = para.seed)
# 使用train_test_split函数进行数据拆分：训练集+验证集

pca = decomposition.PCA(n_components = 0.95)
# 设置模型：主成分分析模型，n_components取(0,1)之间的浮点数时，例如0.95，代表取累计方差贡献率大于0.95的主成分；n_component取>1的整数时，模型取相应数目的主成分
pca.fit(x_train)
# 对训练集拟合主成分分析模型
x_train = pca.transform(x_train)
x_cv = pca.transform(x_cv)
# 通过训练好的pca模型，对训练集合验证集的特征数组进行主成分分析转换，并用转换后的数组替代原数组

'''SVM模型训练'''
model = svm.SVC(kernel = para.svm_kernel, C = para.svm_c)
# 使用svm.SVC函数，赋予模型对象model，核函数为svm_kernel，惩罚系数为svm_c
model.fit(x_train,y_train)
# 模型训练
y_pred_train = model.predict(x_train)  # 训练集预测，得到二值化整数0或1，对应涨跌，用于模型评估
y_score_train = model.decision_function(x_train)  # 训练集预测，得到连续浮点数，对应涨跌的概率
y_pred_cv = model.predict(x_cv)  # 验证集预测，用于调参数
y_score_cv = model.decision_function(x_cv)  # 验证集预测



'''训练结果检验'''
y_pred_train = model.predict(x_train)           # 训练集拟合结果
metrics.accuracy_score(y_train,y_pred_train)    # 训练集拟合正确率
y_pred_cv = model.predict(x_cv)                 # 验证集拟合结果
metrics.accuracy_score(y_cv,y_pred_cv)          # 验证集拟合准确率


'''创建两张空表：1.记录测试月份预测收益率 2.记录每月的组合收益率'''
y_score_test = np.nan * np.zeros([para.n_stock,para.month_test[-1]])
# 创建空表，初始值为nan，记录验证集全部股票的预测收益，行为股票，列为月份
y_score_test = pd.DataFrame(y_score_test)
# 标准化
return_test = np.zeros([para.month_test[-1],1])
# 创建空表，初始值为0，记录每月前n_stock_select的股票的等权组合收益，行为月份，列为1列
return_test = pd.DataFrame(return_test)
# 标准化

'''清洗所有样本外csv中的数据，给出预测收益记录在y_score_test，给出组合收益记录在return_test'''
for i_month in para.month_test:
    data_curr_month = pd.read_csv('./data/csv_demo/'+str(i_month)+'.csv')
    y_true_curr_month = data_curr_month['return']   # 记录真实收益
    data_curr_month = data_curr_month.dropna()      # 剔除空值行
    x_curr_month = data_curr_month.loc[:,para.parameters]
    # 选取所需特征值
    y_pred_curr_month = model.predict(x_curr_month)
    # 预测标签：涨跌
    y_score_curr_month = model.predict_proba(x_curr_month)[:,1]
    # 预测收益率
    y_score_test.iloc[data_curr_month.index,i_month-1] = y_score_curr_month
    # 在y_score_test表中的当前月份保存预测收益
    y_score_curr_month = y_score_test.iloc[:,i_month-1]
    # 取出当前月份预测数据
    y_score_curr_month = y_score_curr_month.sort_values(ascending=False)
    # 按预测收益排序
    index_select = y_score_curr_month[0:para.n_stock_select].index
    # 记录当前月份前n_stock_select个股票的索引
    return_test.iloc[i_month-1,0] = np.mean(y_true_curr_month[index_select])
    # 在return_test表中记录组合收益率，组合为前n_stock_select个股票的等权组合

'''输出csv和plot'''
y_score_test.to_csv('PredResult_全部指标_LR_10组合.csv')  # 输出：预测收益矩阵到csv
value_test = (return_test+1).cumprod()  # 根据return_test计算复利，保存到表value_test
value_test.index = range(1,para.month_test[-1]+1)
value_test.to_csv('Value_全部指标_LR_10组合.csv')         # 输出：组合净值序列

plt.plot(para.month_test,value_test.loc[para.month_test],'r')
plt.xlabel('month')
plt.ylabel('value')
























