# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:18:31 2018

@author: zhengyang
"""
import pandas as pd

# series

s1 = pd.Series([1,2,3,4],index=['a','b','c','d'],name='demo')
s1[0]
s1['a']
s1[0:2]
s1['a':'c']

# 对齐运算
s2 = pd.Series([1,2,3],index=['a','b','c'])

s3 = pd.Series([1,2,3],index=['a','b','d'])

s0=s2+s3  #按索引对齐相加

#DataFrame
df1=pd.DataFrame({'close':[10,15,20],
                  'percent':[0.1,0.05,-0.05],
                  'vol':[1,2,3]},
                  index=['000001','000002','000003'])
df1.loc['000001','close']
df1.iloc[1,2]
df1.loc['000001':'000003','close']
df1.loc['000004']=[8,8,8]
# 筛选
df1[df1['close']<15]
df1[(df1['close']<15)&(df1['vol']<2)]

df2 = pd.DataFrame({'one':[1,3,1,4],
                   'two':[2,5,1,3],
                   'three':[1,2,3,4]},
                    index=['a','d','c','b'])
# 排序-按索引排序
df2.sort_index()
df2.sort_index(ascending=False)
# 排序-按列排序
df2.sort_values(by=['one'])
df2.sort_values(by=['one','two'])
# 统计描述
df2.mean(axis=0) # 按垂直方向操作，按水平方向是axis=1
df2.describe() # 全部描述
df2.cov() # 协方差矩阵

# 读写文件
df1.to_excel('result1.xlsx')
writer = pd.ExcelWriter('result.xlsx')
df1.to_excel(writer,sheet_name='df1')
df2.to_excel(writer,sheet_name='df2')
writer.save()

df3 = pd.read_excel('result.xlsx',sheetname='df1')




