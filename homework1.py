import numpy as np
# (1)	Create 1 dimension array with size 10.(random value)
np1 = np.random.randint(1,10,10,int)
# print(f'np1:{np1}')
# (2)	Test the basic option”+,-,*,/” for two array.
# np2 = np.random.randint(1,10,10,int)
# print(f'np2:{np2}')
# print(f'+:{np1+np2}')
# print(f'-:{np1-np2}')
# print(f'*:{np1*np2}')
# print(f'/:{np.trunc(np1/np2)}')
# # (3)	Try mean, max, min,median, argmax,etc...
# print(f'mean:{np.mean(np1)}')
# print(f'max:{np.max(np1)}')
# print(f'min:{np.min(np1)}')
# print(f'median:{np.median(np1)}')
# counts = np.bincount(np1) # 返回了一个长度为nums最大值的列表
# print(f'argmax:{np.argmax(counts)}')
# # (4)	Create 2 dimension array with size (4,6)
# np3 = np.random.randint(0,10,(4,6),int)
# print(np3)
# # (5)	Try transpose function for it.
# print(f'tran:\n{np3.transpose()}')


# import pandas as pd
# (2)	Print first 10 row record
# data = pd.read_csv('Salaries.csv')
# print(data.head(10))
# (3)	Count for the size of this dataset
# import os
# print(os.path.getsize('Salaries.csv'))
# (4)	Print the column name
# row1 = list(pd.read_csv('Salaries.csv',nrows=0))
# t = 1
# for i in row1:
#     print(f'{t}:{i}')
#     t += 1
# (5)	Calculate the mean value
# df_rank = data.groupby(['rank'])
# print(df_rank.mean())
# (6)	Group “rank”
# df_rank = data.groupby(['rank'])
# print(df_rank)
# (7)	Filter the data which for ”male” and find max salary
# print(data[ data['sex']=='Male'].groupby(['sex'], sort=False)[['salary']].max())
# (8)	Use slicing to find recod from 10 to 20
# print(data[10:20])
# (9)	Use loc to find record from 5-10 and show salary and rank
# print(data.loc[5:10,['salary','rank',]])
# (10)	Sort the data in ascending way for “salary” and show the dataset
# print(data.sort_values(by='salary').head())
# (11)	Filter missing value and show the dataset
# flights = pd.read_csv('Salaries.csv')
# print(flights[flights.isnull().any(axis=1)].head())
#
import matplotlib.pyplot as plt
# (1)	 Y = X^2
# x = np.linspace(-10,10,20)
# y = x**2
# plt.plot(x,y)
# plt.show()
# # (2)	Set x axis is month and y axis is the profit. Y = 1.05*X+5
# x = np.arange(1,13,1)
# y = x*1.05 + 5
# plt.xlabel('month')
# plt.ylabel('profit')
# plt.plot(x,y)
# plt.show()
# # (3)	Based on uniform distribution random generate 50 points and plot it. (np.random.uniform)
y = np.random.uniform(1,100,50)
x = np.arange(1,51,1)
plt.plot(x,y)
plt.show()