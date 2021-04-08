import numpy as np
import matplotlib.pyplot as plt
'''
A = np.arange(3,15)
print(A)
A = np.arange(3,15).reshape(3,4)
print(A)
print(A[2])
print(A[1,1])
print(A[1,1:3])
for row in A:
    print(row)

A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B)))

a = np.array([1,1,1,])
b = a.copy() # deepcopy
print(a,'\n',b)
b[1] = 998
print(a,'\n',b)

A = np.arange(12).reshape(3,4)
print(A)
print(np.split(A,2,axis=1)) # split(ary, indices_or_sections, axis=0) :把一个数组从左到右按顺序切分
print(np.split(A,3,axis=0)) # ary:要切分的数组 indices:如果是一个整数，就用该数平均切分 axis：默认为0，横向切分。为1时，纵向切分

data = {'apple':10,'orange':15,'lemon':5,'lime':20}
names = list(data.keys())
values = list(data.values())
fig,axs = plt.subplots(1,3,figsize=(9,3),sharey=True)
axs[0].bar(names,values)
axs[1].bar(names,values)
axs[2].bar(names,values)
fig.suptitle('Categorical Plotting')
plt.show()

np.random.seed(19680801)
mu = 100
sigma = 15
x = mu + sigma *np.random.randn(437)
num_bins = 50
fig , ax = plt.subplots()
n , bins , patches = ax.hist(x,num_bins,density=True , color = 'green')
y=((1/(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*(1/sigma*(bins-mu))**2))
ax.plot(bins,y,'--')
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\sigma=15$')
fig.tight_layout()
plt.show()
'''
'''
measures of central tendency 集中趋势的量度
集中趋势的度量是一种描述性统计数据，用于描述一组分数的平均值或典型值
共有三种集中趋势的常用度量： 
mode 众数 当出现两个众数的时候称为双峰 多个称为...
median 中位数
mean 平均数
中心倾向测度之间的关系
在对称分布中，中位数和均值相等 对于正态分布 平均数=中位数=众数 在正偏分布中，平均值大于中位数 在负偏分布中，均值小于中位数
变异量度
变异性度量（也称为数据传播）描述了一组观测值有多相似或多变。
有4种常见的可变性度量： 
Range 该范围描述了数据中最大和最小点之间的差异。 范围越大，数据越分散。 
IQR(interquartile range)  
Variance 平均方差 每个数据点和均值之间的差，对它们进行平方，求和，然后取这些数字的平均值来计算方差。
Standard deviation 平均方差的平方根
'''
'''
# 1.	Students and score:
import numpy as np
import matplotlib.pyplot as plt
dic = {}
dic_new = dic.fromkeys('zxcvbnmasdfghjk',0)
for key in dic_new:
    dic_new[key] = np.random.randint(50,100)
print(dic_new)
marks = [dic_new[key] for key in dic_new]
print(marks)
print(f'mean : {round(np.median(marks),2)}')
counts = np.bincount(marks)
print(f'mode : {round(np.argmax(counts),2)}')
print(f'median : {round(np.median(marks),2)}')
print(f'range : {np.max(marks)-np.min(marks)}')
print(f'IQR : {np.percentile(marks,4)}')
print(f'variance : {np.var(marks)}')
print(f'standard deviation : {np.sqrt(np.var(marks))}')
names = [key for key in dic_new]
plt.plot(names,marks)
plt.xlabel('names')
plt.ylabel('mark')
plt.title('Students and score')
plt.show()


ice = np.random.randint(20,100,12,int)
ice_sorted = sorted(ice)
print(ice_sorted)
ice_1_6 = ice_sorted[::2]
print(ice_1_6)
ice_7_12 = sorted(ice_sorted[1::2],reverse=True)
print(ice_7_12)
ice_final = ice_1_6+ice_7_12
print(ice_final)
print(f'variance : {round(np.var(ice_final),2)}')
print(f'standard deviation : {round(np.sqrt(np.var(ice_final)),2)}')



a The standard deviation represents the fluctuation of the data set

data_set = np.random.randint(1,101,100,int)
plt.plot(data_set,'bo')
plt.show()
print(f'mean : {round(np.median(data_set),2)}')
counts = np.bincount(data_set)
print(f'mode : {round(np.argmax(counts),2)}')
print(f'median : {round(np.median(data_set),2)}')
print(np.split(data_set,5,axis=0))
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
import pandas as pd
item = pd.DataFrame({'month':months,'ice_sale':ice_final})
print(item)
x = months
y = ice_final
fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
axs[0].bar(x, y)
axs[1].plot(x, y)
axs[0].set_xlabel('month')
axs[1].set_xlabel('month')
axs[0].set_ylabel('sale')
axs[1].set_ylabel('sale')
fig.suptitle('ice_sale')
plt.show()
'''