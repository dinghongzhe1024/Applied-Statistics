'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ice = np.random.randint(20,100,12,int)
ice_sorted = sorted(ice)
# print(ice_sorted)
ice_1_6 = ice_sorted[::2]
# print(ice_1_6)
ice_7_12 = sorted(ice_sorted[1::2],reverse=True)
# print(ice_7_12)
ice_final = ice_1_6+ice_7_12
print(ice_final)
print(f'variance : {round(np.var(ice_final),2)}')
print(f'standard deviation : {round(np.sqrt(np.var(ice_final)),2)}')
data_set = np.random.randint(1,101,100,int)
plt.plot(data_set,'bo')
plt.show()
print(f'mean : {round(np.median(data_set),2)}')
counts = np.bincount(data_set)
print(f'mode : {round(np.argmax(counts),2)}')
print(f'median : {round(np.median(data_set),2)}')
print(np.split(data_set,5,axis=0))
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
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
