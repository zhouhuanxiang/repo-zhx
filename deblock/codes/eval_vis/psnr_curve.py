import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

x = np.arange('2017-08-01','2017-08-10',dtype=np.datetime64)
y = np.random.randint(10,100,size=9)
y2 = np.random.randint(10,100,size=9)


plt.plot(x,y,color='red',label='APP')
plt.plot(x,y2,color='blue',label='PC')

plt.title(u'aaaa')
plt.xlabel(u'bbb')
plt.ylabel(u'cc')

plt.legend()

plt.show()