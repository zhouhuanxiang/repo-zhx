import numpy as np
from numpy import loadtxt
import seaborn as sns
import matplotlib.pyplot as plt

psnrAll_result = loadtxt('/Users/zhouhuanxiang/Desktop/Vimeo-meta/EDVR_GCB/psnrAll.txt', comments='#', delimiter='\n', unpack=False)
ssimAll_result = loadtxt('/Users/zhouhuanxiang/Desktop/Vimeo-meta/EDVR_GCB/ssimAll.txt', comments='#', delimiter='\n', unpack=False)
psnrAll_input = loadtxt('/Users/zhouhuanxiang/Desktop/Vimeo-meta/blocky37/psnrAll.txt', comments='#', delimiter='\n', unpack=False)
ssimAll_input = loadtxt('/Users/zhouhuanxiang/Desktop/Vimeo-meta/blocky37/ssimAll.txt', comments='#', delimiter='\n', unpack=False)

# sns.distplot(ssimAll_result)
# sns.distplot(ssimAll_input)
sns.distplot(ssimAll_result - ssimAll_input)

plt.tight_layout()
plt.show()



# print(' mean PSNR ', np.mean(psnrAll), ' mean SSIM ', np.mean(ssimAll), '\n')

# sep_testlist = loadtxt('/Users/zhouhuanxiang/Desktop/Vimeo-meta/sep_testlist.txt', dtype=str)
# psnrAll = list(zip(psnrAll, sep_testlist))
# ssimAll = list(zip(ssimAll, sep_testlist))
# # print(psnrAll[:10])
# # print(ssimAll[:10], '\n')

# psnrAll = sorted(psnrAll, key=lambda tup: tup[0])
# ssimAll = sorted(ssimAll, key=lambda tup: tup[0])
# # print(psnrAll[:10])
# # print(ssimAll[:10], '\n')



