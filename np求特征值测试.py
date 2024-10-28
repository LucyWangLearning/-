# import numpy as np
#
# A = np.array([[4, -2, 3],
#               [1,  1, 1],
#               [3,  -2, 4]])
#
# eigenvalues, eigenvectors = np.linalg.eig(A)
#
# print("特征值:", eigenvalues)
# print("特征向量:\n", eigenvectors)
#
#
# n_dim = 2
#
# index = np.argsort(eigenvalues)[:n_dim]
# picked_vector = eigenvectors[:, index]
#
# print(index)  # 输出: [1, 2]
# print(picked_vector)

import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure()
ax=fig.add_subplot(111)
data = np.random.randn(50)
data2 = np.random.randn(50)
ax.plot(data,'rs--',label='data')
ax.plot(data2,'go--',label='data2')
ax.legend(loc = 'best')
plt.show()