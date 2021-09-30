import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

tsne = np.load('data/tsne.npy')
labels =  np.load('data/labels.npy')

for i in range(2500):
    labels[i] = 1
for i in range(2500,5000):
    labels[i] = 0
x_0 = np.array([])
x_1 = np.array([])
y_0 = np.array([])
y_1 = np.array([])

for i in range(5000):
    if labels[i] == 0:
        x_0 = np.append(x_0,tsne[i][0])
        y_0 = np.append(y_0,tsne[i][1])
    else:
        x_1 = np.append(x_1,tsne[i][0])
        y_1 = np.append(y_1,tsne[i][1])

print(x_0.shape,x_1.shape)
plt.scatter(x_0,y_0,c='red')
plt.scatter(x_1,y_1,c='blue')
plt.show()

tsne = None
labels = None