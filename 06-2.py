# k-평균

# KMeans 클래스
import os
os.system("wget https://bit.ly/fruits_300_data -O fruits_300.npy")

import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

print(km.labels_)

print(np.unique(km.labels_, return_counts=True))

import matplotlib.pyplot as plt
def draw_fruits(arr, ratio=1):
  n = len(arr)
  rows = int(np.ceil(n/10))
  cols = n if rows < 2 else 10
  fig, axs = plt.subplots(rows, cols,
                          figsize=(cols*ratio, rows*ratio), squeeze=False)
  for i in range(rows):
    for j in range(cols):
      if i*10 + j < n:
        axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
      axs[i, j].axis('off')
  plt.show()

print(draw_fruits(fruits[km.labels_==0]))

print(draw_fruits(fruits[km.labels_==1]))

print(draw_fruits(fruits[km.labels_==2]))

# 클러스터 중심
print(draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3))

print(km.transform(fruits_2d[100:101]))

print(km.predict(fruits_2d[100:101]))

print(draw_fruits(fruits[100:101]))

print(km.n_iter_)

# 최적의 k 찾기
inertia = []
for k in range(2, 7):
  km = KMeans(n_clusters=k, n_init='auto', random_state=42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()