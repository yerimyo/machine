# 군집 알고리즘

# 과일 사진 데이터 준비하기
import os
os.system("wget https://bit.ly/fruits_300_data -O fruits_300.npy")

import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')

print(fruits.shape)

print(fruits[0, 0, :])

plt.imshow(fruits[0], cmap='gray')
plt.show()

plt.imshow(fruits[0], cmap='gray_r')
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()