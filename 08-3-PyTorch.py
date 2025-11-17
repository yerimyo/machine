import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

# 파이토치 버전
import torch.nn as nn
import torch

model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding='same'))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(2))
model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding='same'))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(2))
model.add_module('flatten', nn.Flatten())
model.add_module('dense1', nn.Linear(3136, 100))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(0.3))
model.add_module('dense2', nn.Linear(100, 10))

# best_cnn_model.pt에서ㅓ 가중치 로드하기
model.load_state_dict(torch.load('best_model.pt', weights_only=True),
                      strict=False)

# 파이토치 모델의 층을 참조하는 여러가지 방법
layers = [layers for layers in model.children()]

print(layers[0])

model[0]

# named_children() 매서드 사용
for name, layer in model.named_children():
    print(f"{name:10s}", layer)

model.conv1

# 평균, 표준편자 구하기
conv_weights = model.conv1.weight.data
print(conv_weights.mean(), conv_weights.std())

# 히스토그램으로 가중치 값 확인하기
import matplotlib.pyplot as plt

plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

print(conv_weights.shape)

# 가중치 그리기
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i,j].imshow(conv_weights[i*16 + j, 0,:,:], vmin=-0.5, vmax=0.5)
        axs[i,j].axis('off')
plt.show()

# 패션 MNIST 데이터 중에서 훈련 세트만 다운로드하기
from torchvision.datasets import FashionMNIST

fm_train = FashionMNIST(root='.', train=True, download=True)
train_input = fm_train.data

plt.imshow(train_input[0], cmap='gray_r')
plt.show()

ankle_boot = train_input[0:1].reshape(1, 1, 28, 28) / 255.0

# 첫 번째 컨볼루션 계층을 통과한 후 특성맵 형태 출력
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)
print(feature_maps.shape)

# 맷플롯립 사용해 그리기
fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0, i*8 + j, :, :])
        axs[i,j].axis('off')
plt.show()

# 합성곱 층, 렐루 함수, 풀링 층, 합성곱 층, 렐루 함수를 이어서 호출하기
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    feature_maps = model.relu1(feature_maps)
    feature_maps = model.pool1(feature_maps)
    feature_maps = model.conv2(feature_maps)
    feature_maps = model.relu2(feature_maps)

# 모델의 층 반복 호출하기
model.eval()
x = ankle_boot
with torch.no_grad():
    for name, layer in model.named_children():
        x = layer(x)
        if name == 'relu2':
            break
feature_maps = x

# 두 번째 합성곱 층이 만든 특성 맵 그리기
fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0, i*8 + j, :, :])
        axs[i,j].axis('off')
plt.show()