# 유방암 데이터 세트 준비
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.data.shape, cancer.target.shape)

print(cancer.data[:3]) # 처음 3개의 샘플
print(cancer.target[:3])

# Boxplot으로 데이터 관찰
import matplotlib.pyplot as plt
plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()

cancer.feature_names[[3,13,23]] # Python은 index가 0부터 시작

import numpy as np
print(np.unique(cancer.target, return_counts=True))

# 훈련 데이터 세트 저장
x = cancer.data
y = cancer.target
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)
print(np.unique(y_train, return_counts=True)) # 양성(1)이 음성(0) 클래스보다 1.7 배 정도 많음

# 로지스틱 회귀 구현
import numpy as np
a=np.array([1,2,3])
b=np.array([3,4,5])
print(a) # [1 2 3]
print(b) # [3 4 5]
print(a+b) # [4 6 8]
print(a*b) # [ 3 8 15]
print(np.sum(a*b)) # 26 = 3+8+15

class LogisticNeuron: 
    def __init__(self): # 가중치와 절편은 미리 초기화 하지 않음
        self.w = None
        self.b = None
    def forpass(self, x):
        z = np.sum(x * self.w) + self.b # 직선 방정식을 계산, x와 w는 1차원 numpy 배열, 배열의 요소끼리 자동계산
        return z
    def backprop(self, x, err):
        w_grad = x * err # 가중치에 대한 그래디언트 계산
        b_grad = 1 * err # 절편에 대한 그래디언트 계산
        return w_grad, b_grad
    def activation(self, z):
        a = 1 / (1 + np.exp(-z)) # 시그모이드 계산
        return a
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1]) # 가중치를 1로 초기화, np.ones( ) 함수는 지정한 크기의 배열을 만들고 모두 1로 채움
        self.b = 0 # 절편을 0으로 초기화
        for i in range(epochs): # epochs만큼 반복
            for x_i, y_i in zip(x, y): # 모든 샘플에 대해 반복
                z = self.forpass(x_i) # 정방향 계산
                a = self.activation(z) # 활성화 함수 적용
                err = -(y_i - a) # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
                self.w -= w_grad # 가중치 업데이트 self.w = self.w - w_grad
                self.b -= b_grad # 절편 업데이트 self.b = self.b - b_grad
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x] # 정방향 계산
        a = self.activation(np.array(z)) # 활성화 함수 적용
        return a > 0.5 # 계단 함수 적용 
neuron = LogisticNeuron()
neuron.fit(x_train, y_train)
print(np.mean(neuron.predict(x_test) == y_test))

import numpy as np
print(np.zeros((2,3)))
print(np.full((2,3),7))

# 단층신경망
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

class SingleLayer:
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []
    def forpass(self, x):
        z = np.sum(x * self.w) + self.b # 직선 방정식을 계산
        return z
    def backprop(self, x, err):
        w_grad = x * err # 가중치에 대한 그레이디언트를 계산
        b_grad = 1 * err # 절편에 대한 그레이디언트를 계산
        return w_grad, b_grad
    def add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X] # 행렬의 맨 앞에 1로 채워진 열 벡터를 추가
    def activation(self, z):
        a = 1 / (1 + np.exp(-z)) # 시그모이드 계산
        return a
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1]) # 가중치를 초기화
        self.b = 0 # 절편을 초기화
        for i in range(epochs): # epochs만큼 반복
            loss = 0
            # 인덱스를 섞습니다
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes: # 모든 샘플에 대해 반복
                z = self.forpass(x[i]) # 정방향 계산
                a = self.activation(z) # 활성화 함수 적용
                err = -(y[i] - a) # 오차 계산
                w_grad, b_grad = self.backprop(x[i], err) # 역방향 계산
                self.w -= w_grad # 가중치 업데이트
                self.b -= b_grad # 절편 업데이트
                # 안전한 로그 계산을 위하여 클리핑한 후 손실을 누적함
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            # 에포크마다 평균 손실을 저장
            self.losses.append(loss/len(y))
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x] # 정방향 계산
        return np.array(z) > 0 # 스텝 함수 적용
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    
layer = SingleLayer()
layer.fit(x_train, y_train)
print(layer.score(x_test, y_test))

plt.plot(layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()