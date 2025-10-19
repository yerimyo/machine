# Neuron 클래스(class) 만들기(경사 하강법 알고리즘
class Neuron: 
    def __init__(self):
        self.w = 1.0 # 가중치를 초기화합니다
        self.b = 1.0 # 절편을 초기화합니다

    def forpass(self, x):
        y_hat = x * self.w + self.b # 직선 방정식을 계산합니다
        return y_hat

    def backprop(self, x, err):
        w_grad = x * err # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad # 2개의 결과를 반환 (return)
    
    def fit(self, x, y, epochs=100):
        for i in range(epochs): # 에포크만큼 반복합니다
            for x_i, y_i in zip(x, y): # 모든 샘플에 대해 반복합니다
                y_hat = self.forpass(x_i) # 정방향 계산
                err = -(y_i - y_hat) # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
                self.w = self.w - w_grad # 가중치 업데이트
                self.b = self.b - b_grad # 절편 업데이트
        print('w=',self.w, 'b=',self.b)

neuron = Neuron()
neuron.fit(x, y)

import matplotlib.pyplot as plt

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * neuron.w + neuron.b)
pt2 = (0.15, 0.15 * neuron.w + neuron.b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()