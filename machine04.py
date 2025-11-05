class SingleLayer:
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []
    def forpass(self, x):
        z = np.sum(x * self.w) + self.b 
        return z
    def backprop(self, x, err):
        w_grad = x * err 
        b_grad = 1 * err 
        return w_grad, b_grad
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x))) # 인덱스 섞기
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
                self.w -= w_grad # 학습률(eta)은 1로 가정
                self.b -= b_grad 
                # 손실 계산을 위한 클리핑
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5 
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 로드 (URL에서 직접 불러오기)
import pandas as pd
url="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
AB=pd.read_csv(url, header=None,
names=['Sex','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings'])
print(AB.head())

X=AB[['Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings']]
y=AB[['Sex']]

import numpy as np
y_ab=np.where(y=="I",1,0)
print(np.unique(y_ab, return_counts=True)) # (array([0, 1]), array([2835, 1342]))

# NumPy 배열로 변환
X = X.values
y = y_ab

# 3. 훈련 세트와 테스트 세트 분리
x_train, x_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print("훈련 데이터 셋 크기:", x_train.shape)
print("테스트 데이터 셋 크기:", x_test.shape)


# 모델 구축
abalone_layer = SingleLayer()

# 훈련 (epochs=200으로 설정하여 더 안정적인 학습을 시도)
abalone_layer.fit(x_train, y_train, epochs=200)

# 모델 정확도 평가
test_score = abalone_layer.score(x_test, y_test)
print(f"\n--- Abalone 데이터 분류 결과 ---")
print(f"SingleLayer 모델 (로지스틱 회귀) 테스트 정확도: {test_score:.4f}")

# 손실 그래프 시각화 (학습이 잘 되는지 확인)
plt.plot(abalone_layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Abalone Classification Loss Curve')
plt.show()