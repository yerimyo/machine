import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Iris 데이터 로드 및 준비
# load_diabetes 대신 seaborn을 사용하여 iris 데이터셋을 로드합니다.
iris = sns.load_dataset('iris')

# 슬라이드에 제시된 대로 특성(X)과 타깃(Y)을 지정합니다.
# X: 꽃잎 길이 (petal_length)
# Y: 꽃받침 길이 (sepal_length)
x = iris['petal_length'].values
y = iris['sepal_length'].values

print("--- 1. 데이터 확인 ---")
print(f"특성(X - 꽃잎 길이) 샘플 5개:\n {x[:5]}")
print(f"타깃(Y - 꽃받침 길이) 샘플 5개:\n {y[:5]}")
print(f"데이터셋 크기 (샘플 수): {len(x)}개\n")

# 2. Neuron 클래스 정의 (경사 하강법 알고리즘)
class Neuron:
    """
    단일 뉴런을 이용한 선형 회귀 모델입니다.
    경사 하강법을 사용하여 가중치(w)와 절편(b)을 학습합니다.
    """
    def __init__(self):
        # 가중치와 절편을 초기화합니다.
        # 데이터의 스케일이 다르므로 초기값을 1.0으로 설정합니다.
        self.w = 1.0 
        self.b = 1.0 

    def forpass(self, x):
        """정방향 계산: 예측값 y_hat = w * x + b 계산"""
        y_hat = x * self.w + self.b
        return y_hat

    def backprop(self, x, err):
        """역방향 계산: 가중치와 절편에 대한 그래디언트(변화율) 계산"""
        w_grad = x * err  # w에 대한 그래디언트 (기울기)
        b_grad = 1 * err  # b에 대한 그래디언트 (절편)
        return w_grad, b_grad
    
    def fit(self, x, y, epochs=100):
        """모델을 훈련시키는 메서드 (경사 하강법)"""
        # w와 b 업데이트를 위한 학습률(learning rate)을 도입합니다.
        # Iris 데이터는 정규화되지 않았으므로 학습률을 작게 설정해야 수렴이 안정적입니다.
        learning_rate = 0.001 
        
        for i in range(epochs): # 에포크만큼 반복
            for x_i, y_i in zip(x, y): # 모든 샘플에 대해 반복
                y_hat = self.forpass(x_i) # 순전파
                err = -(y_i - y_hat) # 오차 계산: (y_hat - y_i)
                
                w_grad, b_grad = self.backprop(x_i, err) # 역전파 (그래디언트 계산)
                
                # 가중치와 절편 업데이트 (학습률 적용)
                self.w = self.w - w_grad * learning_rate 
                self.b = self.b - b_grad * learning_rate

        print(f"--- 2. 학습 결과 (Epochs: {epochs}, LR: {learning_rate}) ---")
        print(f'최종 w (기울기): {self.w:.4f}')
        print(f'최종 b (절편): {self.b:.4f}')

# 3. 모델 훈련 및 결과 시각화

# Neuron 객체 생성
neuron = Neuron()

# 훈련 시작 (에포크 100번으로 수정)
neuron.fit(x, y, epochs=100)

# 4. 학습 결과 시각화
plt.scatter(x, y) # 산점도만 간결하게 표시

# 학습된 직선 y = w * x + b 를 그리기 위해 x 범위 설정
# x 데이터의 최소/최대 범위는 대략 1.0에서 7.0 사이입니다.
x_range_min = np.min(x) - 0.5
x_range_max = np.max(x) + 0.5

# pt1과 pt2 계산: 학습된 모델의 시작점과 끝점
pt1 = (x_range_min, x_range_min * neuron.w + neuron.b)
pt2 = (x_range_max, x_range_max * neuron.w + neuron.b)

# 학습된 직선 그리기 (간결한 형태)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-')

plt.xlabel('x') # x축 레이블만 간단히
plt.ylabel('y') # y축 레이블만 간단히
plt.show() # 그래프 출력

# 5. 새로운 값 예측
x_new = 4.5 # 꽃잎 길이 4.5cm일 때
y_pred = neuron.forpass(x_new)

print(f"\n--- 3. 새로운 예측 ---")
print(f"꽃잎 길이 {x_new} cm에 대한 꽃받침 길이 예측값: {y_pred:.4f} cm")

# 예측 결과를 그래프에 표시 (간결한 형태)
plt.scatter(x, y)
plt.scatter(x_new, y_pred, color='red', marker='X', s=200) # 예측점 표시
plt.xlabel('x')
plt.ylabel('y')
plt.show()
