from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

print(diabetes.data.shape, diabetes.target.shape)

print(diabetes.data[0:3])

print(diabetes.target[:3])

# 맷플롯립(matplotlib)의 scatter() 함수로 산점도 그리기
import matplotlib.pyplot as plt
plt.scatter(diabetes.data[:, 2], diabetes.target) # 입력데이터의 3번째 변수와 타깃데이터
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 훈련 데이터 준비하기
x = diabetes.data[:, 2] # 3번째 변수만 선택
y = diabetes.target

print(x[0:5])
print(y[0:5])

# w와 b 초기화
w = 1.0
b = 1.0

# 훈련데이터의 첫번째 데이터로 𝑦̂ 구하기
y_hat = x[0] * w + b
print(y_hat) # 예측값
print(y[0]) # 실제값

# w값을 조절하여 예측값 바꾸기
w_inc = w + 0.1
y_hat_inc = w_inc * x[0] + b
print(y_hat_inc) # y_hat_inc가 y_hat(1.0616962065186886 )보다 조금 증가

# w값 조정 후 예측값 증가 정도 확인
w_rate = (y_hat_inc - y_hat) / (w_inc - w)
print(w_rate)

print(x[0]) # x[0] 값과 동일

# 변화율이 양수(음수)일 때 가중치를 업데이트하는 방법
w_new = w + w_rate
print(w_new)

# 변화율로 절편 업데이트하기
b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc
print(y_hat_inc)

b_rate = (y_hat_inc - y_hat) / (b_inc - b)
print(b_rate)

b_new = b + 1
print(b_new)

# 오차(err)와 변화율(w_rate, 1)을 곱하여 가중치 업데이트
err = y[0] - y_hat
w_new = w + w_rate * err
b_new = b + 1 * err
print(w_new, b_new)

# 두 번째 샘플 x[1]을 이용
y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]
w_new = w_new + w_rate * err
b_new = b_new + 1 * err
print(w_new, b_new)

# 전체 샘플 이용
for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print(w, b)

# 산점도를 이용하여 단계3의 학습결과 확인
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 여러 에포크(epoch)를 반복 (100번의 에포크 반복)
for i in range(1, 100):
    for x_i, y_i in zip(x, y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err
print(w, b)

# 산점도를 이용하여 단계3의 학습결과 확인
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 모형으로 예측하기, x=0.18일 때 예측값 구하기
x_new = 0.18
y_pred = x_new * w + b
print(y_pred)

plt.scatter(x, y)
plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()