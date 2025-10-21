# 1. 라이브러리 임포트
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# 2. 유방암 데이터 세트 준비
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

# 훈련 세트와 테스트 세트 분리 (80:20 비율, stratify로 클래스 비율 유지)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.2, random_state=42
)

print("--- 데이터 분리 확인 ---")
print(f"훈련 데이터 크기: {x_train.shape}")
print(f"테스트 데이터 크기: {x_test.shape}")
print(f"훈련 세트 클래스 비율: {np.unique(y_train, return_counts=True)}")

# 3. 사이킷런 SGDClassifier 모델 설정
# loss='log_loss'를 지정하여 로지스틱 회귀(이진 크로스 엔트로피 손실 함수)로 설정
sgd = SGDClassifier(
    loss='log_loss', 
    max_iter=100,      # 최대 반복 횟수 (epochs)
    tol=1e-3,          # 수렴 허용 오차 (손실 감소량이 이 값 미만이면 중단)
    random_state=42    # 난수 초깃값 설정 (재현성 확보)
)

# 4. 모델 훈련 (fit)
# 사이킷런은 복잡한 경사 하강법 구현을 이 한 줄로 처리합니다.
print("\n--- 모델 훈련 시작 ---")
sgd.fit(x_train, y_train) 
print("모델 훈련 완료.")

# 5. 모델 예측 및 평가
# 테스트 세트에 대한 예측 수행
y_pred = sgd.predict(x_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred) # 또는 sgd.score(x_test, y_test) 사용 가능

print("\n--- 모델 평가 결과 ---")
print(f"테스트 세트 정확도: {accuracy:.4f}")

# 6. 일부 샘플에 대한 예측 결과 확인
print("\n--- 일부 예측 샘플 (앞 5개) ---")
print(f"예측 결과: {sgd.predict(x_test[:5])}")
print(f"실제 정답: {y_test[:5]}")

print(sgd.fit(x_train, y_train))
print(sgd.score(x_test, y_test))

print(sgd.predict(x_test[0:10]))