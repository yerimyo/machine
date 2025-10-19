my_list = [10, 'hello list', 20]
print(my_list[1])

my_list_2 = [[10, 20, 30], [40, 50, 60]]
print(my_list_2)
print(my_list_2[0])   # 첫 번째 행 출력
print(my_list_2[1])   # 두 번째 행 출력
print(my_list_2[1][1]) # 파이썬의 인덱스는 0부터 시작

import numpy as np
print(np.__version__) # 넘파이 버전 확인

my_arr = np.array([[10, 20, 30], [40, 50, 60]])
print(my_arr) # 2차원 배열 만들기

print(type(my_arr)) # 넘파이 배열인지 확인

print(my_arr[0][2]) # 특정 배열 요소 선택

print(np.sum(my_arr)) # 넘파이 내장함수 사용
print(my_arr.shape)

# 맷플롯립으로 그래프 그리기
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) # x 좌표와 y 좌표를 파이썬 리스트로 전달합니다.
plt.show() # 선 그래프

import matplotlib.pyplot as plt
plt.scatter([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
plt.show() # 산점도

import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(1000) # 표준 정규 분포를 따르는 난수 1,000개를 만듭니다.
y = np.random.randn(1000) # 표준 정규 분포를 따르는 난수 1,000개를 만듭니다.
plt.scatter(x, y)
plt.show() # 넘파이 배열로 산점도 그리기

# Example
import matplotlib.pyplot as plt
import numpy as np
X = np.random.poisson(5,100)
Y = 2*X+np.random.normal(3,2,100) # 평균=3, 표준편차=2
print(X)
plt.scatter(X, Y)
plt.show() # 넘파이 배열로 산점도 그리기