# 결정 트리

# 로지스틱 회귀로 와인 분류
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
print(wine.head())

print(wine.info())
print(wine.describe())

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)
print(train_input.shape, test_input.shape)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

## 설명하기 쉬운 모델과 어려운 모델
print(lr.coef_, lr.intercept_)