# 합성곱 신경망을 사용한 이미지 분류
# 패션 MNIST 데이터 불러오기

# import numpy as np
# x=[1,2,3,4,5]
# x=np.clip(x,2,4)
# print(x) # (1-1)
# indexes = np.random.permutation(np.arange(len(x)))
# print(np.sum(indexes)) # (1-2)


# import pandas as pd
# url="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
# AB=pd.read_csv(url, header=None, names=['Sex','Length','Diameter','Height','Whole','Shucked','Viscera','Shell','Rings'])
# x=AB['Shucked']
# y=AB['Viscera']

# class Neuron:    
#     def __init__(self,w=500):
#         self.w = w     # 가중치
#         self.b = 1.0     # 절편    
#     def forpass(self, x):
#         y_hat = x * self.w + self.b       
#         return y_hat    
#     def backprop(self, x, err):
#         w_grad = x * err    
#         b_grad = 1 * err    
#         return w_grad, b_grad
#     def fit(self, x, y, epochs=100):
#         for i in range(epochs):           
#             for x_i, y_i in zip(x, y):    
#                 y_hat = self.forpass(x_i) 
#                 err = -(y_i - y_hat)      
#                 w_grad, b_grad = self.backprop(x_i, err)  
#                 self.w = self.w - w_grad          
#                 self.b = self.b - b_grad          
# neuron = Neuron(w=0.5)
# neuron.fit(x, y)

# print(neuron.w, neuron.b)



import numpy as np
import seaborn as sns 
iris=sns.load_dataset('iris')
x=np.array(iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
y=iris['species']
y=np.where(y=="setosa",1,0) 
print(np.unique(y, return_counts=True))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, train_size=0.7)
print(np.unique(y_train, return_counts=True)) 
print(np.unique(y_test, return_counts=True))  

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
    def add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X] 
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))  
        return a
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])               
        self.b = 0                                 
        for i in range(epochs):                    
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:                      
                z = self.forpass(x[i])             
                a = self.activation(z)             
                err = -(y[i] - a)                  
                w_grad, b_grad = self.backprop(x[i], err) 
                self.w -= w_grad                   
                self.b -= b_grad                                   
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]     
        return np.array(z) > 0                   
    def score(self, x, y):
        return np.mean(self.predict(x) == y)

layer = SingleLayer()
layer.fit(x_train, y_train)
print(layer.score(x_train,y_train))
print(layer.score(x_test, y_test))


