# 시그모이드함수가 만들어지는 과정
# 오즈비 실습
import matplotlib.pyplot as plt
import numpy as np
probs = np.arange(0, 1, 0.01)
print(probs)
odds = [p/(1-p) for p in probs]
plt.plot(probs, odds)
plt.xlabel('p')
plt.ylabel('p/(1-p)')
plt.show()

# 로짓함수 실습
import matplotlib.pyplot as plt
import numpy as np
probs = np.arange(0.001, 0.999, 0.001)
print(probs)
logit = [np.log(p/(1-p)) for p in probs]
plt.plot(probs, logit)
plt.xlabel('p')
plt.ylabel('log(p/(1-p))')
plt.show()

# 로지스틱함수 실습
import matplotlib.pyplot as plt
import numpy as np
zs = np.arange(-10., 10., 0.1)
print(zs)
gs = [1/(1+np.exp(-z)) for z in zs]
plt.plot(zs, gs)
plt.xlabel('z')
plt.ylabel('1/(1+e^-z)')
plt.show()