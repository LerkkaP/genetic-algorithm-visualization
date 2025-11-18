import matplotlib.pyplot as plt
import numpy as np

def rastrigin(X: np.ndarray, A=10):
    n = X.shape[-1] 
    return A*n + np.sum(X**2 - A*np.cos(2*np.pi*X), axis=-1)  

x = np.linspace(-5.12, 5.12, 400)
X = x[:, np.newaxis]
y = rastrigin(X)

plt.plot(X, y)
plt.show()