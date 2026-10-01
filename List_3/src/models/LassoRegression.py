import numpy as np

class LassoRegression:
  def __init__(self, alpha=0.1, learning_rate=0.01, epochs=1000):
    self.alpha = alpha # sila regularyzacji (kaganca)
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.weights = None
    self.intercept = None
    
  def fit(self, X, y):
    m, n = X.shape
    
    # wypelnienie wag zeramy
    self.weights = np.zeros(n)
    self.intercept = 0.0
    
    y_np = np.array(y)
    
    for _ in range(self.epochs):
      y_pred = X @ self.weights + self.intercept
      error = y_pred - y_np
      
      # obliczenie gradientu bledu
      dw = (2/m) * (X.T @ error)
      db = (2/m) * np.sum(error)
      
      # dodanie regularyzacji- kara = alpha * znak wagi (+1 lub -1)
      dw += self.alpha * np.sign(self.weights)
      
      self.weights -= self.learning_rate * dw
      self.intercept -= self.learning_rate * db
    
  def predict(self, X):
    return X @ self.weights + self.intercept