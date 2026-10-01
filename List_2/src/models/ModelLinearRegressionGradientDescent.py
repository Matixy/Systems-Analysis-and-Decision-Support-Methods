from .ModelAbstract import ModelAbstract
import numpy as np

class ModelLinearRegressionGradientDescent(ModelAbstract):
  def __init__(self):
    super().__init__()
    
  def fit(self, X, y, learning_rate = 0.0001, epochs=10000):
    X_b = self._add_bias_column(X)
    y_np = np.array(y)
    m, n = X_b.shape
    
    # inicjcacja wag
    self.weights = np.zeros(n)
    
    for _ in range(epochs):
      # predyckja z aktualnymi wagami
      y_pred = X_b @ self.weights
      error = y_pred - y_np
      
      # wyliczenie gradientu
      gradients = (2/m) * (X_b.T @ error)
      
      # aktualizacja wag
      self.weights -= learning_rate * gradients
      
    self.intercept = self.weights[0]
    self.coef = self.weights[1:]