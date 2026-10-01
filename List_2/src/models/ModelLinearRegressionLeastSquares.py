import numpy as np
from .ModelAbstract import ModelAbstract

class ModelLinearRegressionLeastSquares(ModelAbstract):
  def __init__(self):
    super().__init__()
    
  def fit(self, X, y):
    X_b = self._add_bias_column(X)
    y_np = np.array(y)
    
    # w = (X^T * X)^(-1) * X^T * y
    # bezpośrednia pseudoodwrotności Moore'a-Penrose'a na macierzy X_b omija to błędy powstające przy potęgowaniu macierzy (X^T * X) i rozwiazuje problem kolumn zależnych liniowo po one hot encodingu (powstają macierze osobliwe)
    theta = np.linalg.pinv(X_b) @ y_np
    
    self.weights = theta
    self.intercept = theta[0] # 1 elem to przesuniecie funkcji (param. b)
    self.coef = theta[1:] # wagi cech w1, w2, ...