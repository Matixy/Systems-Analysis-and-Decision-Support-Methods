from abc import ABC, abstractmethod
import numpy as np

class ModelAbstract(ABC):
  def __init__(self):
    self.weights = None # wszystkie wagi
    self.intercept = None # wyraz wolny- b
    self.coef = None # wagi dla cech
    
  @abstractmethod
  def fit(self, *args, **kwargs):
    pass
  
  def _add_bias_column(self, X):
    """"Funkcja dodająca kolumne samych jedynek do danej macierzy (potrzebne do wyznaczenia parametru b)"""
    
    X_np = np.array(X) # konwersja na tablice numpy
    return np.c_[np.ones((len(X_np), 1)), X_np] # wypelnienie 1
  
  def predict(self, X):
    X_b = self._add_bias_column(X)
    return X_b @ self.weights