import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingModel:
  def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.max_depth = max_depth
    self.trees = []
    self.initial_prediction = None
  
  def fit(self, X, y):
    X_np = np.array(X)
    y_np = np.array(y)
    
    # pczatek- wziecie sredniej
    self.initial_prediction = np.mean(y_np)
    
    # wypelnienie drzew na start kazde ma srednia
    current_predictions = np.full(shape=y_np.shape, fill_value=self.initial_prediction)
    
    # zerowanie drzew przed kazda nauka 
    self.trees = []
    
    for _ in range(self.n_estimators):
      # obliczenie blad (reszta / residuals) aktualnego modelu
      # r_i = Y_i - F_m-1 * X_i
      residuals = y_np - current_predictions
      
      # trenowanie nowego plytkiego drzewa
      tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
      tree.fit(X_np, residuals)
      self.trees.append(tree)
      
      # aktualizacja glownego modelu
      tree_predictions = tree.predict(X_np)
      current_predictions += self.learning_rate * tree_predictions
      
  def predict(self, X):
    X_np = np.array(X)
    
    predictions = np.full(shape=X_np.shape[0], fill_value=self.initial_prediction)
    
    # kazde drzewo dorzuca wyuczony blad poprzednika do wyniku
    for tree in self.trees:
      predictions += self.learning_rate * tree.predict(X_np)
    
    return predictions