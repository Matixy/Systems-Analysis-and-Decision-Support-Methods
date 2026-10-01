import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression, RidgeClassifier


class StackingModel:
  """
    Model sam dokonuje skalowania zbioru
    Modele L0: Ridge, Decision tree, knn
  """
  def __init__(self, max_depth=5, alpha=10, n_neighbors=5):
    self.max_depth = max_depth
    self.alpha = alpha
    self.n_neighbors = n_neighbors
    self.level_0_models = {}
    self.meta_model = None
    self.scaler = StandardScaler()
    
  def fit(self, X, y):
    X_np = np.array(X)
    y_np = np.array(y)
    
    #przeskalowanie danych dla knn i lasso
    X_scaled = self.scaler.fit_transform(X_np)
    
    model_tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
    model_ridge = RidgeClassifier(alpha=self.alpha, random_state=42)
    model_knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
    
    self.level_0_models = {
      "tree": model_tree,
      "ridge": model_ridge,
      "knn": model_knn
    }
    
    meta_X = np.zeros((X_scaled.shape[0], len(self.level_0_models)))
    
    # trenowanie modeli
    for i, (name, model) in enumerate(self.level_0_models.items()):
      # generacja predyckji przy uzyciu cross-validation
      meta_X[:, i] = cross_val_predict(model, X_scaled, y_np, cv=5)
      
      model.fit(X_scaled, y_np)
    
    # trenowanie meta modelu
    self.meta_model = LogisticRegression(class_weight='balanced', random_state=42)
    self.meta_model.fit(meta_X, y_np)
    
  def predict(self, X):
    X_np = np.array(X)
    X_scaled = self.scaler.transform(X_np)
    
    # modele dokonuja oceny zbioru
    meta_X = np.zeros((X_scaled.shape[0], len(self.level_0_models)))
    for i, (name, model) in enumerate(self.level_0_models.items()):      
      meta_X[:, i] = model.predict(X_scaled)
      
    # meta model dokonuje oceny wyborow modeli z poziomu 0    
    return self.meta_model.predict(meta_X)
  
  # getters & setters
  def get_level_0_models(self):
    return self.level_0_models
  
  def get_meta_model(self):
    return self.meta_model
  
  def get_scaler(self):
    return self.scaler