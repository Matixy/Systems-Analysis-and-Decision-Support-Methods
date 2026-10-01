import pandas as pd
from sklearn.metrics import accuracy_score
from typing import Any

def predict(tenure: Any, threshold: Any, *args, **kwargs) -> int: # 0,1
	"""
	Prosty system decyzyjny
	"""
 
	return 1 if tenure <= threshold else 0

def grid_search(x_train_column: pd.Series, y_train: pd.Series, min_range: int, max_range: int) -> int:
	best_accuracy: float = 0.0
	best_threshold: int = 0
  
	for threshold in range(min_range, max_range + 1):
		prediction = x_train_column.apply(lambda x: predict(tenure=x, threshold=threshold))

		accuracy: float = accuracy_score(y_train, prediction)

		if accuracy > best_accuracy:
			best_accuracy = accuracy
			best_threshold = threshold
  
	return best_threshold