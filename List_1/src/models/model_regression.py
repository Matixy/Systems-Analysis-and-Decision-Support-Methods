import pandas as pd

def prepare_calculated_means(
    mean_no_internet: float, 
    mean_fiber_premium: float, 
    mean_fiber_standard: float, 
    mean_fiber_basic: float, 
    mean_dsl_extras: float, 
    mean_dsl_basic: float) -> dict: 
    return {
        'no_internet': mean_no_internet,
        'fiber_premium': mean_fiber_premium,
        'fiber_standard': mean_fiber_standard,
        'fiber_basic': mean_fiber_basic,
        'dsl_extras': mean_dsl_extras,
        'dsl_basic': mean_dsl_basic
    }


def predict_regression(row: pd.Series, means_dict: dict, *args, **kwargs) -> float:
  """Model regresji, zwraca przewidywana kwote rachunku klienta na podstawie wyliczonych srednich dla danych grup w zbiorze"""
  
  # klienci bez internetu
  if row.get('InternetService_No', 0) == 1:
    return means_dict.get('no_internet', 0.0)
  
  # klienci z swiatlowodem- silna cecha w kontekscie miesiecznych oplat
  elif row.get('InternetService_Fiber optic', 0) == 1:
    tv = row.get('StreamingTV_Yes', 0)
    movies = row.get('StreamingMovies_Yes', 0)
    protection = row.get('DeviceProtection_Yes', 0)
    adds_sum = tv + movies + protection
    
    # sprawdzenie dodatkow
    if adds_sum == 3:
      return means_dict.get('fiber_premium', 0.0)
    elif adds_sum > 0:
      return means_dict.get('fiber_standard', 0.0)
    else:
      return means_dict.get('fiber_basic', 0.0)
    
  else:
    tv = row.get('StreamingTV_Yes', 0)
    movies = row.get('StreamingMovies_Yes', 0)
    protection = row.get('DeviceProtection_Yes', 0)
    
    # DSL z jakimikolwiek dodatkami
    if (tv + movies + protection) > 0:
        return means_dict.get('dsl_extras', 0.0)
    # Goły DSL
    else:
        return means_dict.get('dsl_basic', 0.0)