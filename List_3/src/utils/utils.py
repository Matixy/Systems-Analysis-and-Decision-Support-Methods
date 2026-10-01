import pandas as pd
import numpy as np

def map_dtypes_str_to_float(df: pd.DataFrame, column_name: str) -> None:
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce') # jezeli nie da sie przekonwertowac wartosci- NaN
    
    median: float = df[column_name].median()
    df[column_name] = df[column_name].fillna(median) 
    
def map_dtypes_str_to_bool(df: pd.DataFrame, column_name: str) -> None:
    df[column_name] = df[column_name].map({'Yes': 1, 'No': 0})

def apply_one_hot_encoding(x_train: pd.DataFrame, x_test: pd.DataFrame, columns_to_encode: list) -> tuple:
    """
    Funkcja stosujaca One-Hot encoding dla columns_to_encode
    zwraca zakodowane zbiory x_train i x_test z wartościami 0/1
    """
    
    # drop_first = True zapewnia usuniecie pierwszej kategori (problem wielokolinearności)
    # dtype = int zapewnia format 0/1
    x_train_encoded = pd.get_dummies(x_train, columns=columns_to_encode, drop_first=True, dtype=int)
    x_test_encoded = pd.get_dummies(x_test, columns=columns_to_encode, drop_first=True, dtype=int)
    
    # align zapewnia spojnosc danych wypelniajac nowe kolumny domyslna wartoscia 0
    x_train_encoded, x_test_encoded = x_train_encoded.align(x_test_encoded, join='left', axis=1, fill_value=0)
    
    return x_train_encoded, x_test_encoded

def entropy(x) -> float:
    # pusty wezel- entropia = 0
    if len(x) == 0:
        return 0.00
    
    # prawdopodobienstwo wystapienia kazdej z klas
    p = x.value_counts(normalize=True).values
    
    # obliczenie entropi - suma(p * log2(p)) + 1e-9 (bardzo mala liczba w celu unikniecia bledu log2(0))
    return -np.sum(p * np.log2(p + 1e-9))

def information_gain(parent, left, right) -> float:
    n = len(parent)
    n_l = len(left)
    n_r = len(right)
    
    #entropia przed podziałem
    parent_entropy = entropy(parent)
    
    #srednia wazona entropii po podziale
    childrens_entropy = (n_l / n) * entropy(left) + (n_r / n) * entropy(right)
    
    return parent_entropy - childrens_entropy