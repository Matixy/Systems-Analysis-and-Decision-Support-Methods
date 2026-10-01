import pandas as pd

def map_dtypes_str_to_float(df: pd.DataFrame, column_name: str) -> None:
    df[column_name] = pd.to_numeric(df[column_name])
    
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