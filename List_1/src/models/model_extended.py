import pandas as pd

def predict_extended(row: pd.Series) -> int:
    """
    Złożony system decyzyjny oparty na wnioskach analizy EDA
    """
    
    # 1 klienci z umowami na rok/dwa lata są lojalni
    if row.get('Contract_One year', 0) == 1 or row.get('Contract_Two year', 0) == 1:
        return 0
        
    # analiza klientow z umowami miesiecznymi
    
    # 2 krotki staz + wysokie oplaty
    if row['tenure'] <= 10 and row['MonthlyCharges'] > 70 :
        return 1
        
    # 3 swiatlowod + warunek malego stazu
    if row.get('InternetService_Fiber optic', 0) == 1:
        if row['tenure'] <= 20: 
            return 1
            
    # 4 brak opieki nad klientem + wysokie opłaty
    if row.get('TechSupport_Yes', 1) == 0 and row.get('OnlineSecurity_Yes', 1) == 0 and row.get('OnlineBackup_Yes', 1) == 0:  
        if row['MonthlyCharges'] > 70:
            return 1
                
    # 5 klient ma umowę miesieczna, ale nie spelnia powyzszych warunkow
    return 0