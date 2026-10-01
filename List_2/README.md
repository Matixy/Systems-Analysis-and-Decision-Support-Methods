# Projekt: Klasyczne Algorytmy Uczenia Maszynowego (Implementation & Analysis)

## Opis projektu
Celem drugiej części projektu było przejście od ręcznie tworzonych reguł decyzyjnych do automatycznych algorytmów uczenia maszynowego. Należało zrozumieć matematycznyne fundamenty działania modeli poprzez samodzielną implementację od zera przy użyciu biblioteki NumPy oraz porównanie ich wyników z gotowymi rozwiązaniami z biblioteki scikit-learn. Projekt obejmuje analizę interpretowalności modeli, wpływu skalowania cech na proces uczenia oraz badanie zjawisk przeuczenia i niedouczenia.

## Wykorzystany zbiór danych
W projekcie kontynuowano pracę na zbiorze [**Telco Customer Churn**](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), dotyczący rezygnacji klientów z usług telekomunikacyjnych.
*   **Problem Klasyfikacji:** Analiza rezygnacji klientów przy użyciu Drzew Decyzyjnych (`Churn`: 0/1).
*   **Problem Regresji:** Przewidywanie wysokości miesięcznego rachunku klienta (`MonthlyCharges`) przy użyciu Regresji Liniowej.

## Główne cechy projektu
*   **Implementacja własna:**
    *   Regresja Liniowa: rozwiązanie analityczne (Metoda najmniejszych kwadratów / macierz pseudoodwrotna).
    *   Regresja Liniowa: rozwiązanie iteracyjne (Algorytm spadku gradientu).
    *   Funkcje obliczające entropię i zysk informacyjny do weryfikacji ważności cech.
*   **Analiza Drzew Decyzyjnych:** Wizualizacja wygenerowanych reguł `if/else` oraz porównanie ich z intuicją z poprzedniej listy.
*   **Skalowanie i Interpretacja:** Badanie wpływu `StandardScaler` na wagi modelu regresji liniowej i możliwość ich interpretacji.
*   **Bias-Variance Tradeoff:**
    *   Analiza krzywych złożoności modelu.
    *   Demonstracja problemu "Czarnego Łabędzia" (błąd ekstrapolacji w modelach przeuczonych).
*   **Ewaluacja:** Porównanie metryk (Accuracy, F1-Score, MSE, MAE) pomiędzy implementacjami własnymi a modelami z `scikit-learn`.

## Struktura projektu
Projekt utrzymuje modułową strukturę, rozszerzoną o własne klasy modeli:

```text
List_2/
├── data/                   # Pliki danych (WA_Fn-UseC_-Telco-Customer-Churn.csv)
├── docs/                   # Dokumentacja, treść listy oraz raporty (pdf/html)
├── notebooks/              # Główny raport w formacie Jupyter Notebook (report.ipynb)
├── src/                    # Kod źródłowy
│   ├── models/             # Implementacje modeli
│   │   ├── ModelAbstract.py              # Klasa bazowa dla modeli
│   │   ├── ModelLinearRegressionLeastSquares.py
│   │   └── ModelLinearRegressionGradientDescent.py
│   └── utils/              # Funkcje pomocnicze (preprocessing, info gain)
│       └── utils.py
├── requirements.txt        # Zależności biblioteczne
└── README.md               # Opis projektu
```

## Instalacja i uruchomienie
1. Skonfiguruj wirtualne środowisko:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```
2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install -r requirements.txt
   ```
3. Uruchom Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Otwórz plik `notebooks/report.ipynb`.

## Technologie
*   Python 3.x
*   **NumPy** (główna biblioteka do implementacji macierzowej modeli)
*   Pandas (manipulacja danymi)
*   Matplotlib & Seaborn (wizualizacja)
*   Scikit-learn (jako punkt odniesienia/benchmark oraz do preprocessingu)