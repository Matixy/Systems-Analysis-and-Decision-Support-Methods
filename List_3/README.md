# Projekt: Klasyczne Algorytmy Uczenia Maszynowego (Implementation & Analysis)

## Opis projektu
Część III opiera się na zrozumieniu mechanizmów ochrony przed zjawiskami przeuczania (Overfitting) i "Czarnego Łabędzia". W liście opracowano m.in. procesy regularyzacji, koncepcji modeli zespołowych, baggingu, kaskadowy stacking, sekwencyjny boosting oraz archtekture Mixture of Expert (MoE).


## Wykorzystany zbiór danych
W projekcie kontynuowano pracę na zbiorze [**Telco Customer Churn**](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), dotyczący rezygnacji klientów z usług telekomunikacyjnych.
*   **Problem Klasyfikacji:** Analiza rezygnacji klientów przy użyciu Drzew Decyzyjnych (`Churn`: 0/1).
*   **Problem Regresji:** Przewidywanie wysokości miesięcznego rachunku klienta (`MonthlyCharges`) przy użyciu Regresji Liniowej.

## Główne cechy projektu

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
