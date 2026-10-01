# Projekt: Ręczny System Decyzyjny (Rule-based System)

## Opis projektu
Celem pierwszej części projektu było zapoznanie się z analizą danych, statystyką opisową oraz fundamentalnymi koncepcjami podejmowania decyzji na podstawie danych. Zamiast od razu korzystać z gotowych algorytmów uczenia maszynowego, samodzielnie zbudowano od podstaw prosty, oparty na regułach system decyzyjny (tzw. podejście rule-based). Pozwoliło to zrozumieć, na czym polega proces poszukiwania zależności w danych, jak oceniać jakość logiki decyzyjnej oraz dlaczego podział danych na zbiór uczący i testowy jest absolutnie kluczowy w każdym problemie analitycznym.

## Wykorzystany zbiór danych
Do projektu wybrano zbiór [**Telco Customer Churn**](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), dotyczący rezygnacji klientów z usług telekomunikacyjnych.
*   **Problem Klasyfikacji:** Przewidywanie, czy klient zrezygnuje z usług (`Churn`: 0/1).
*   **Problem Regresji:** Przewidywanie wysokości miesięcznego rachunku klienta (`MonthlyCharges`).

## Główne cechy projektu
*   **Analiza EDA:** Szczegółowa wizualizacja cech numerycznych i kategorialnych (histogramy, boxploty, scatterploty).
*   **Inżynieria cech:** Mapowanie zmiennych binarnych oraz implementacja *One-Hot Encodingu* dla cech wieloklasowych.
*   **Automatyzacja:** Własna implementacja mechanizmu *Grid Search* do optymalizacji progów decyzyjnych w modelu prostym.
*   **Modelowanie:**
    *   Prosty model klasyfikacji (1 reguła).
    *   Złożony model klasyfikacji (5+ zagnieżdżonych reguł).
    *   Model regresyjny oparty na średnich wartościach grup treningowych.
*   **Ewaluacja:** Pomiar skuteczności na zbiorze testowym za pomocą metryk: Accuracy, MAE, MSE.

## Struktura projektu
Projekt został podzielony na moduły, oddzielając logikę przetwarzania i modele od warstwy raportowej:

```text
List_1/
├── data/                   # Pliki danych (.csv)
├── docs/                   # dokumentacja w tym raport z Jupyter Notebook w formacie html
├── src/                    # Kod źródłowy projektu
│   ├── utils/              # Narzędzia do czyszczenia i transformacji danych
│   └── models/             # Implementacja logiki modelów (reguły if/else)
├── notebooks/              # Raporty w formacie Jupyter Notebook
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
   Otwórz plik z raportem znajdujący się w folderze `notebooks/`.

## Technologie
*   Python 3.x
*   Pandas (przetwarzanie danych)
*   Matplotlib & Seaborn (wizualizacja)
*   Scikit-learn (wyłącznie do podziału danych i obliczenia metryk)