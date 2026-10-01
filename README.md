# 🧠 Systems Analysis and Decision Support Methods 
*(Metody systemowe i decyzyjne)*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/NumPy-Data_Manipulation-013243?logo=numpy)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?logo=scikit-learn)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter)]()

## 📖 O projekcie
To repozytorium zawiera kompleksowy zbiór projektów laboratoryjnych zrealizowanych w ramach przedmiotu **Metody systemowe i decyzyjne** w trakcie **semestru letniego 2025/2026**. Projekt przedstawia kompletną ścieżkę od podstawowej analityki danych (EDA) i systemów eksperckich, aż po zaawansowane algorytmy uczenia maszynowego (Machine Learning) i nowoczesne modele zespołowe.

## 📂 Struktura repozytorium i zrealizowane zagadnienia

Projekt został podzielony na 3 główne etapy, z których każdy bazuje na wiedzy zdobytej w poprzednim:

### 🔹 [List 1: Podstawy Analizy Danych i Systemy Regułowe (Rule-Based)](./List_1)
Pierwszy etap skupia się na zrozumieniu danych biznesowych (np. Customer Churn, koszty ubezpieczeń, diagnoza medyczna) i podejmowaniu decyzji w oparciu o analitykę.
* **Eksploracyjna Analiza Danych (EDA):** Analiza rozkładów, poszukiwanie korelacji i transformacja danych kategorialnych.
* **System decyzyjny zoptymalizowany analitycznie:** Budowa prostych klasyfikatorów opartych na zagnieżdżonych instrukcjach `if/else`, wyposażonych we własny mechanizm *grid search* do poszukiwania optymalnych progów odcięcia.
* **Fundamenty ML:** Prawidłowy podział zbiorów na treningowy i testowy (unikanie *Data Leakage*) oraz weryfikacja logiki na podstawie metryk (Accuracy, MSE, MAE).

### 🔹 [List 2: Klasyczne Modele ML od Zera i Bias-Variance Tradeoff](./List_2)
Przejście od ręcznych reguł do algorytmów optymalizacyjnych automatyzujących proces decyzyjny.
* **Regresja Liniowa (Od zera):** Implementacja uczenia na dwa sposoby:
  * *Rozwiązanie analityczne* (Metoda Najmniejszych Kwadratów / Macierz Pseudoodwrotna).
  * *Rozwiązanie iteracyjne* (Własny algorytm Spadku Gradientu - Gradient Descent).
* **Drzewa Decyzyjne:** Interpretacja Entropii oraz wyliczanie Zysku Informacyjnego (*Information Gain*) do udowodnienia ważności cech (*Feature Importance*).
* **Złożoność Modeli:** Skalowanie danych (StandardScaler) oraz wizualizacja i symulacja zjawisk niedouczenia (*Underfitting*) i przeuczenia (*Overfitting*).
* **Problem "Czarnego Łabędzia":** Badanie zachowania przeuczonych wielomianów wysokiego stopnia w zderzeniu z anomaliami i błędami ekstrapolacji.

### 🔹 [List 3: Modele Zespołowe (Ensemble), Regularyzacja i MoE](./List_3)
Zaawansowane techniki ochrony modeli przed przeuczeniem oraz implementacja najpotężniejszych architektur ML.
* **Regularyzacja (L1/L2):** Modyfikacja własnego Spadku Gradientu o kary regularyzacyjne Lasso (selekcja cech) oraz Ridge.
* **Bagging od zera:** Autorska implementacja algorytmu przypominającego *Random Forest*, obejmująca próbkowanie losowe ze zwracaniem (*Bootstrap*) i agregację głosów (*Voting*).
* **Stacking & Boosting:** Łączenie heterogenicznych modeli (np. regresja zwalidowana z KNN i Drzewem) przy użyciu meta-modeli oraz minimalizacja obciążenia (Bias) algorytmami sekwencyjnymi.
* **Mixture of Experts (MoE):** Zaprojektowanie uproszczonej architektury (dziś napędzającej gigantów takich jak GPT-4), wykorzystującej algorytm klasteryzacji (*K-Means*) jako "sieć-bramkę" (Router/Gating Network) do dynamicznego kierowania wnioskowania do wyspecjalizowanych ekspertów (modeli wąskodziedzinowych).

---

## 🎯 Kluczowe kompetencje zaprezentowane w kodzie
* Umiejętność analitycznego myślenia i wizualnego wyciągania wniosków z danych (Matplotlib / Seaborn).
* **Biegła znajomość NumPy** i wektoryzacji operacji w Pythonie.
* Zdolność przełożenia **złożonych wzorów matematycznych** (pochodne, funkcje straty, entropia) na optymalny kod.
* Rozwiązywanie realnych problemów biznesowych (klasyfikacja i regresja) w oparciu o czyste dane.
* Świadomość najnowocześniejszych architektur używanych w dzisiejszej sztucznej inteligencji (*Ensemble*, *Mixture of Experts*).

---

## 🛠️ Narzędzia i Technologie
* **Język:** Python 3
* **Przetwarzanie danych i algorytmy od zera:** NumPy, Pandas
* **Modelowanie i benchmarking:** `scikit-learn`
* **Wizualizacja:** Matplotlib, Seaborn
* **Środowisko programistyczne:** Jupyter Notebook

---

## 🚀 Jak uruchomić projekt lokalnie?

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/Matixy/Systems-Analysis-and-Decision-Support-Methods.git
   ```
2. Przejdź do folderu z projektem:
   ```bash
   cd Systems-Analysis-and-Decision-Support-Methods
   ```
3. Stwórz i aktywuj środowisko wirtualne (zalecane):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```
4. Zainstaluj wymagane zależności:
   ```bash
   pip install -r requirements.txt
   ```
5. Uruchom wybrane laboratoria z poziomu Jupyter Notebook:
   ```bash
   jupyter notebook
   ```