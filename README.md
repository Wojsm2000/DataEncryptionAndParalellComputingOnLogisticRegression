# DataEncryptionAndParalellComputingOnLogisticRegression
Szyfrowanie Danych i Obliczenia Równoległe w Regresji Logistycznej

## Przegląd Projektu

Projekt demonstruje **uczenie maszynowe zachowujące prywatność** przy użyciu **w pełni homomorficznego szyfrowania (FHE)** z modelami regresji logistycznej. Zaimplementowaliśmy podejścia sekwencyjne i równoległe dla zaszyfrowanego wnioskowania, porównując wydajność różnych zbiorów danych i frameworków.

## Co Zrealizowaliśmy

### Implementacja Szyfrowania Homomorficznego
- **Schemat CKKS**: Wykorzystanie biblioteki TenSEAL do przybliżonego szyfrowania homomorficznego
- **Wnioskowanie Zachowujące Prywatność**: Predykcje modelu na zaszyfrowanych danych bez konieczności deszyfrowania
- **Bezpieczna Konfiguracja Kontekstu**: Optymalne parametry szyfrowania dla równowagi bezpieczeństwo-wydajność

### Framework Obliczeń Równoległych
- **Pipeline Multiprocessingu**: Rozproszenie szyfrowania, ewaluacji i deszyfrowania między rdzenie CPU
- **Optymalizacja Wydajności**: Osiągnięcie znaczących przyspieszeń na większych zbiorach danych
- **Równoległość Oparta na Procesach**: Ominięcie ograniczeń wątkowych TenSEAL poprzez niezależne procesy robocze

### Implementacja w Dwóch Frameworkach

#### Wersja PyTorch (`torch_project.ipynb`)
- Niestandardowy model regresji logistycznej z optymalizacją SGD
- **Zbiór Danych**: Breast Cancer Dataset (~188 próbek, 30 cech)
- **Wyniki**: 97.87% trafności na danych jawnych, 88.83% na zaszyfrowanych
- **Obserwacja**: Przetwarzanie równoległe wolniejsze ze względu na mały zbiór danych

#### Wersja Scikit-Learn (`sklearn_covtype_project.ipynb`)
- Produkcyjna regresja logistyczna z wbudowaną regularyzacją
- **Zbiór Danych**: Forest Cover Type Dataset (~30k próbek, 54 cechy)
- **Wyniki**: Osiągnięto 3.6x przyspieszenie z przetwarzaniem równoległym
- **Skalowalność**: Wykazano korzyści wydajnościowe na większych zbiorach danych

## Kluczowe Osiągnięcia Techniczne

### Analiza Złożoności Obliczeniowej
Przeanalizowaliśmy złożoność algorytmiczną każdego komponentu:

**Najkosztowniejsze Operacje:**
- `encrypt_data()`: O(n·d·N) - szyfrowanie dominuje koszt obliczeniowy
- `evaluate_encrypted()`: O(n·d·N) - złożoność zaszyfrowanego wnioskowania

**Funkcje Równoległe:**
- Teoretyczne przyspieszenie: O(X/p) gdzie p = liczba procesów
- Praktyczne korzyści zależą od rozmiaru danych i overhead'u komunikacji


## Struktura Projektu

```
├── encrypted.py                   # Podstawowe funkcje szyfrowania/deszyfrowania
├── torch_project.ipynb            # Implementacja PyTorch
├── sklearn_covtype_project.ipynb  # Implementacja Scikit-learn  
├── projekt-opis.ipynb             # Szczegółowa analiza
└── README.md                      # Ten plik
```

## Kluczowe Funkcje

### Pipeline Podstawowy (`encrypted.py`)
- `encrypt_chunk()`: Równoległe szyfrowanie danych
- `evaluate_chunk()`: Ewaluacja zaszyfrowanego modelu
- `parallel_encrypt()`: Rozproszone szyfrowanie między procesami
- `parallel_evaluate()`: Równoległy pipeline wnioskowania

### Konfiguracja Kontekstu Szyfrowania
```python
context = ts.context(ts.SCHEME_TYPE.CKKS, 4096, 
                    coeff_mod_bit_sizes=[40, 20, 40])
context.global_scale = 2**20
context.generate_galois_keys()
```

## Przykład Użycia Pipeline'u

### Sequential Pipeline
```python
# 1. Trenowanie modelu
lr = train_model(X_train, y_train)
weight = lr.coef_[0]
bias = lr.intercept_[0]

# 2. Konfiguracja kontekstu HE
context = setup_he_context()

# 3. Szyfrowanie danych testowych
enc_X_test = encrypt_data(X_test, context)

# 4. Ewaluacja zaszyfrowanych danych
outputs = evaluate_encrypted(enc_X_test, weight, bias, context)

# 5. Deszyfrowanie wyników
logits = decrypt_outputs(outputs)

# 6. Predykcja i ewaluacja
preds = predict_from_logits(logits)
acc = np.mean(preds == y_test)
```

### Parallel Pipeline
```python
# Serializacja kontekstu do multiprocessingu
ctx_serialized = context.serialize()

# 1. Równoległe szyfrowanie
enc_X_test_par = parallel_encrypt(X_test, ctx_serialized)

# 2. Równoległa ewaluacja
outputs_serialized = parallel_evaluate(enc_X_test_par, y_test, 
                                     ctx_serialized, weight, bias, context)

# 3. Równoległe deszyfrowanie
logits_par = parallel_decrypt(outputs_serialized, context)

# 4. Predykcja i ewaluacja
preds_par = predict_from_logits(logits_par)
acc_par = np.mean(preds_par == y_test)
```

## Wyniki Badań

### Optymalne Przypadki Użycia
1. **Klasyfikacja Danych Medycznych**: Aplikacje krytyczne dla prywatności
2. **Ocena Ryzyka Finansowego**: Wymagania zgodności regulacyjnej  
3. **Wdrożenia na Dużą Skalę**: Gdzie rozmiar zbioru danych uzasadnia overhead

### Progi Wydajnościowe
- **Punkt Równowagi Równoległości**: ~200-10,000 próbek w zależności od cech
- **Optymalna Liczba Procesów**: Liczba rdzeni CPU (malejące zwroty powyżej)
- **Względy Pamięciowe**: Wektory CKKS wymagają znacznej ilości RAM

### Główne Obserwacje

#### PyTorch vs Scikit-Learn

| Framework | Zalety | Najlepsze dla |
|-----------|--------|---------------|
| **PyTorch** | • Większa elastyczność i kontrola<br>• Lepsza jakość modelu na małych zbiorach (97.87%)<br>• Niższy czas ewaluacji na małych danych | • Niestandardowe architektury<br>• Pełna kontrola nad modelem<br>• Integracja z deep learning |
| **Scikit-Learn** | • Szybszy i prostszy w użyciu<br>• Świetna współpraca z równoległością (3.6x)<br>• Wbudowana regularyzacja | • Klasyczne problemy ML<br>• Duże zbiory danych<br>• Szybkie prototypowanie |

#### Kiedy Stosować Równoległość

| ✅ **Opłacalna gdy:** | ❌ **Nieopłacalna gdy:** |
|----------------------|-------------------------|
| • n·d·N > próg (duże dane)<br>• p ≤ liczba rdzeni CPU<br>• Czas obliczeń > czas komunikacji | • Małe zbiory danych<br>• Więcej procesów niż rdzeni<br>• Wysokie koszty serializacji |

## Wykorzystane Technologie

- **TenSEAL**: Szyfrowanie homomorficzne
- **PyTorch**: Framework głębokiego uczenia
- **Scikit-Learn**: Klasyczne uczenie maszynowe
- **NumPy**: Operacje numeryczne
- **Matplotlib**: Wizualizacja danych
- **Multiprocessing**: Obliczenia równoległe

## Instalacja i Uruchomienie

```bash
# Instalacja wymaganych bibliotek
pip install tenseal torch scikit-learn numpy matplotlib pandas

# Uruchomienie notebooków
jupyter notebook sklearn_covtype_project.ipynb
# lub
jupyter notebook torch_project.ipynb
```

## Wnioski Końcowe

Projekt wykazał, że:

1. **Szyfrowanie homomorficzne jest wykonalne** dla klasyfikacji binarnej, ale wiąże się z znacznym kosztem obliczeniowym
2. **Równoległość przynosi korzyści** tylko przy dużych zbiorach danych (>10k próbek)
3. **Kompromis prywatność-wydajność** jest akceptowalny dla aplikacji krytycznych
4. **Punkt progu opłacalności** równoległości znajduje się między 200 a 10,000 próbek

## Autorzy i Licencja

Projekt realizowany w ramach projektu magisterskiego z zakresu **Metod Kryptografii**.

- Mikotaj Stefanski
- Aliaksei Shauchenka
- Kamil Bednarz
- Wojtek Smolarczyk
- Damian Torbus

---
*💡 **Tip**: Dla najlepszych wyników używaj Scikit-Learn z dużymi zbiorami danych i włączoną równoległością!*