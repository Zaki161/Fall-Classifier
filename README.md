# Klasyfikator wypadków
Celem pracy jest analiza, wybór oraz implementacja modelu klasyfikacji do wykrywania upadków u osóbstarszych z wykorzystaniem biosensorów wbudowanych w smartfony. Praca obejmuje przegląd metod uczenia maszynowego, przygotowanie zbioru danych na podstawie sygnałów z akcelerometru i żyroskopu, a następnie trenowanie modelu klasyfikacyjnego. Finalnym etapem jest integracja modelu z aplikacją mobilną działającą na systemie Android oraz testowanie skuteczności detekcji upadków.

## Dane 

https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20/data

Dopasowanie danych w **merge.py** polega na łączeniu wartości z akcelerometru (ACC) z najbliższymi czasowo wartościami z żyroskopu (GYRO).

Żyroskop rejestruje pomiary znacznie częściej niż akcelerometr – około trzykrotnie częściej.

Częstotliwości próbkowania:
	Akcelerometr (ACC):
	•	Odstęp między pomiarami: 0,02 s (20 ms)
	•	Częstotliwość próbkowania: 50 Hz
	Żyroskop (GYRO):
	•	Odstęp między pomiarami: 0,0005 s (0,5 ms)
	•	Częstotliwość próbkowania: 2000 Hz

Ze względu na różnicę w częstotliwościach, niektóre próbki żyroskopu mogą zostać pominięte podczas dopasowywania do danych z akcelerometru.

```bash
python mine.py
```


## Inzynieria danych 

Aby poprawic jakosc modelu zostały stworzone dodatkowe cechy:
 - wielkosc wektora przyspieszenia, 
 - wielkosc wektora predkosci katowej, 
 - okresy ruchu

Okno przesuwajace ( sliding window) - dzielimy na dlugosc 100 probek i w kazdym oknie obliczamy rozne cechcy. Okno przesuwa się o pewna stala luczbe -10 probek, aby uzyskac plynnosc analizy w czasie rzeczywistym.

cechy x:
'acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender'
cecha y:'fall'


## Algorytm
z wykorzystaniem sklearn


### RF


### knn 


## Podzial danych

### train_test_split()
https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_early_stopping.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-early-stopping-py



## Wynik 

**DLA KNN (n=5)**

              precision    recall  f1-score   support

           0       0.97      0.99      0.98     23108
           1       0.95      0.87      0.91      5871

    accuracy                           0.96     28979
   macro avg       0.96      0.93      0.94     28979
weighted avg       0.96      0.96      0.96     28979

Wykrycie wypadkow dla 95% 
**DLA RF**

  precision    recall  f1-score   support

           0       0.98      0.99      0.99     23108
           1       0.96      0.93      0.95      5871

    accuracy                           0.98     28979
   macro avg       0.97      0.96      0.97     28979
weighted avg       0.98      0.98      0.98     28979

