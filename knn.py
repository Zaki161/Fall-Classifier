"""
Taaak 
knn klasyfikator 
"""

import pandas as pd
import glob
from sklearn.model_selection import train_test_split,StratifiedKFold, LeaveOneOut,TimeSeriesSplit
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np

# moj zbior danych
files = glob.glob('DANE/*.csv')
df_list = [pd.read_csv(file) for file in files]
df = pd.concat(df_list, ignore_index=True)

# Obliczanie cech 
df['acc_magnitude'] = (df['x_acc']**2 + df['y_acc']**2 + df['z_acc']**2) ** 0.5
df['gyro_magnitude'] = (df['x_gyro']**2 + df['y_gyro']**2 + df['z_gyro']**2) ** 0.5
df['acc_diff'] = df['acc_magnitude'].diff().fillna(0)
df['gyro_diff'] = df['gyro_magnitude'].diff().fillna(0)

# Funkcja do ekstrakcji cech  ( bez danych osobowych)
def extract_features(df, window_size, shift_size):
    features = []
    for start in range(0, len(df) - window_size, shift_size):
        end = start + window_size
        window_data = df.iloc[start:end]
        feature = {
            'acc_magnitude_mean': window_data['acc_magnitude'].mean(), #siła ruchu
            'gyro_magnitude_mean': window_data['gyro_magnitude'].mean(),
            'acc_diff_mean': window_data['acc_diff'].mean(), #zmiany
            'gyro_diff_mean': window_data['gyro_diff'].mean(),
            'acc_magnitude_std': window_data['acc_magnitude'].std(), # odchylenie standardowe 
            'gyro_magnitude_std': window_data['gyro_magnitude'].std(),
            'fall': window_data['fall'].max() #czy byl upadek
        }
        features.append(feature)
    return pd.DataFrame(features)

# Funkcja do ekstrakcji cech (pelna)
def extract_features_full(df, window_size, shift_size):
    
    # Mapowanie płci tylko raz
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    features = []
    for start in range(0, len(df) - window_size, shift_size):
        end = start + window_size
        window_data = df.iloc[start:end]
        feature = {
            'acc_magnitude_mean': window_data['acc_magnitude'].mean(), # siła ruchu
            'gyro_magnitude_mean': window_data['gyro_magnitude'].mean(),
            'acc_diff_mean': window_data['acc_diff'].mean(),           # zmiany
            'gyro_diff_mean': window_data['gyro_diff'].mean(),
            'acc_magnitude_std': window_data['acc_magnitude'].std(),   # odchylenie standardowe 
            'gyro_magnitude_std': window_data['gyro_magnitude'].std(),
            'fall': window_data['fall'].max(),                         # czy był upadek
            'age': window_data['age'].max(),
            'height': window_data['height'].max(),
            'weight': window_data['weight'].max(),
            'gender': window_data['gender'].max(),
        }
        features.append(feature)
    # wynikami cech dla wszystkich okien
    # czyli dla każdego przesunięcia okna w danych
    # Każdy wiersz w tym DataFrame odpowiada jednemu oknu i zawiera obliczone cechy.
    return pd.DataFrame(features)

def podzial_losowy():
    print("Jeste to Podział losowy")

    # Podział na dane treningowe i testowe 0.7 i 0.3
    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) ##powtarzalność wyników =42

    # Normalizacja danych (KNN wymaga standaryzacji)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Trenowanie modelu KNN (k=5)
    n_neighbors=3
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    print(f"Jest to knn {n_neighbors}")
    # Predykcja i ocena modelu
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))


def podzial_co_x():
    print("Jeste to Podział co x ")

    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']

    # Podział na dane treningowe i testowe co 10 próbek
    split_ratio = 0.7
    X_train = X.iloc[:int(len(X) * split_ratio)]
    X_test = X.iloc[int(len(X) * split_ratio):]

    y_train = y.iloc[:int(len(y) * split_ratio)]
    y_test = y.iloc[int(len(y) * split_ratio):]

    # Normalizacja danych (KNN wymaga standaryzacji)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Trenowanie modelu KNN (k=3)
    n_neighbors = 7
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    print(f"Jest to knn {n_neighbors}")

    # Predykcja i ocena modelu
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))

def kroswalidacja():
    print("Jeste to Kroswalidacja K-Fold ")
    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']
    # Ustalamy liczbę "fałd" (części), np. 5 fałd
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    n_neighbors = 7
    # Inicjalizujemy klasyfikator KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    print(f"Jest to knn {n_neighbors}")

    # Kroswalidacja - ocena modelu
    scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')  # Zmieniamy scoring na metrykę, którą chcemy (np. 'accuracy')

    print(f"Średnia dokładność: {np.mean(scores)}")
    print(f"Odchylenie standardowe: {np.std(scores)}")

def StratifiedKFold():
    print("Jeste to StratifiedKFold ")
    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']
    # Ustalamy liczbę fałd i losowy podział
    n_splits=5
    stratified_kf = StratifiedKFold(n_splits, shuffle=True, random_state=42)

    # Inicjalizujemy klasyfikator KNN
    knn = KNeighborsClassifier(n_neighbors=3)

    # Kroswalidacja z zachowaniem proporcji klas
    for train_index, test_index in stratified_kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        print(classification_report(y_test, y_pred))


def stratified_kfold_model():
    print("Jeste to StratifiedKFold")
    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']
    stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    for train_index, test_index in stratified_kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        print(classification_report(y_test, y_pred))

def loo_cv_model():
    print("Jeste to Leave-One-Out Cross-Validation ")
    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']
    loo = LeaveOneOut()

    knn = KNeighborsClassifier(n_neighbors=3)
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        print(classification_report(y_test, y_pred))

def time_series_split_model():
    print("Jeste to TimeSeriesSplit ")
    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']
    tscv = TimeSeriesSplit(n_splits=5)

    knn = KNeighborsClassifier(n_neighbors=3)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        print(classification_report(y_test, y_pred))



# Ekstrakcja cech
window_size = 100 #dlugosc okna -probki 
shift_size = 10 #co ile przesuwamy

# features_df = extract_features(df, window_size, shift_size)
features_df = extract_features_full(df, window_size, shift_size)


podzial_losowy() #dziala
# podzial_co_x() #dziala
# kroswalidacja() #dziala
# StratifiedKFold() # niedziala
# LOO_CV()#test
###

# stratified_kfold_model()
# loo_cv_model()
# time_series_split_model()





