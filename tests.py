"""

STAREEEE
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# import xgboost as xgb
# from lightgbm import LGBMClassifier

from sklearn.svm import SVC

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
            'subject_id': window_data['subject_id'].max()  

        }
        features.append(feature)
    # wynikami cech dla wszystkich okien
    # czyli dla każdego przesunięcia okna w danych
    # Każdy wiersz w tym DataFrame odpowiada jednemu oknu i zawiera obliczone cechy.
    return pd.DataFrame(features)

def losowy(features_df):

    # Podział na dane treningowe i testowe 0.7 i 0.3
    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) ##powtarzalność wyników =42

    # Normalizacja danych (KNN wymaga standaryzacji)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def osoby(features_df,procent_testowych):
    # Zbieramy osoby z danych
    unique_subjects = features_df['subject_id'].unique()
    
    # Mieszamy osoby losowo
    np.random.shuffle(unique_subjects)

    # Wyznaczamy ile osób ma być w zbiorze testowym
    n_testowych = int(len(unique_subjects) * procent_testowych)
    
    test_subjects = unique_subjects[:n_testowych]
    train_subjects = unique_subjects[n_testowych:]
    print("TEST:", test_subjects)
    print("TRAIN:", train_subjects)

    # Podział danych na podstawie subject_id
    train_df = features_df[features_df['subject_id'].isin(train_subjects)]
    test_df = features_df[features_df['subject_id'].isin(test_subjects)]

    # Wydzielenie X i y
    X_train = train_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean',
                        'acc_magnitude_std', 'gyro_magnitude_std', 'age', 'height', 'weight', 'gender']]
    y_train = train_df['fall']

    X_test = test_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean',
                      'acc_magnitude_std', 'gyro_magnitude_std', 'age', 'height', 'weight', 'gender']]
    y_test = test_df['fall']

    return X_train, X_test, y_train, y_test

def osoby_v2(features_df,procent_testowych):
    # Lista osób które mają zarówno fall jak i normalną aktywność
    osoby_fall_activity = [2, 3, 4, 5, 7, 8, 9, 10, 11]

    # Wszystkie osoby
    unique_subjects = features_df['subject_id'].unique()

    # Osoby tylko z upadkami (czyli reszta)
    osoby_only_fall = [s for s in unique_subjects if s not in osoby_fall_activity]

    # Losowe przetasowanie
    np.random.shuffle(osoby_fall_activity)
    np.random.shuffle(osoby_only_fall)

    # Wyznaczanie liczby osób do testu
    n_testowych_fall_activity = max(1, int(len(osoby_fall_activity) * procent_testowych)) # min. 1 osoba
    n_testowych_only_fall = max(1, int(len(osoby_only_fall) * procent_testowych)) # min. 1 osoba

    # Podział
    test_subjects = np.concatenate([
        osoby_fall_activity[:n_testowych_fall_activity],
        osoby_only_fall[:n_testowych_only_fall]
    ])

    train_subjects = np.setdiff1d(unique_subjects, test_subjects)

    print("TEST:", test_subjects)
    print("TRAIN:", train_subjects)

    # Podział danych na podstawie subject_id
    train_df = features_df[features_df['subject_id'].isin(train_subjects)]
    test_df = features_df[features_df['subject_id'].isin(test_subjects)]

    # Wydzielenie X i y
    X_train = train_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean',
                        'acc_magnitude_std', 'gyro_magnitude_std', 'age', 'height', 'weight', 'gender']]
    y_train = train_df['fall']

    X_test = test_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean',
                      'acc_magnitude_std', 'gyro_magnitude_std', 'age', 'height', 'weight', 'gender']]
    y_test = test_df['fall']

    return X_train, X_test, y_train, y_test



 
def KNN_podzial_losowy(features_df):
    print("===Jeste to Podział losowy KNN===")

    X_train, X_test, y_train, y_test = losowy(features_df)


    # Trenowanie modelu KNN (k=5)
    n_neighbors=3
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    print(f"Jest to knn {n_neighbors}")
    # Predykcja i ocena modelu
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))

def KNN_podzial_na_osoby(features_df, procent_testowych=0.3):
    print("===Jest to podział na osoby KNN===")

    X_train, X_test, y_train, y_test = osoby_v2(features_df,procent_testowych)

    

    # Trenowanie modelu KNN (k=3)
    n_neighbors = 5
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    print(f"Jest to KNN dla podziału na osoby, liczba sąsiadów {n_neighbors}")
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))

def RF_podzial_losowy():
    print("===Jeste to Podział losowy RF===")
    # Podział na dane treningowe i testowe
    X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std','age', 'height','weight','gender']]
    y = features_df['fall']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Trenowanie modelu
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predykcja i ocena
    y_pred = clf.predict(X_test)
    print("Raport RF:")
    print(classification_report(y_test, y_pred))

def RF_podzial_na_osoby(features_df, procent_testowych=0.3):
    print("===Jest to podział na osoby rf===")

    X_train, X_test, y_train, y_test = osoby_v2(features_df,procent_testowych)


#    # Trenowanie modelu
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train, y_train)
    
    # Trenowanie modelu z uwzględnieniem wag klas
    clf = RandomForestClassifier(
        n_estimators=300, 
        random_state=42,
        class_weight='balanced'  # <<< DODANE!
    )
    clf.fit(X_train, y_train)

    # Predykcja i ocena
    y_pred = clf.predict(X_test)
    print("Raport RF:")
    print(classification_report(y_test, y_pred))

def SVC_podzial_na_osoby(features_df, procent_testowych=0.2):
    print("===Jest to podział na osoby SVC===")

    X_train, X_test, y_train, y_test = osoby_v2(features_df,procent_testowych)
    # Ustalanie wag klas (jeśli dane są nierównomierne, np. rzadkie przypadki upadków)
    
    class_weights = {0: 1, 1: 5}  # Przykład ręcznego ustawienia wag, gdzie klasa 1 (upadki) ma wyższą wagę
    print(class_weights)
    # Tworzenie klasyfikatora SVM z wagami klas
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight=class_weights)  # Dodanie parametrów 'class_weight'
    

    # Tworzenie klasyfikatora SVM
    # svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)

    # Predykcja
    y_pred = svm.predict(X_test)

    # Ocena wyników
    print(classification_report(y_test, y_pred, zero_division=1))

def MLP_podzial_na_osoby(features_df, procent_testowych=0.2):
    print("===Jest to podział na osoby SVC===")

    X_train, X_test, y_train, y_test = osoby(features_df,procent_testowych)


    # Tworzenie klasyfikatora MLP
    mlp = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000)
    mlp.fit(X_train, y_train)

    # Predykcja
    y_pred = mlp.predict(X_test)

    # Ocena wyników
    print(classification_report(y_test, y_pred))



# Ekstrakcja cech
window_size = 100 #dlugosc okna -probki 
shift_size = 10 #co ile przesuwamy

features_df = extract_features_full(df, window_size, shift_size)

# print("MAMY 0.2")
KNN_podzial_na_osoby(features_df,0.2)

# print("MAMY 0.2")
# RF_podzial_na_osoby(features_df,0.2)

# print("MAMY 0.2")
RF_podzial_na_osoby(features_df,0.2)


# print("MAMY 0.2")
# RF_podzial_na_osoby(features_df,0.3)

# print("MAMY 0.2")
# RF_podzial_na_osoby(features_df,0.2)

print("MAMY 0.2")
SVC_podzial_na_osoby(features_df,0.2)
# print("MAMY 0.1")
# SVC_podzial_na_osoby(features_df,0.1)
# print("MAMY 0.3")
# LightGBM_podzial_na_osoby(features_df)