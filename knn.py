"""
Taaak 
knn klasyfikator 
"""

import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

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




# Ekstrakcja cech
window_size = 100 #dlugosc okna -probki 
shift_size = 10 #co ile przesuwamy

# features_df = extract_features(df, window_size, shift_size)
features_df = extract_features_full(df, window_size, shift_size)



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