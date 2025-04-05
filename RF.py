import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Wczytanie wszystkich plików CSV
files = glob.glob('DANE/*.csv')
df_list = [pd.read_csv(file) for file in files]
df = pd.concat(df_list, ignore_index=True)

# Sprawdzenie danych
print(df.head())
print(df.isnull().sum())
print(df['fall'].value_counts())

# Obliczanie cech
df['acc_magnitude'] = (df['x_acc']**2 + df['y_acc']**2 + df['z_acc']**2) ** 0.5
df['gyro_magnitude'] = (df['x_gyro']**2 + df['y_gyro']**2 + df['z_gyro']**2) ** 0.5
df['acc_diff'] = df['acc_magnitude'].diff().fillna(0)
df['gyro_diff'] = df['gyro_magnitude'].diff().fillna(0)

# Funkcja do ekstrakcji cech
def extract_features(df, window_size, shift_size):
    features = []
    for start in range(0, len(df) - window_size, shift_size):
        end = start + window_size
        window_data = df.iloc[start:end]
        feature = {
            'acc_magnitude_mean': window_data['acc_magnitude'].mean(),
            'gyro_magnitude_mean': window_data['gyro_magnitude'].mean(),
            'acc_diff_mean': window_data['acc_diff'].mean(),
            'gyro_diff_mean': window_data['gyro_diff'].mean(),
            'acc_magnitude_std': window_data['acc_magnitude'].std(),
            'gyro_magnitude_std': window_data['gyro_magnitude'].std(),
            'fall': window_data['fall'].max()
        }
        features.append(feature)
    return pd.DataFrame(features)

# Ekstrakcja cech
window_size = 100
shift_size = 10
features_df = extract_features(df, window_size, shift_size)

# Podział na dane treningowe i testowe
X = features_df[['acc_magnitude_mean', 'gyro_magnitude_mean', 'acc_diff_mean', 'gyro_diff_mean', 'acc_magnitude_std', 'gyro_magnitude_std']]
y = features_df['fall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trenowanie modelu
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predykcja i ocena
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Zapisanie przygotowanych danych
features_df.to_csv('prepared_data.csv', index=False)