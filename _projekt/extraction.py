import pandas as pd
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