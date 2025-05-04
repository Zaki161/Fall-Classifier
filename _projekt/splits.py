import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def losowy(features_df):
    X = features_df.drop(columns=['fall', 'subject_id'])
    y = features_df['fall']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test

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

def osoby_v2(features_df, procent_testowych):
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