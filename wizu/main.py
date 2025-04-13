import glob
import random
import pandas as pd
import matplotlib.pyplot as plt

### WARTOSCI  XYZ sensorow dla kilku losowych plikow TESTY

#moje pliki 
# files = glob.glob('DANE/*.csv')

# Losuj 5 plików
random_files = random.sample(files, 5)

# Tworzenie wykresów
for file in random_files:
    # Wczytaj dane z pliku CSV
    df = pd.read_csv(file)
    
    # Pobierz dane osoby z pierwszego wiersza - dane sa stale dla pliku
    subject_id = df['subject_id'].iloc[0]
    age = df['age'].iloc[0]
    height = df['height'].iloc[0]
    weight = df['weight'].iloc[0]
    gender = df['gender'].iloc[0]
    
    opis_osoby = f'Subject {subject_id} | {gender}, {age} lat | {height} cm, {weight} kg'

    # Wykres akcelerometru
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['x_acc'], label='x_acc')
    plt.plot(df['timestamp'], df['y_acc'], label='y_acc')
    plt.plot(df['timestamp'], df['z_acc'], label='z_acc')
    plt.title(f'Akcelerometr - {opis_osoby}')
    plt.xlabel('Czas (timestamp)')
    plt.ylabel('Przyspieszenie (g)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Wykres żyroskopu
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['x_gyro'], label='x_gyro')
    plt.plot(df['timestamp'], df['y_gyro'], label='y_gyro')
    plt.plot(df['timestamp'], df['z_gyro'], label='z_gyro')
    plt.title(f'Żyroskop - {opis_osoby}')
    plt.xlabel('Czas (timestamp)')
    plt.ylabel('Prędkość kątowa (deg/s)')
    plt.legend()
    plt.tight_layout()
    plt.show()