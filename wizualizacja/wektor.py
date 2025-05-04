import glob
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#### zapisujemy wykresy wektorowe sensorow dla pacjenta 

# Folder na zapis wykresów
os.makedirs('wykresy', exist_ok=True)

# moje pliki
files = glob.glob('DANE/*.csv')

# Losuj 5 plików
random_files = random.sample(files, 5)

for file in random_files:
    df = pd.read_csv(file)
    
    # Info osoby
    subject_id = df['subject_id'].iloc[0]
    activity_id = df['activity'].iloc[0]
    age = df['age'].iloc[0]
    height = df['height'].iloc[0]
    weight = df['weight'].iloc[0]
    gender = df['gender'].iloc[0]
    opis_osoby = f'Subject {subject_id} | {gender}, {age} lat | {height} cm, {weight} kg'

    # Wektory (moduł wektora)
    df['acc_vector'] = np.sqrt(df['x_acc']**2 + df['y_acc']**2 + df['z_acc']**2)
    df['gyro_vector'] = np.sqrt(df['x_gyro']**2 + df['y_gyro']**2 + df['z_gyro']**2)

    # --- Wykres Akcelerometru ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['acc_vector'], label='|acc|', color='black', linewidth=2)
    plt.plot(df['timestamp'], df['x_acc'], label='x_acc', alpha=0.6)
    plt.plot(df['timestamp'], df['y_acc'], label='y_acc', alpha=0.6)
    plt.plot(df['timestamp'], df['z_acc'], label='z_acc', alpha=0.6)
    plt.title(f'Akcelerometr - {opis_osoby}')
    plt.xlabel('Czas (timestamp)')
    plt.ylabel('Przyspieszenie (g)')
    plt.legend()
    plt.tight_layout()

    # Zapis wykresu
    fname = f"wykresy/{activity_id}_{subject_id}_acc.png"
    plt.savefig(fname)
    plt.close()

    # --- Wykres Żyroskopu ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['gyro_vector'], label='|gyro|', color='black', linewidth=2)
    plt.plot(df['timestamp'], df['x_gyro'], label='x_gyro', alpha=0.6)
    plt.plot(df['timestamp'], df['y_gyro'], label='y_gyro', alpha=0.6)
    plt.plot(df['timestamp'], df['z_gyro'], label='z_gyro', alpha=0.6)
    plt.title(f'Żyroskop - {opis_osoby}')
    plt.xlabel('Czas (timestamp)')
    plt.ylabel('Prędkość kątowa (deg/s)')
    plt.legend()
    plt.tight_layout()

    # Zapis wykresu
    fname = f"wykresy/{activity_id}_{subject_id}_gyro.png"
    plt.savefig(fname)
    plt.close()