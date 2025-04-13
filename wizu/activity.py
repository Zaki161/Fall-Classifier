import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Mapa aktywności: id -> (code, name, type)
activity_map = {
    1:  ("STD", "Standing", "ADL"),
    2:  ("WAL", "Walking", "ADL"),
    3:  ("JOG", "Jogging", "ADL"),
    4:  ("JUM", "Jumping", "ADL"),
    5:  ("STU", "Stairs_up", "ADL"),
    6:  ("STN", "Stairs_down", "ADL"),
    7:  ("SCH", "Sit_chair", "ADL"),
    8:  ("CSI", "Car_step_in", "ADL"),
    9:  ("CSO", "Car_step_out", "ADL"),
    10: ("FOL", "Forward_lying", "Fall"),
    11: ("FKL", "Front_knees_lying", "Fall"),
    12: ("BSC", "Back_sitting_chair", "Fall"),
    13: ("SDL", "Sideward_lying", "Fall"),
}

#  pliki CSV
files = glob.glob('DANE/*.csv')

for file in files:
    df = pd.read_csv(file)

    # Wczytaj podstawowe info
    subject_id = df['subject_id'].iloc[0]
    activity_id = df['activity'].iloc[0]
    age = df['age'].iloc[0]
    gender = df['gender'].iloc[0]

    #dane aktywności
    code, activity_name, category = activity_map.get(activity_id, ("UNK", "Unknown", "Unknown"))
    base_folder = f"{'sub_activity' if category == 'ADL' else 'falls'}/{activity_name}"
    os.makedirs(base_folder, exist_ok=True)

    # Oblicz wektory
    df['acc_vector'] = np.sqrt(df['x_acc']**2 + df['y_acc']**2 + df['z_acc']**2)
    df['gyro_vector'] = np.sqrt(df['x_gyro']**2 + df['y_gyro']**2 + df['z_gyro']**2)

    # Opis osoby
    opis_osoby = f'Subject {subject_id} | {gender}, {age} lat'

    # # Akcelerometr
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['timestamp'], df['acc_vector'], label='|acc|', color='black', linewidth=2)
    # plt.plot(df['timestamp'], df['x_acc'], label='x_acc', alpha=0.6)
    # plt.plot(df['timestamp'], df['y_acc'], label='y_acc', alpha=0.6)
    # plt.plot(df['timestamp'], df['z_acc'], label='z_acc', alpha=0.6)
    # plt.title(f'{activity_name} ({code}) - Akcelerometr\n{opis_osoby}')
    # plt.xlabel('Czas')
    # plt.ylabel('Przyspieszenie (g)')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'{base_folder}/{code}_{subject_id}_acc.png')
    # plt.close()

    # # Żyroskop
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['timestamp'], df['gyro_vector'], label='|gyro|', color='black', linewidth=2)
    # plt.plot(df['timestamp'], df['x_gyro'], label='x_gyro', alpha=0.6)
    # plt.plot(df['timestamp'], df['y_gyro'], label='y_gyro', alpha=0.6)
    # plt.plot(df['timestamp'], df['z_gyro'], label='z_gyro', alpha=0.6)
    # plt.title(f'{activity_name} ({code}) - Żyroskop\n{opis_osoby}')
    # plt.xlabel('Czas')
    # plt.ylabel('Prędkość kątowa (deg/s)')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'{base_folder}/{code}_{subject_id}_gyro.png')
    # plt.close()

        # MIX
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['gyro_vector'], label='|gyro|', color='blue', linewidth=2)
    plt.plot(df['timestamp'], df['acc_vector'], label='|acc|', color='green', linewidth=2)
    # plt.plot(df['timestamp'], df['y_gyro'], label='y_gyro', alpha=0.6)
    # plt.plot(df['timestamp'], df['z_gyro'], label='z_gyro', alpha=0.6)
    plt.title(f'{activity_name} ({code}) - ACC, GYRO\n{opis_osoby}')
    plt.xlabel('Czas')
    plt.ylabel('Prędkość kątowa (deg/s)\n Przyspieszenie (g)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{base_folder}/{code}_{subject_id}_mix.png')
    plt.close()