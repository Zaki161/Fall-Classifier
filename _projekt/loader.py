import pandas as pd
import glob

def load_data(path='../DANE/*.csv'):
    files = glob.glob(path)
    df_list = [pd.read_csv(file) for file in files]
    df = pd.concat(df_list, ignore_index=True)
    
    # Obliczanie cech 
    df['acc_magnitude'] = (df['x_acc']**2 + df['y_acc']**2 + df['z_acc']**2) ** 0.5
    df['gyro_magnitude'] = (df['x_gyro']**2 + df['y_gyro']**2 + df['z_gyro']**2) ** 0.5
    df['acc_diff'] = df['acc_magnitude'].diff().fillna(0)
    df['gyro_diff'] = df['gyro_magnitude'].diff().fillna(0)
    
    # df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    return df