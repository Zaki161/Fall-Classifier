"""
TAAK
SKrypt do zeglądania głównego katalogu (data_dir), który zawiera dane. 
Następnie przechodzi do podfolderów, w których znajdują się dane. 
az znajdzie pliki txt i uzywa klasy SensorDataProcessor z pliku merge.py i tam tworzy jeden pllik
"""

import os
from merge import SensorDataProcessor
# from merge import SensorDataProcessor

# data_dir = "/Users/zs/.cache/kagglehub/datasets/kmknation/mobifall-dataset-v20/versions/1/MobiFall_Dataset_v2.0/"
## UWAGA ZMINENILAS WERSJE NA 3 (SKROCONY KATALOG)  
data_dir ="/Users/zakiashefa/.cache/kagglehub/datasets/kmknation/mobifall-dataset-v20/versions/1/MobiFall_Dataset_v2.0/"

FILE_COUNT = 0  # Deklarujemy zmienną globalną

def getFiles(catalog_path,fall):
    files = sorted(os.listdir(catalog_path))
    acc_files = []
    gyro_files = []
    ori_files = []
    
    # Rozdzielamy pliki na te, które zaczynają się od FOL_acc, FOL_gyro i FOL_ori
    for file in files:
        # Sprawdzamy, co znajduje się po pierwszym podkreśleniu w nazwie
        parts = file.split('_')
        if len(parts) >= 3:
            file_type = parts[1]  # acc, gyro, ori
            if file_type == "acc":
                acc_files.append(file)
                print(file)
            elif file_type == "gyro":
                gyro_files.append(file)
                print(file)
            elif file_type == "ori":
                ori_files.append(file)
                print(file)

    print("#### We are looking for pairs:")
    
    # Tworzymy listy par plików, porównując numery w nazwach plików
    # i=0
    for acc in acc_files:
        print(" we are here now ", acc)
        # Ekstrahujemy numer z nazwy pliku (np. "9_1" z "FOL_acc_9_1.txt")
        acc_num = acc.split('_')[3]  # "9_1"
        print(acc_num)
        # j=0
        for gyro in gyro_files:
            # j+=1
            global FILE_COUNT
            FILE_COUNT+=1

            gyro_num=gyro.split('_')[3]

            if acc_num == gyro_num:
                print(f"Para! {acc} i {gyro}")

                processor = SensorDataProcessor(os.path.join(catalog_path,acc), os.path.join(catalog_path,gyro),fall)
                name_file = f"{FILE_COUNT}.csv"
                print(name_file)
                full_file_path = os.path.join("DANE", name_file)
                print(full_file_path)

                processor.save_to_csv(full_file_path)

        
        
        
    print("#### THANKS :")


def ADL():
    files = sorted(os.listdir(sub_fol_fol_path))
    for file in files:
        print(f"   Plik: {file}")
    for pliki in files[1:]:
        sub_path = os.path.join(sub_fol_fol_path, pliki)
    # Sprawdzamy, czy to katalog
        if not os.path.isdir(sub_path):
            continue 
        print(f"        #######") 
        print(f"        Folder:", sub_fol)
        print(f"        Ścieżka:", sub_path)
        getFiles(sub_path,0)



def FALL():
    files = sorted(os.listdir(sub_fol_fol_path))
    for file in files:
        print(f"   Plik: {file}")
    for pliki in files[1:]:
        sub_path = os.path.join(sub_fol_fol_path, pliki)
    # Sprawdzamy, czy to katalog
        if not os.path.isdir(sub_path):
            continue 
        print(f"        #######") 
        print(f"        Folder:", sub_fol)
        print(f"        Ścieżka:", sub_path)
        getFiles(sub_path,1)



print("Moja ścieżka:", data_dir)

# Pobieramy listę folderów i sortujemy
l = sorted(os.listdir(data_dir))

# Pomijamy README, więc zaczynamy od indeksu 1
i=0
for sub_fol in l[1:]:  
    sub_path = os.path.join(data_dir, sub_fol)
    
    # Sprawdzamy, czy to katalog
    if not os.path.isdir(sub_path):
        continue  

    print("#######") 
    print("Folder:", sub_fol)
    print("Ścieżka:", sub_path)

    print("///POCZĄTEK///")
    
    # Pobieramy listę podfolderów ADL/FALL
    try:
        sub_folders = sorted(os.listdir(sub_path))
        for sub_fol_fol in sub_folders:
            sub_fol_fol_path = os.path.join(sub_path, sub_fol_fol)
            if os.path.isdir(sub_fol_fol_path):
                print(f"Podfolder: {sub_fol_fol}")
                if sub_fol_fol == "ADL":
                    ADL()
                if sub_fol_fol == "FALLS":
                    FALL()
                # Pobieramy pliki w podfolderze
                # files = sorted(os.listdir(sub_fol_fol_path))
                # for file in files:
                #     print(f"   Plik: {file}")

    except Exception as e:
        print("Błąd odczytu:", e)

    print("///KONIEC///")
