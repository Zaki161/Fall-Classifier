from merge import SensorDataProcessor

# Tworzymy obiekt klasy i zapisujemy połączone dane do pliku
processor = SensorDataProcessor("FKL_acc_9_1.txt", "FKL_gyro_9_1.txt")
processor.save_to_csv("merged_data_with_subject_info.csv")

print( "done")