"""
TAAK
Skrypt do laczenia ze soba danych z plikow acc i gyro oraz dodawaniu odpowienich etykiet 
"""
import pandas as pd
import io

class SensorDataProcessor:
    def __init__(self, acc_file, gyro_file,fall):
        """Inicjalizuje obiekt klasy, wczytując pliki z danymi."""
        self.acc_file = acc_file
        self.gyro_file = gyro_file
        self.fall=fall
        self.subject_info = self.getSubject(gyro_file)
        self.acc_data = self.load_sensor_data(acc_file)
        self.gyro_data = self.load_sensor_data(gyro_file)

    def getSubject(self, filename):
        """Funkcja wczytuje nagłówki pliku i wyciąga dane o subjekcie i aktywności."""
        subject_info = {}
        
        with open(filename, "r") as f:
            lines = f.readlines()

        # Szukanie i wyciąganie danych za pomocą wyrażeń regularnych
        for line in lines:
            # if "#Subject ID:" in line:
            #     subject_info["subject_id"] = int(line.split(":")[1].strip())
            # elif "#First Name:" in line:
            #     subject_info["first_name"] = line.split(":")[1].strip()
            # elif "#Last Name:" in line:
            #     subject_info["last_name"] = line.split(":")[1].strip()
            if "#Age:" in line:
                subject_info["age"] = int(line.split(":")[1].strip())
            elif "#Height(cm):" in line:
                subject_info["height"] = int(line.split(":")[1].strip())
            elif "#Weight(kg):" in line:
                subject_info["weight"] = int(line.split(":")[1].strip())
            elif "#Gender:" in line:
                subject_info["gender"] = line.split(":")[1].strip()
            subject_info["fall"]=self.fall
            # elif "#Activity:" in line:
            #     activity_info = line.split(":")[1].strip().split(" - ")
            #     subject_info["activity_id"] = int(activity_info[0].strip())
            #     subject_info["activity_name"] = activity_info[1].strip()

        return subject_info

    def load_sensor_data(self, filename):
        """Wczytuje dane z pliku, pomijając nagłówki i linie niebędące danymi."""
        with open(filename, "r") as f:
            lines = f.readlines()

        # Znalezienie linii z danymi (@DATA) i pobranie tylko wartości numerycznych
        data_start = next(i for i, line in enumerate(lines) if "@DATA" in line) + 1
        data = lines[data_start:]

        # Wczytanie do pandas
        df = pd.read_csv(
            io.StringIO("".join(data)),  # Konwersja na strumień tekstowy
            names=["timestamp", "x", "y", "z"],
            dtype={"timestamp": float, "x": float, "y": float, "z": float}
        )

        return df

    def process_data(self):
        """Łączy dane akcelerometru i żyroskopu z dodatkowymi informacjami."""
        # Sortowanie danych według timestamp (ważne dla merge_asof)
        self.gyro_data = self.gyro_data.sort_values("timestamp")
        self.acc_data = self.acc_data.sort_values("timestamp")

        # Dopasowanie najbliższego timestamp z żyroskopu do akcelerometru
        merged = pd.merge_asof(self.acc_data, self.gyro_data, on="timestamp", direction="nearest", suffixes=("_acc", "_gyro"))

        # Dodanie danych osobowych oraz informacji o upadku do każdej próbki
        for column, value in self.subject_info.items():
            merged[column] = value

        return merged

    def save_to_csv(self, output_filename):
        """Zapisuje połączone dane do pliku CSV."""
        merged_data = self.process_data()
        merged_data.to_csv(output_filename, index=False)
        print(f"Połączone dane zapisane do {output_filename}")

# # Przykład użycia klasy
# processor = SensorDataProcessor("FKL_acc_9_1.txt", "FKL_gyro_9_1.txt")
# processor.save_to_csv("merged_data_with_subject_info.csv")