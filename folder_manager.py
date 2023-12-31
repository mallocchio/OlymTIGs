import os
import shutil
from datetime import datetime

def create_folder(TIG, base_directory):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f'{TIG}_{timestamp}'
    folder_path = os.path.join(base_directory, folder_name)
    try:
        os.makedirs(folder_path)
        print(f"Cartella '{folder_name}' creata con successo.\n")
        return folder_path
    except FileExistsError:
        print(f"La cartella '{folder_name}' esiste già.\n")
        return folder_path

def move_file(self, source_file_path, destination_folder):
    if not os.path.isfile(source_file_path):
        print(f"Il file '{source_file_path}' non esiste.\n")
        return

    destination_folder_path = os.path.join(self.base_directory, destination_folder)

    if not os.path.exists(destination_folder_path):
        print(f"La cartella '{destination_folder}' non esiste.\n")
        return

    try:
        shutil.move(source_file_path, os.path.join(destination_folder_path, os.path.basename(source_file_path)))
        print(f"File '{source_file_path}' spostato nella cartella '{destination_folder}' con successo.\n")
    except Exception as e:
        print(f"Si è verificato un errore durante lo spostamento del file: {e}\n")
