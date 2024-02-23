import os
import json
import gzip

"""
    Convert gzip-compressed JSON files in a specified input folder to plain JSON files in an output folder.

    Parameters:
    - input_folder: str
        The path to the folder containing gzip-compressed JSON files.
    - output_folder: str 
        The path to the folder where the resulting plain JSON files will be saved.

    
    Notes:
    - This function takes gzip-compressed JSON files from the 'input_folder', decompresses them,
      and saves the resulting plain JSON files to the 'output_folder'.
    - If the 'output_folder' does not exist, it will be created.
    - The function iterates through each gzip file in the 'input_folder' and converts it to a JSON file.
    - The new JSON files will have the same name as the original gzip files but with a '.json' extension.

    """
def gz_to_json(input_folder, output_folder):
    # Verificar si la carpeta de salida existe, si no, crearla
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtener la lista de archivos en la carpeta de entrada
    files = [f for f in os.listdir(input_folder) if f.endswith('.gz')]

    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace('.gz', '.json'))

        with gzip.open(input_path, 'rt') as gz_file, open(output_path, 'w') as json_file:
            # Leer el contenido del archivo gzip y cargarlo como un objeto JSON
            data = json.load(gz_file)

            # Escribir el objeto JSON en el nuevo archivo
            json.dump(data, json_file, indent=2)