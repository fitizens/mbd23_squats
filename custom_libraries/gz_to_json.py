import os
import json
import gzip
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