import requests
import json

"""
Para ejecutar este archivo primero ejecutar python main.py y luego python curl.py en dos terminales distintas
"""

# Define la URL de tu API
url = 'http://127.0.0.1:5000/'

# Define los datos que quieres enviar en formato JSON
data = {
    "cursoLocal": "Fundamentos de Programacion",
    "silaboLocal": "Fundamentos de programacion: algoritmos, estructuras de control, funciones y arreglos",
    "cursoPostulante": "Introduccion a la Programacion",
    "silaboPostulante": "Introduccion a la Programacion: uso de pseint, conceptos generales de programacion"
}

# Convierte el diccionario de datos a una cadena JSON
json_data = json.dumps(data)

# Define los encabezados de la solicitud
headers = {'Content-Type': 'application/json'}

# Realiza la solicitud POST
response = requests.post(url, data=json_data, headers=headers)

# Verifica el código de estado de la respuesta
if response.status_code == 200:
    # Imprime la respuesta en formato JSON
    print(response.json())
else:
    print(f'Error en la solicitud: {response.status_code} {response.reason}')
