# PRIMERO EJECUTAR EN TERMINAL: python spacy_sklearn_bert.py
# SEGUNDO EJECUTAR EN OTRA TERMINAL: python test_api.py

import requests
import json

API_URL = "http://127.0.0.1:5001/comparar_cursos"

def enviar_comparacion(comparaciones):
    """
    Envía una solicitud POST a la API con las comparaciones de cursos.

    :param comparaciones: Lista de diccionarios con cursos origen y destino.
    :return: Respuesta de la API o error capturado.
    """
    payload = {
        "comparaciones": comparaciones
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Lanza error si status >= 400
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud: {e}")
        return None

def cargar_datos_prueba():
    """
    Retorna un listado de comparaciones de prueba.

    Puedes modificar o extender esta función para cargar desde un archivo o base de datos.
    """
    with open('comparaciones_demo.json', 'r', encoding='utf-8') as f:
        return json.load(f)['comparaciones']

def main():
    """
    Función principal que carga los datos de prueba, hace la solicitud y muestra la respuesta.
    """
    comparaciones = cargar_datos_prueba()
    resultado = enviar_comparacion(comparaciones)

    if resultado:
        print("=====RESULTADO=====\n")
        print(json.dumps(resultado, indent=4, ensure_ascii=False))
    else:
        print("No se recibió respuesta de la API.")

if __name__ == "__main__":
    main()
