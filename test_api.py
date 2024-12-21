import requests
import json

# URL de la API (ajusta según tu configuración)
API_URL = "http://127.0.0.1:5000"

# Función para enviar una solicitud a la API
def test_api(curso_local, silabo_local, curso_postulante, silabo_postulante):
    payload = {
        "cursoLocal": curso_local,
        "silaboLocal": silabo_local,
        "cursoPostulante": curso_postulante,
        "silaboPostulante": silabo_postulante
    }
    response = requests.post(API_URL, json=payload)
    return response.json()

# Conjuntos de prueba
test_cases = [
    {
        "curso_local": "Fundamentos de Programacion",
        "silabo_local": "Introducción a Python, estructuras de datos, control de flujo, funciones, manejo de errores",
        "curso_postulante": "Introduccion a la Programacion",
        "silabo_postulante": "Fundamentos de programación en Python, estructuras condicionales, ciclos, listas, funciones"
    },
    {
        "curso_local": "Programación avanzada en Java",
        "silabo_local": "Programación orientada a objetos, patrones de diseño, concurrencia, APIs de Java",
        "curso_postulante": "Desarrollo de aplicaciones empresariales con Java",
        "silabo_postulante": "Java EE, Servlets, JSP, JPA, Spring Framework"
    },
    {
        "curso_local": "Fundamentos de Programacion",
        "silabo_local": "Fundamentos de programacion: algoritmos, estructuras de control, funciones y arreglos",
        "curso_postulante": "Introduccion a la Programacion",
        "silabo_postulante": "Introduccion a la Programacion: uso de pseint, conceptos generales de programacion"
    },
    {
        "curso_local": "Sistemas Operativos",
        "silabo_local": "Sistemas operativos: gestion de procesos, memoria y dispositivos.",
        "curso_postulante": "Sistemas Operativos Avanzados",
        "silabo_postulante": "Diseño e implementacion de sistemas operativos avanzados, incluyendo gestion de memoria y sistemas de archivos."
    },
    {
        "curso_local": "Diseño de Interfaces de Usuario",
        "silabo_local": "Principios de diseño de interfaces de usuario y experiencia de usuario (UI/UX).",
        "curso_postulante": "Diseño de Experiencia de Usuario",
        "silabo_postulante": "Diseño de interfaces de usuario centrado en el usuario para aplicaciones moviles y web."
    },
    {
        "curso_local": "Bases de Datos Relacionales",
        "silabo_local": "Fundamentos de bases de datos relacionales, modelado de datos y SQL.",
        "curso_postulante": "Administracion de Bases de Datos",
        "silabo_postulante": "Administracion y optimizacion de sistemas de gestion de bases de datos relacionales y no relacionales."
    },
    {
        "curso_local": "Inteligencia Artificial",
        "silabo_local": "Introduccion a la inteligencia artificial, algoritmos de busqueda, aprendizaje automatico y redes neuronales.",
        "curso_postulante": "Computacion Bioinspirada",
        "silabo_postulante": "Computacion inspirada en la naturaleza: algoritmos geneticos, optimizacion por enjambres y redes neuronales artificiales."
    },
    {
        "curso_local": "Desarrollo de Software",
        "silabo_local": "Desarrollo de software: metodologias agiles, diseño orientado a objetos, patrones de diseño.",
        "curso_postulante": "Ingenieria de Software",
        "silabo_postulante": "Fundamentos y tecnicas avanzadas en el desarrollo de software, incluyendo metodologias agiles y practicas de ingenieria de software."
    },
    {
        "curso_local": "Seguridad en Redes",
        "silabo_local": "Seguridad en redes de computadoras: firewalls, deteccion de intrusiones, analisis forense y politicas de seguridad.",
        "curso_postulante": "Ethical Hacking",
        "silabo_postulante": "Pruebas de penetracion y evaluacion de vulnerabilidades de seguridad para fortalecer la infraestructura informatica."
    },
    {
        "curso_local": "Sistemas Distribuidos",
        "silabo_local": "Sistemas distribuidos: arquitecturas cliente-servidor, middleware, comunicacion y concurrencia.",
        "curso_postulante": "Computacion en la Nube",
        "silabo_postulante": "Provisionamiento, despliegue y gestion de servicios en la nube para escalabilidad y disponibilidad."
    },
    {
        "curso_local": "Desarrollo de Aplicaciones Moviles",
        "silabo_local": "Desarrollo de aplicaciones moviles: plataformas iOS y Android, diseño de interfaces y optimizacion de rendimiento.",
        "curso_postulante": "Computacion Ubicua",
        "silabo_postulante": "Integracion de tecnologias moviles en entornos ubicuos para la interaccion y el procesamiento de datos."
    },
    {
        "curso_local": "Computacion Cuantica",
        "silabo_local": "Introduccion a la computacion cuantica, qubits, algoritmos cuanticos y aplicaciones potenciales.",
        "curso_postulante": "Criptografia Cuantica",
        "silabo_postulante": "Seguridad de la informacion basada en principios cuanticos, incluyendo criptografia y comunicacion segura."
    }
]

# Ejecutar pruebas
for i, test_case in enumerate(test_cases, 1):
    print(f"\nPrueba #{i}")
    result = test_api(**test_case)
    print(json.dumps(result, indent=2))

    # Análisis básico de resultados
    print(f"Similitud del curso: {result['curso_similarity']:.2f}")
    print(f"Similitud del sílabo: {result['silabo_similarity']:.2f}")
    print(f"Similitud total: {result['total_similarity']:.2f}")
    print(f"Palabras similares en el curso: {len(result['similar_words_curso'])}")
    print(f"Palabras similares en el sílabo: {len(result['similar_words_silabo'])}")
