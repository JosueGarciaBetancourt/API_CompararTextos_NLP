# 📊 COMPARACIÓN DE CURSOS PLN

![NLP](https://img.shields.io/badge/NLP-Processing-blue)
![Python](https://img.shields.io/badge/Python-3.11.5-green)

Este proyecto utiliza **Procesamiento de Lenguaje Natural (PLN)** para analizar y comparar textos como descripciones de cursos o sílabos de diferentes instituciones educativas.

## 🛠️ Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- **Python 3.11.5** o superior
- Git (para clonar el repositorio)
- Entorno virtual (recomendado)

## 🚀 Configuración inicial

### 1. Verificar versión de Python
```bash
python --version
```
Si no tienes la versión compatible, descárgala desde: https://www.python.org/downloads/

### 2. Clonar repositorio
```bash
git clone https://github.com/JosueGarciaBetancourt/API_CompararTextos_NLP.git
cd API_CompararTextos_NLP
```

### 3. Crear y activar entorno virtual
```bash
python -m venv venv
```
#### Windows:
```bash
venv\Scripts\activate
```
#### Linux/Mac:
```bash
source venv/bin/activate
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. EJECUCIÓN
1. Primero ejecuta el servidor local de procesamiento NLP:
```bash
python spacy_sklearn_bert.py
```

2. Luego en otra terminal ejecuta la API:
```bash
python test_api.py
```

## 📌 Notas:
- Asegúrate de tener ambas terminales abiertas simultáneamente
- El entorno virtual debe estar activado en ambas terminales
