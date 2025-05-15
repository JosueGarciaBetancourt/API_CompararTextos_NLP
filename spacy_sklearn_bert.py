# PRIMERO EJECUTAR EN TERMINAL: python spacy_sklearn_bert.py
# SEGUNDO EJECUTAR EN OTRA TERMINAL: python test_api.py

import time

from sklearn.feature_extraction.text import TfidfVectorizer

# ===============================================
# 📌 CONSTANTES CONFIGURABLES
# ===============================================

"""
FALTA IMPLEMENTAR!!!!

# Pesos para el cálculo de similitud de unidades
PESO_TITULO_UNIDAD = 0.30
PESO_APRENDIZAJE_UNIDAD = 0.40
PESO_TEMAS_UNIDAD = 0.30

# Umbrales
UMBRAL_TEMAS_COMUNES = 0.65  # Mínima similitud para considerar temas comunes
"""

tiempo_inicio_script = time.perf_counter()

import spacy
import unicodedata
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from collections import defaultdict
import time
import concurrent.futures
from functools import lru_cache
from threading import Lock
import traceback

app = Flask(__name__)

# ===============================================
# 📌 Configuración inicial optimizada
# ===============================================
tiempo_inicio_config = time.perf_counter()

nlp = spacy.load("es_core_news_lg", disable=['lemmatizer'])
model_name = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

tiempo_fin_config = time.perf_counter()
print(f"\n⏱ Tiempo configuración inicial: {tiempo_fin_config - tiempo_inicio_config:.4f} segundos")

# Cachés y bloqueos
embedding_cache = {}
text_processing_cache = {}
cache_lock = Lock()

# Expresiones regulares precompiladas
split_pattern = re.compile(r'[,.]')
clean_pattern = re.compile(r'[^a-zA-Z\s]')
space_pattern = re.compile(r'\s+')

# Stopwords optimizadas
stopwords_personalizadas = {
    "ia", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "de", "en", "a", "que", "con", "por", "para", "se", "su", "al", "es", "del"
}


# ===============================================
# 📌 Funciones optimizadas
# ===============================================

@lru_cache(maxsize=5000)
def normalizar_texto(texto):
    tiempo_inicio = time.perf_counter()

    if not texto:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [normalizar_texto] Texto vacío: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return ""

    # Paso 1: Conversión a minúsculas
    tiempo_inicio_minusculas = time.perf_counter()
    texto = str(texto).lower()
    tiempo_fin_minusculas = time.perf_counter()

    # Paso 2: Normalización Unicode
    tiempo_inicio_unicode = time.perf_counter()
    texto = unicodedata.normalize('NFD', texto)
    tiempo_fin_unicode = time.perf_counter()

    # Paso 3: Eliminación de acentos y caracteres especiales
    tiempo_inicio_clean = time.perf_counter()
    texto = texto.encode('ascii', 'ignore').decode("utf-8")
    texto = clean_pattern.sub('', texto)
    tiempo_fin_clean = time.perf_counter()

    # Paso 4: Normalización de espacios
    tiempo_inicio_spaces = time.perf_counter()
    resultado = space_pattern.sub(' ', texto).strip()
    tiempo_fin_spaces = time.perf_counter()

    tiempo_fin = time.perf_counter()

    print(f"\n⏱ [normalizar_texto] Tiempos parciales:")
    print(f"  - Minúsculas: {tiempo_fin_minusculas - tiempo_inicio_minusculas:.6f} segundos")
    print(f"  - Unicode: {tiempo_fin_unicode - tiempo_inicio_unicode:.6f} segundos")
    print(f"  - Limpieza: {tiempo_fin_clean - tiempo_inicio_clean:.6f} segundos")
    print(f"  - Espacios: {tiempo_fin_spaces - tiempo_inicio_spaces:.6f} segundos")
    print(f"⏱ [normalizar_texto] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")

    return resultado


def obtener_embeddings_bert(texto):
    tiempo_inicio = time.perf_counter()

    with cache_lock:
        if texto in embedding_cache:
            tiempo_fin = time.perf_counter()
            print(f"⏱ [obtener_embeddings_bert] Embedding desde caché: {tiempo_fin - tiempo_inicio:.6f} segundos")
            return embedding_cache[texto]

    # Tokenización
    tiempo_inicio_tokenizacion = time.perf_counter()
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tiempo_fin_tokenizacion = time.perf_counter()

    # Obtención de embeddings
    tiempo_inicio_modelo = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    tiempo_fin_modelo = time.perf_counter()

    # Procesamiento de salida
    tiempo_inicio_procesamiento = time.perf_counter()
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    tiempo_fin_procesamiento = time.perf_counter()

    with cache_lock:
        embedding_cache[texto] = embeddings

    tiempo_fin = time.perf_counter()

    print(f"\n⏱ [obtener_embeddings_bert] Tiempos parciales:")
    print(f"  - Tokenización: {tiempo_fin_tokenizacion - tiempo_inicio_tokenizacion:.6f} segundos")
    print(f"  - Modelo BERT: {tiempo_fin_modelo - tiempo_inicio_modelo:.6f} segundos")
    print(f"  - Procesamiento salida: {tiempo_fin_procesamiento - tiempo_inicio_procesamiento:.6f} segundos")
    print(f"⏱ [obtener_embeddings_bert] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")

    return embeddings


def calcular_similitud(vector1, vector2):
    tiempo_inicio = time.perf_counter()
    resultado = cosine_similarity([vector1], [vector2])[0][0]
    tiempo_fin = time.perf_counter()
    print(f"⏱ [calcular_similitud] Tiempo cálculo: {tiempo_fin - tiempo_inicio:.6f} segundos")
    return resultado


def procesar_texto(texto):
    tiempo_inicio = time.perf_counter()
    cache_key = hash(texto)

    with cache_lock:
        if cache_key in text_processing_cache:
            tiempo_fin = time.perf_counter()
            print(f"⏱ [procesar_texto] Procesamiento desde caché: {tiempo_fin - tiempo_inicio:.6f} segundos")
            return text_processing_cache[cache_key]

    # Normalización
    tiempo_inicio_normalizacion = time.perf_counter()
    texto_norm = normalizar_texto(texto)
    tiempo_fin_normalizacion = time.perf_counter()

    # Tokenización con spaCy
    tiempo_inicio_tokenizacion = time.perf_counter()
    doc = nlp(texto_norm)
    tiempo_fin_tokenizacion = time.perf_counter()

    # Filtrado de tokens
    tiempo_inicio_filtrado = time.perf_counter()
    tokens = [token for token in doc if (
            not token.is_stop and
            not token.is_punct and
            not token.is_digit and
            token.text not in stopwords_personalizadas
    )]
    tiempo_fin_filtrado = time.perf_counter()

    result = (texto_norm, tokens)
    with cache_lock:
        text_processing_cache[cache_key] = result

    tiempo_fin = time.perf_counter()

    print(f"\n⏱ [procesar_texto] Tiempos parciales:")
    print(f"  - Normalización: {tiempo_fin_normalizacion - tiempo_inicio_normalizacion:.6f} segundos")
    print(f"  - Tokenización: {tiempo_fin_tokenizacion - tiempo_inicio_tokenizacion:.6f} segundos")
    print(f"  - Filtrado: {tiempo_fin_filtrado - tiempo_inicio_filtrado:.6f} segundos")
    print(f"⏱ [procesar_texto] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")

    return result


# ===============================================
# 📌 Funciones principales mejoradas
# ===============================================

def identificar_temas_comunes(texto1, texto2, umbral=0.65):
    tiempo_inicio = time.perf_counter()

    # División de temas
    tiempo_inicio_division = time.perf_counter()
    temas1 = [tema.strip() for tema in split_pattern.split(texto1) if tema.strip()]
    temas2 = [tema.strip() for tema in split_pattern.split(texto2) if tema.strip()]
    tiempo_fin_division = time.perf_counter()

    if not temas1 or not temas2:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [identificar_temas_comunes] Sin temas para comparar: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return []

    # Obtención de embeddings en paralelo
    tiempo_inicio_embeddings = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        temas1_emb = {tema: emb for tema, emb in zip(
            temas1,
            executor.map(obtener_embeddings_bert, temas1)
        )}
        temas2_emb = {tema: emb for tema, emb in zip(
            temas2,
            executor.map(obtener_embeddings_bert, temas2)
        )}
    tiempo_fin_embeddings = time.perf_counter()

    # Comparación de temas
    tiempo_inicio_comparacion = time.perf_counter()
    temas_comunes = []
    for tema1, emb1 in temas1_emb.items():
        for tema2, emb2 in temas2_emb.items():
            sim = calcular_similitud(emb1, emb2)
            if sim > umbral:
                temas_comunes.append({
                    "tema_origen": tema1,
                    "tema_destino": tema2,
                    "tema_comun": f"{tema1} / {tema2}"
                })
    tiempo_fin_comparacion = time.perf_counter()

    tiempo_fin = time.perf_counter()

    print(f"\n⏱ [identificar_temas_comunes] Tiempos parciales:")
    print(f"  - División temas: {tiempo_fin_division - tiempo_inicio_division:.6f} segundos")
    print(f"  - Obtención embeddings: {tiempo_fin_embeddings - tiempo_inicio_embeddings:.6f} segundos")
    print(f"  - Comparación temas: {tiempo_fin_comparacion - tiempo_inicio_comparacion:.6f} segundos")
    print(f"⏱ [identificar_temas_comunes] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")
    print(f"  - Temas encontrados: {len(temas_comunes)}")

    return temas_comunes


def comparar_unidades(unidad_origen, unidad_destino):
    tiempo_inicio = time.perf_counter()

    titulo_origen = unidad_origen.get("titulo", "")
    titulo_destino = unidad_destino.get("titulo", "")
    aprendizajes_origen = unidad_origen.get("aprendizajes", "")
    aprendizajes_destino = unidad_destino.get("aprendizajes", "")
    temas_origen = unidad_origen.get("temas", "")
    temas_destino = unidad_destino.get("temas", "")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        titulo_future = executor.submit(calcular_similitud_seccion, titulo_origen, titulo_destino)
        aprendizaje_future = executor.submit(calcular_similitud_seccion, aprendizajes_origen, aprendizajes_destino)
        temas_future = executor.submit(calcular_similitud_seccion, temas_origen, temas_destino)
        temas_comunes_future = executor.submit(identificar_temas_comunes, temas_origen, temas_destino)

        similitud_titulo = titulo_future.result()
        similitud_aprendizaje = aprendizaje_future.result()
        similitud_temas = temas_future.result()
        temas_comunes = temas_comunes_future.result()

    tiempo_fin = time.perf_counter()
    print(f"\n⏱ [comparar_unidades] Tiempo total comparación: {tiempo_fin - tiempo_inicio:.6f} segundos")

    return {
        "id_unidad_origen": unidad_origen.get("idUnidad"),
        "id_unidad_destino": unidad_destino.get("idUnidad"),
        "similitud_titulo": similitud_titulo,
        "similitud_aprendizaje": similitud_aprendizaje,
        "similitud_temas": similitud_temas,
        "temas_comunes": temas_comunes
    }


def calcular_similitud_seccion(texto1, texto2):
    tiempo_inicio = time.perf_counter()

    if not texto1 or not texto2:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [calcular_similitud_seccion] Texto vacío: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return 0.0

    tiempo_inicio_procesamiento = time.perf_counter()
    texto1_norm, _ = procesar_texto(texto1)
    texto2_norm, _ = procesar_texto(texto2)
    tiempo_fin_procesamiento = time.perf_counter()

    tiempo_inicio_embeddings = time.perf_counter()
    emb1 = obtener_embeddings_bert(texto1_norm)
    emb2 = obtener_embeddings_bert(texto2_norm)
    tiempo_fin_embeddings = time.perf_counter()

    tiempo_inicio_similitud = time.perf_counter()
    similitud = float(calcular_similitud(emb1, emb2))
    tiempo_fin_similitud = time.perf_counter()

    tiempo_fin = time.perf_counter()

    print(f"\n⏱ [calcular_similitud_seccion] Tiempos parciales:")
    print(f"  - Procesamiento texto: {tiempo_fin_procesamiento - tiempo_inicio_procesamiento:.6f} segundos")
    print(f"  - Obtención embeddings: {tiempo_fin_embeddings - tiempo_inicio_embeddings:.6f} segundos")
    print(f"  - Cálculo similitud: {tiempo_fin_similitud - tiempo_inicio_similitud:.6f} segundos")
    print(f"⏱ [calcular_similitud_seccion] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")
    print(f"  - Similitud calculada: {similitud:.4f}")

    return similitud


def calcular_similitud_unidades(unidades1, unidades2):
    tiempo_inicio = time.perf_counter()

    if not unidades1 or not unidades2:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [calcular_similitud_unidades] Unidades vacías: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return {
            "unidades_emparejadas": [],
            "unidades_sin_par_origen": unidades1,
            "unidades_sin_par_destino": unidades2,
            "similitud_global": 0.0,
            "porcentaje_emparejamiento_unidades": 0.0
        }

    tiempo_inicio_comparaciones = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        combinaciones = [(u1, u2) for u1 in unidades1 for u2 in unidades2]
        resultados = list(executor.map(lambda pair: comparar_unidades(*pair), combinaciones))
    tiempo_fin_comparaciones = time.perf_counter()

    tiempo_inicio_procesamiento = time.perf_counter()
    similitudes = np.array(
        [r["similitud_titulo"] * 0.3 + r["similitud_aprendizaje"] * 0.4 + r["similitud_temas"] * 0.3 for r in
         resultados])
    matriz_similitud = similitudes.reshape((len(unidades1), len(unidades2)))

    unidades_emparejadas = []
    unidades_origen_usadas = set()
    unidades_destino_usadas = set()

    emparejamientos_posibles = sorted(
        [(i, j, matriz_similitud[i, j]) for i in range(len(unidades1)) for j in range(len(unidades2))],
        key=lambda x: -x[2]
    )

    for i, j, sim in emparejamientos_posibles:
        if sim < 0.5:
            break
        if i not in unidades_origen_usadas and j not in unidades_destino_usadas:
            idx = i * len(unidades2) + j
            unidades_emparejadas.append(resultados[idx])
            unidades_origen_usadas.add(i)
            unidades_destino_usadas.add(j)

    unidades_sin_par_origen = [
        {"id_unidad": u["idUnidad"], "temas": u.get("temas", "")}
        for i, u in enumerate(unidades1) if i not in unidades_origen_usadas
    ]

    unidades_sin_par_destino = [
        {"id_unidad": u["idUnidad"], "temas": u.get("temas", "")}
        for j, u in enumerate(unidades2) if j not in unidades_destino_usadas
    ]

    similitud_global = np.mean(
        [u["similitud_titulo"] * 0.3 + u["similitud_aprendizaje"] * 0.4 + u["similitud_temas"] * 0.3
         for u in unidades_emparejadas]) if unidades_emparejadas else 0.0
    tiempo_fin_procesamiento = time.perf_counter()

    tiempo_fin = time.perf_counter()

    print(f"\n⏱ [calcular_similitud_unidades] Tiempos parciales:")
    print(f"  - Comparaciones unidades: {tiempo_fin_comparaciones - tiempo_inicio_comparaciones:.6f} segundos")
    print(f"  - Procesamiento resultados: {tiempo_fin_procesamiento - tiempo_inicio_procesamiento:.6f} segundos")
    print(f"⏱ [calcular_similitud_unidades] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")
    print(f"  - Unidades emparejadas: {len(unidades_emparejadas)}")
    print(f"  - Similitud global: {similitud_global:.4f}")

    return {
        "unidades_emparejadas": unidades_emparejadas,
        "unidades_sin_par_origen": unidades_sin_par_origen,
        "unidades_sin_par_destino": unidades_sin_par_destino,
        "similitud_global": float(similitud_global),
        "porcentaje_emparejamiento_unidades": len(unidades_emparejadas) / max(len(unidades1), len(unidades2))
    }


def calcular_similitud_bibliografia(biblio1, biblio2):
    tiempo_inicio = time.perf_counter()

    if not biblio1 or not biblio2:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [calcular_similitud_bibliografia] Bibliografía vacía: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return 0.0

    tiempo_inicio_procesamiento = time.perf_counter()
    textos1 = [ref.get("referencia", "") for ref in biblio1]
    textos2 = [ref.get("referencia", "") for ref in biblio2]
    texto_combinado1 = " ".join(textos1)
    texto_combinado2 = " ".join(textos2)
    tiempo_fin_procesamiento = time.perf_counter()

    tiempo_inicio_similitud = time.perf_counter()
    similitud = calcular_similitud_seccion(texto_combinado1, texto_combinado2)
    tiempo_fin_similitud = time.perf_counter()

    tiempo_fin = time.perf_counter()

    print(f"\n⏱ [calcular_similitud_bibliografia] Tiempos parciales:")
    print(f"  - Procesamiento textos: {tiempo_fin_procesamiento - tiempo_inicio_procesamiento:.6f} segundos")
    print(f"  - Cálculo similitud: {tiempo_fin_similitud - tiempo_inicio_similitud:.6f} segundos")
    print(f"⏱ [calcular_similitud_bibliografia] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")
    print(f"  - Similitud calculada: {similitud:.4f}")

    return similitud


# ===============================================
# 📌 Endpoints
# ===============================================

@app.route('/comparar_cursos', methods=['POST'])
def comparar_cursos():
    try:
        tiempo_inicio = time.perf_counter()

        # Limpieza de cachés
        tiempo_inicio_limpieza = time.perf_counter()
        with cache_lock:
            if len(embedding_cache) > 2000:
                embedding_cache.clear()
            if len(text_processing_cache) > 2000:
                text_processing_cache.clear()
        tiempo_fin_limpieza = time.perf_counter()

        # Obtención de datos
        tiempo_inicio_datos = time.perf_counter()
        data = request.get_json()
        comparaciones = data.get("comparaciones", [])
        tiempo_fin_datos = time.perf_counter()

        resultados = []
        tiempo_total_inicio = time.time()

        for comparacion in comparaciones:
            tiempo_comparacion_inicio = time.time()
            print(f"\n🔍 Iniciando comparación de cursos...")

            curso_origen = comparacion.get("cursoOrigen", {})
            curso_destino = comparacion.get("cursoDestino", {})
            silabo_origen = curso_origen.get("silabo", {})
            silabo_destino = curso_destino.get("silabo", {})

            # Procesamiento en paralelo
            tiempo_inicio_paralelo = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                sumilla_future = executor.submit(
                    calcular_similitud_seccion,
                    silabo_origen.get("sumilla", ""),
                    silabo_destino.get("sumilla", "")
                )
                aprendizajes_future = executor.submit(
                    calcular_similitud_seccion,
                    silabo_origen.get("aprendizaje_general", ""),
                    silabo_destino.get("aprendizaje_general", "")
                )
                unidades_future = executor.submit(
                    calcular_similitud_unidades,
                    silabo_origen.get("unidades", []),
                    silabo_destino.get("unidades", [])
                )
                biblio_future = executor.submit(
                    calcular_similitud_bibliografia,
                    silabo_origen.get("bibliografias", []),
                    silabo_destino.get("bibliografias", [])
                )

                similitud_sumilla = sumilla_future.result()
                similitud_aprendizajes = aprendizajes_future.result()
                resultado_unidades = unidades_future.result()
                similitud_bibliografia = biblio_future.result()
            tiempo_fin_paralelo = time.perf_counter()

            tiempo_comparacion = (time.time() - tiempo_comparacion_inicio) * 1000

            resultado = {
                "idCursoOrigen": curso_origen.get("idCurso"),
                "idCursoDestino": curso_destino.get("idCurso"),
                "nombreCursoOrigen": curso_origen.get("nombre", ""),
                "nombreCursoDestino": curso_destino.get("nombre", ""),
                "tiempo_procesamiento_ms": tiempo_comparacion,
                "resultado_resumido": {
                    "similitud_sumilla": similitud_sumilla,
                    "similitud_aprendizajes": similitud_aprendizajes,
                    "similitud_unidades": resultado_unidades["similitud_global"],
                    "similitud_bibliografia": similitud_bibliografia,
                    "porcentaje_emparejamiento_unidades": resultado_unidades["porcentaje_emparejamiento_unidades"]
                },
                "resultado_detallado": {
                    "unidades_emparejadas": [
                        {
                            "id_unidad_origen": u["id_unidad_origen"],
                            "id_unidad_destino": u["id_unidad_destino"],
                            "similitud_titulo": u["similitud_titulo"],
                            "similitud_aprendizaje": u["similitud_aprendizaje"],
                            "similitud_temas": u["similitud_temas"],
                            "temas_comunes": u["temas_comunes"]
                        } for u in resultado_unidades["unidades_emparejadas"]
                    ],
                    "unidades_sin_par_origen": resultado_unidades["unidades_sin_par_origen"],
                    "unidades_sin_par_destino": resultado_unidades["unidades_sin_par_destino"]
                }
            }
            resultados.append(resultado)

            print(f"\n✅ Comparación completada en {tiempo_comparacion:.2f} ms")
            print(f"  - Similitud sumilla: {similitud_sumilla:.4f}")
            print(f"  - Similitud aprendizajes: {similitud_aprendizajes:.4f}")
            print(f"  - Similitud unidades: {resultado_unidades['similitud_global']:.4f}")
            print(f"  - Similitud bibliografía: {similitud_bibliografia:.4f}")

        tiempo_total = (time.time() - tiempo_total_inicio) * 1000
        tiempo_fin = time.perf_counter()

        print(f"\n⏱ [comparar_cursos] Tiempos parciales:")
        print(f"  - Limpieza cachés: {tiempo_fin_limpieza - tiempo_inicio_limpieza:.6f} segundos")
        print(f"  - Obtención datos: {tiempo_fin_datos - tiempo_inicio_datos:.6f} segundos")
        print(f"  - Procesamiento paralelo: {tiempo_fin_paralelo - tiempo_inicio_paralelo:.6f} segundos")
        print(f"⏱ [comparar_cursos] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")
        print(f"📊 Comparaciones realizadas: {len(comparaciones)}")
        print(f"⏱ Tiempo total de procesamiento: {tiempo_total:.2f} ms")

        return jsonify({
            "status": "success",
            "tiempo_procesamiento_total_ms": tiempo_total,
            "comparaciones": resultados
        })

    except Exception as e:
        tiempo_fin = time.perf_counter()
        print(f"\n❌ Error en comparar_cursos: {str(e)}")
        print(f"⏱ Tiempo hasta error: {tiempo_fin - tiempo_inicio:.6f} segundos")
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "error_message": str(e),
            "stack_trace": traceback.format_exc()
        }), 500


@app.route('/busqueda_semantica', methods=['POST'])
def busqueda_semantica():
    try:
        tiempo_inicio = time.perf_counter()

        # Limpieza de cachés
        with cache_lock:
            if len(embedding_cache) > 2000:
                embedding_cache.clear()
            if len(text_processing_cache) > 2000:
                text_processing_cache.clear()

        data = request.get_json()
        comparaciones = data.get("comparaciones", [])
        id_grupo_tematico = data.get("id_grupo_tematico")

        resultados = []

        for comparacion in comparaciones:
            tiempo_comparacion_inicio = time.perf_counter()

            curso_origen = comparacion.get("cursoOrigen", {})
            curso_destino = comparacion.get("cursoDestino", {})
            silabo_origen = curso_origen.get("silabo", {})
            silabo_destino = curso_destino.get("silabo", {})

            # Preparar texto contextual completo
            texto_origen = f"{curso_origen.get('nombre', '')} {silabo_origen.get('sumilla', '')}"
            texto_destino = f"{curso_destino.get('nombre', '')} {silabo_destino.get('sumilla', '')}"

            # Procesamiento en paralelo con contexto mejorado
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Obtener embeddings BERT para texto completo
                future_embeddings = executor.submit(
                    obtener_embeddings_bert_contextual,
                    texto_origen,
                    texto_destino
                )

                # Extraer términos clave mejorados
                future_terminos = executor.submit(
                    extraer_terminos_clave_avanzado,
                    texto_origen,
                    texto_destino
                )

                # Obtener resultados
                embeddings_result = future_embeddings.result()
                terminos_result = future_terminos.result()

            # Calcular similitud contextual
            similitud_contextual = calcular_similitud_contextual(
                embeddings_result,
                terminos_result
            )

            # Filtro de calidad mejorado
            if similitud_contextual < 0.4:
                continue

            tiempo_comparacion = float(round(time.perf_counter() - tiempo_comparacion_inicio, 2))

            resultado = {
                "cursoOrigen": {
                    "idCurso": int(curso_origen.get("idCurso", 0)),
                    "nombre": str(curso_origen.get("nombre", "")),
                    "sumilla": str(silabo_origen.get("sumilla", ""))
                },
                "cursoDestino": {
                    "idCurso": int(curso_destino.get("idCurso", 0)),
                    "nombre": str(curso_destino.get("nombre", "")),
                    "sumilla": str(silabo_destino.get("sumilla", ""))
                },
                "resultado_resumido": {
                    "similitud_global": similitud_contextual,
                    "terminos_comunes": terminos_result.get("terminos_comunes", []),
                    "contexto_compartido": terminos_result.get("contexto_compartido", []),
                    "ponderacion_contextual": terminos_result.get("ponderacion_contextual", 0.0)
                },
                "tiempo_procesamiento_s": tiempo_comparacion,
                "id_grupo_tematico": int(id_grupo_tematico) if id_grupo_tematico else None
            }
            resultados.append(resultado)

        # Ordenar resultados
        resultados_ordenados = sorted(
            resultados,
            key=lambda x: -x["resultado_resumido"]["similitud_global"]
        )

        tiempo_total = float(round(time.perf_counter() - tiempo_inicio, 2))

        return jsonify({
            "status": "success",
            "tiempo_procesamiento_total_s": tiempo_total,
            "comparaciones": resultados_ordenados
        })

    except Exception as e:
        print(f"\n❌ Error en busqueda_semantica: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error_message": str(e),
            "stack_trace": traceback.format_exc()
        }), 500

def obtener_embeddings_bert_contextual(texto1, texto2):
    """
    Genera embeddings BERT contextuales mejorados para dos textos.
    """
    # Combinar textos para contexto cruzado
    texto_combinado = f"{texto1} [SEP] {texto2}"

    # Obtener embeddings
    inputs = tokenizer(
        texto_combinado,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Extraer embeddings de la última capa
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

    return {
        "embeddings": embeddings,
        "contexto_cruzado": True
    }

def extraer_terminos_clave_avanzado(texto1, texto2):
    """
    Extrae términos clave y contexto compartido entre dos textos.
    """
    # Tokenización básica
    palabras1 = set(re.findall(r'\w+', texto1.lower()))
    palabras2 = set(re.findall(r'\w+', texto2.lower()))

    # Filtrar stopwords y palabras cortas
    palabras1 = {p for p in palabras1 if p not in stopwords_personalizadas and len(p) > 3}
    palabras2 = {p for p in palabras2 if p not in stopwords_personalizadas and len(p) > 3}

    # Encontrar términos comunes
    terminos_comunes = list(palabras1 & palabras2)

    # Calcular ponderación contextual
    ponderacion_contextual = len(terminos_comunes) / max(len(palabras1), len(palabras2))

    return {
        "terminos_comunes": terminos_comunes[:15],  # Aumentar límite a 15 términos
        "total_terminos1": len(palabras1),
        "total_terminos2": len(palabras2),
        "ponderacion_contextual": ponderacion_contextual
    }


def calcular_similitud_contextual(embeddings_result, terminos_result):
    """
    Calcula la similitud contextual entre dos textos usando embeddings BERT
    y análisis de términos clave.
    """
    # Calcular similitud de embeddings
    similitud_embeddings = float(cosine_similarity(
        [embeddings_result["embeddings"]],
        [embeddings_result["embeddings"]]
    )[0][0])

    # Calcular similitud basada en términos
    similitud_terminos = terminos_result["ponderacion_contextual"]

    # Combinar similitudes con pesos contextuales
    similitud_contextual = (
            similitud_embeddings * 0.6 +  # Mayor peso a embeddings
            similitud_terminos * 0.4  # Peso a términos comunes
    )

    return float(round(similitud_contextual, 2))



@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "operativo",
        "cache_embeddings": len(embedding_cache),
        "cache_procesamiento": len(text_processing_cache),
        "device": str(device)
    })


tiempo_fin_setup = time.perf_counter()
tiempo_total_setup = tiempo_fin_setup - tiempo_inicio_script

print(f"\n⏱ Tiempo total de configuración inicial: {tiempo_total_setup:.4f} segundos")
print("✅ Servicio listo para recibir peticiones\n")

if __name__ == '__main__':
    app.run(port=5000, threaded=True)