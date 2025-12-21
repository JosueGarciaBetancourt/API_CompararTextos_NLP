# PRIMERO EJECUTAR EN TERMINAL: python spacy_sklearn_bert.py
# SEGUNDO EJECUTAR EN OTRA TERMINAL: python test_api.py

import time
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from collections import defaultdict
import concurrent.futures
from functools import lru_cache
from threading import Lock
import traceback
import spacy
import unicodedata
import re

app = Flask(__name__)

# ===============================================
# 📌 CONSTANTES CONFIGURABLES
# ===============================================

# Pesos para el cálculo de similitud de unidades
PESO_TITULO_UNIDAD = 0.20
PESO_APRENDIZAJE_UNIDAD = 0.10
PESO_TEMAS_UNIDAD = 0.70

# Umbrales
UMBRAL_TEMAS_COMUNES = 0.45  # Mínima similitud para considerar temas comunes
UMBRAL_EMPAREJAMIENTO_UNIDADES = 0.50  # Mínima similitud para emparejar unidades

tiempo_inicio_script = time.perf_counter()

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

def convert_floats(obj):
    """Convierte todos los valores numpy a tipos nativos de Python para serialización JSON."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_floats(item) for item in obj]
    return obj

@lru_cache(maxsize=5000)
def normalizar_texto(texto):
    tiempo_inicio = time.perf_counter()

    if not texto:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [normalizar_texto] Texto vacío: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return ""

    texto = str(texto).lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode("utf-8")
    texto = clean_pattern.sub('', texto)
    resultado = space_pattern.sub(' ', texto).strip()

    tiempo_fin = time.perf_counter()
    print(f"⏱ [normalizar_texto] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")

    return resultado

def obtener_embeddings_bert(texto):
    tiempo_inicio = time.perf_counter()

    with cache_lock:
        if texto in embedding_cache:
            tiempo_fin = time.perf_counter()
            print(f"⏱ [obtener_embeddings_bert] Embedding desde caché: {tiempo_fin - tiempo_inicio:.6f} segundos")
            return embedding_cache[texto]

    inputs = tokenizer(
        texto,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

    with cache_lock:
        embedding_cache[texto] = embeddings

    tiempo_fin = time.perf_counter()
    print(f"⏱ [obtener_embeddings_bert] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")

    return embeddings

def calcular_similitud(vector1, vector2):
    tiempo_inicio = time.perf_counter()
    resultado = float(cosine_similarity([vector1], [vector2])[0][0])
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

    texto_norm = normalizar_texto(texto)
    doc = nlp(texto_norm)
    tokens = [token for token in doc if (
            not token.is_stop and
            not token.is_punct and
            not token.is_digit and
            token.text not in stopwords_personalizadas
    )]

    result = (texto_norm, tokens)
    with cache_lock:
        text_processing_cache[cache_key] = result

    tiempo_fin = time.perf_counter()
    print(f"⏱ [procesar_texto] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")

    return result

# ===============================================
# 📌 Funciones principales mejoradas
# ===============================================

def identificar_temas_comunes(texto1, texto2, umbral=UMBRAL_TEMAS_COMUNES):
    tiempo_inicio = time.perf_counter()

    temas1 = [tema.strip() for tema in split_pattern.split(texto1) if tema.strip()]
    temas2 = [tema.strip() for tema in split_pattern.split(texto2) if tema.strip()]

    if not temas1 or not temas2:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [identificar_temas_comunes] Sin temas para comparar: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        temas1_emb = {tema: emb for tema, emb in zip(
            temas1,
            executor.map(obtener_embeddings_bert, temas1)
        )}
        temas2_emb = {tema: emb for tema, emb in zip(
            temas2,
            executor.map(obtener_embeddings_bert, temas2)
        )}

    temas_comunes = []
    for tema1, emb1 in temas1_emb.items():
        for tema2, emb2 in temas2_emb.items():
            sim = calcular_similitud(emb1, emb2)
            if sim > umbral:
                temas_comunes.append({
                    "tema_origen": tema1,
                    "tema_destino": tema2,
                    "tema_comun": f"{tema1} / {tema2}",
                    "similitud": float(sim)
                })

    tiempo_fin = time.perf_counter()
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

    # Calcular similitud ponderada usando las constantes
    similitud_ponderada = (
        similitud_titulo * PESO_TITULO_UNIDAD +
        similitud_aprendizaje * PESO_APRENDIZAJE_UNIDAD +
        similitud_temas * PESO_TEMAS_UNIDAD
    )

    tiempo_fin = time.perf_counter()
    print(f"\n⏱ [comparar_unidades] Tiempo total comparación: {tiempo_fin - tiempo_inicio:.6f} segundos")

    return {
        "id_unidad_origen": unidad_origen.get("idUnidad"),
        "id_unidad_destino": unidad_destino.get("idUnidad"),
        "similitud_titulo": float(similitud_titulo),
        "similitud_aprendizaje": float(similitud_aprendizaje),
        "similitud_temas": float(similitud_temas),
        "similitud_ponderada": float(similitud_ponderada),
        "temas_comunes": temas_comunes
    }

def calcular_similitud_seccion(texto1, texto2):
    tiempo_inicio = time.perf_counter()

    if not texto1 or not texto2:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [calcular_similitud_seccion] Texto vacío: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return 0.0

    texto1_norm, _ = procesar_texto(texto1)
    texto2_norm, _ = procesar_texto(texto2)

    emb1 = obtener_embeddings_bert(texto1_norm)
    emb2 = obtener_embeddings_bert(texto2_norm)

    similitud = float(calcular_similitud(emb1, emb2))

    tiempo_fin = time.perf_counter()
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

    with concurrent.futures.ThreadPoolExecutor() as executor:
        combinaciones = [(u1, u2) for u1 in unidades1 for u2 in unidades2]
        resultados = list(executor.map(lambda pair: comparar_unidades(*pair), combinaciones))

    # Usar similitud_ponderada en lugar del cálculo manual
    similitudes = np.array([r["similitud_ponderada"] for r in resultados])
    matriz_similitud = similitudes.reshape((len(unidades1), len(unidades2)))

    unidades_emparejadas = []
    unidades_origen_usadas = set()
    unidades_destino_usadas = set()

    emparejamientos_posibles = sorted(
        [(i, j, matriz_similitud[i, j]) for i in range(len(unidades1)) for j in range(len(unidades2))],
        key=lambda x: -x[2]
    )

    for i, j, sim in emparejamientos_posibles:
        if sim < UMBRAL_EMPAREJAMIENTO_UNIDADES:
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

    similitud_global = np.mean([u["similitud_ponderada"] for u in unidades_emparejadas]) if unidades_emparejadas else 0.0

    tiempo_fin = time.perf_counter()
    print(f"\n⏱ [calcular_similitud_unidades] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")
    print(f"  - Unidades emparejadas: {len(unidades_emparejadas)}")
    print(f"  - Similitud global: {similitud_global:.4f}")

    return {
        "unidades_emparejadas": unidades_emparejadas,
        "unidades_sin_par_origen": unidades_sin_par_origen,
        "unidades_sin_par_destino": unidades_sin_par_destino,
        "similitud_global": float(similitud_global),
        "porcentaje_emparejamiento_unidades": float(len(unidades_emparejadas) / max(len(unidades1), len(unidades2)))
    }

def calcular_similitud_bibliografia(biblio1, biblio2):
    tiempo_inicio = time.perf_counter()

    if not biblio1 or not biblio2:
        tiempo_fin = time.perf_counter()
        print(f"⏱ [calcular_similitud_bibliografia] Bibliografía vacía: {tiempo_fin - tiempo_inicio:.6f} segundos")
        return 0.0

    textos1 = [ref.get("referencia", "") for ref in biblio1]
    textos2 = [ref.get("referencia", "") for ref in biblio2]
    texto_combinado1 = " ".join(textos1)
    texto_combinado2 = " ".join(textos2)

    similitud = calcular_similitud_seccion(texto_combinado1, texto_combinado2)

    tiempo_fin = time.perf_counter()
    print(f"⏱ [calcular_similitud_bibliografia] Tiempo total: {tiempo_fin - tiempo_inicio:.6f} segundos")
    print(f"  - Similitud calculada: {similitud:.4f}")

    return float(similitud)

# ===============================================
# 📌 Endpoints
# ===============================================

@app.route('/comparar_cursos', methods=['POST'])
def comparar_cursos():
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

        resultados = []
        tiempo_total_inicio = time.time()

        for comparacion in comparaciones:
            tiempo_comparacion_inicio = time.time()
            print(f"\n🔍 Iniciando comparación de cursos...")

            curso_origen = comparacion.get("cursoOrigen", {})
            curso_destino = comparacion.get("cursoDestino", {})
            silabo_origen = curso_origen.get("silabo", {})
            silabo_destino = curso_destino.get("silabo", {})

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

            tiempo_comparacion = (time.time() - tiempo_comparacion_inicio) * 1000

            resultado = {
                "idCursoOrigen": curso_origen.get("idCurso"),
                "idCursoDestino": curso_destino.get("idCurso"),
                "nombreCursoOrigen": curso_origen.get("nombre", ""),
                "nombreCursoDestino": curso_destino.get("nombre", ""),
                "tiempo_procesamiento_ms": tiempo_comparacion,
                "resultado_resumido": {
                    "similitud_sumilla": float(similitud_sumilla),
                    "similitud_aprendizajes": float(similitud_aprendizajes),
                    "similitud_unidades": float(resultado_unidades["similitud_global"]),
                    "similitud_bibliografia": float(similitud_bibliografia),
                    "porcentaje_emparejamiento_unidades": float(resultado_unidades["porcentaje_emparejamiento_unidades"])
                },
                "resultado_detallado": convert_floats({
                    "unidades_emparejadas": [
                        {
                            "id_unidad_origen": u["id_unidad_origen"],
                            "id_unidad_destino": u["id_unidad_destino"],
                            "similitud_titulo": u["similitud_titulo"],
                            "similitud_aprendizaje": u["similitud_aprendizaje"],
                            "similitud_temas": u["similitud_temas"],
                            "similitud_ponderada": u["similitud_ponderada"],
                            "temas_comunes": u["temas_comunes"]
                        } for u in resultado_unidades["unidades_emparejadas"]
                    ],
                    "unidades_sin_par_origen": resultado_unidades["unidades_sin_par_origen"],
                    "unidades_sin_par_destino": resultado_unidades["unidades_sin_par_destino"]
                })
            }
            resultados.append(convert_floats(resultado))

            print(f"\n✅ Comparación completada en {tiempo_comparacion:.2f} ms")
            print(f"  - Similitud sumilla: {similitud_sumilla:.4f}")
            print(f"  - Similitud aprendizajes: {similitud_aprendizajes:.4f}")
            print(f"  - Similitud unidades: {resultado_unidades['similitud_global']:.4f}")
            print(f"  - Similitud bibliografía: {similitud_bibliografia:.4f}")

        tiempo_total = (time.time() - tiempo_total_inicio) * 1000
        tiempo_fin = time.perf_counter()

        print(f"\n📊 Comparaciones realizadas: {len(comparaciones)}")
        print(f"⏱ Tiempo total de procesamiento: {tiempo_total:.2f} ms")

        return jsonify(convert_floats({
            "status": "success",
            "tiempo_procesamiento_total_ms": float(tiempo_total),
            "comparaciones": resultados
        }))

    except Exception as e:
        tiempo_fin = time.perf_counter()
        print(f"\n❌ Error en comparar_cursos: {str(e)}")
        print(f"⏱ Tiempo hasta error: {tiempo_fin - tiempo_inicio:.6f} segundos")
        traceback.print_exc()

        return jsonify(convert_floats({
            "status": "error",
            "error_message": str(e),
            "stack_trace": traceback.format_exc()
        })), 500


@app.route('/busqueda_semantica', methods=['POST'])
def busqueda_semantica():
    try:
        tiempo_inicio = time.perf_counter()

        with cache_lock:
            if len(embedding_cache) > 2000:
                embedding_cache.clear()
            if len(text_processing_cache) > 2000:
                text_processing_cache.clear()

        data = request.get_json()
        comparaciones = data.get("comparaciones", [])

        resultados = []

        for comparacion in comparaciones:
            tiempo_comparacion_inicio = time.perf_counter()

            curso_origen = comparacion.get("cursoOrigen", {})
            curso_destino = comparacion.get("cursoDestino", {})
            silabo_origen = curso_origen.get("silabo", {})
            silabo_destino = curso_destino.get("silabo", {})

            texto_origen = f"{curso_origen.get('nombre', '')} {silabo_origen.get('sumilla', '')}"
            texto_destino = f"{curso_destino.get('nombre', '')} {silabo_destino.get('sumilla', '')}"

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_embeddings = executor.submit(
                    obtener_embeddings_bert_contextual,
                    texto_origen,
                    texto_destino
                )
                future_terminos = executor.submit(
                    extraer_terminos_clave_avanzado,
                    texto_origen,
                    texto_destino
                )

                embeddings_result = future_embeddings.result()
                terminos_result = future_terminos.result()

            similitud_contextual = calcular_similitud_contextual(
                embeddings_result,
                terminos_result
            )

            if similitud_contextual < 0.6:
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
                    "similitud_global": float(similitud_contextual),
                    "terminos_comunes": terminos_result.get("terminos_comunes", []),
                    "contexto_compartido": terminos_result.get("contexto_compartido", []),
                    "ponderacion_contextual": float(terminos_result.get("ponderacion_contextual", 0.0))
                },
                "tiempo_procesamiento_s": float(tiempo_comparacion)
            }
            resultados.append(convert_floats(resultado))

        resultados_ordenados = sorted(
            resultados,
            key=lambda x: -x["resultado_resumido"]["similitud_global"]
        )

        tiempo_total = float(round(time.perf_counter() - tiempo_inicio, 2))

        return jsonify(convert_floats({
            "status": "success",
            "tiempo_procesamiento_total_s": float(tiempo_total),
            "comparaciones": resultados_ordenados
        }))
    except Exception as e:
        print(f"\n❌ Error en busqueda_semantica: {str(e)}")
        traceback.print_exc()
        return jsonify(convert_floats({
            "status": "error",
            "error_message": str(e),
            "stack_trace": traceback.format_exc()
        })), 500

def obtener_embeddings_bert_contextual(texto1, texto2):
    texto_combinado = f"{texto1} [SEP] {texto2}"

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

    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

    return {
        "embeddings": embeddings,
        "contexto_cruzado": True
    }

def extraer_terminos_clave_avanzado(texto1, texto2):
    palabras1 = set(re.findall(r'\w+', texto1.lower()))
    palabras2 = set(re.findall(r'\w+', texto2.lower()))

    palabras1 = {p for p in palabras1 if p not in stopwords_personalizadas and len(p) > 3}
    palabras2 = {p for p in palabras2 if p not in stopwords_personalizadas and len(p) > 3}

    terminos_comunes = list(palabras1 & palabras2)
    ponderacion_contextual = len(terminos_comunes) / max(len(palabras1), len(palabras2))

    return {
        "terminos_comunes": terminos_comunes[:15],
        "total_terminos1": len(palabras1),
        "total_terminos2": len(palabras2),
        "ponderacion_contextual": float(ponderacion_contextual)
    }

def calcular_similitud_contextual(embeddings_result, terminos_result):
    similitud_embeddings = float(cosine_similarity(
        [embeddings_result["embeddings"]],
        [embeddings_result["embeddings"]]
    )[0][0])

    similitud_terminos = terminos_result["ponderacion_contextual"]

    similitud_contextual = (
            similitud_embeddings * 0.6 +
            similitud_terminos * 0.4
    )

    return float(round(similitud_contextual, 2))

@app.route('/status', methods=['GET'])
def status():
    return jsonify(convert_floats({
        "status": "operativo",
        "cache_embeddings": len(embedding_cache),
        "cache_procesamiento": len(text_processing_cache),
        "device": str(device),
        "constantes_activas": {
            "PESO_TITULO_UNIDAD": float(PESO_TITULO_UNIDAD),
            "PESO_APRENDIZAJE_UNIDAD": float(PESO_APRENDIZAJE_UNIDAD),
            "PESO_TEMAS_UNIDAD": float(PESO_TEMAS_UNIDAD),
            "UMBRAL_TEMAS_COMUNES": float(UMBRAL_TEMAS_COMUNES),
            "UMBRAL_EMPAREJAMIENTO_UNIDADES": float(UMBRAL_EMPAREJAMIENTO_UNIDADES)
        }
    }))

tiempo_fin_setup = time.perf_counter()
tiempo_total_setup = tiempo_fin_setup - tiempo_inicio_script

print(f"\n⏱ Tiempo total de configuración inicial: {tiempo_total_setup:.4f} segundos")
print("✅ Servicio listo para recibir peticiones\n")

if __name__ == '__main__':
    app.run(port=5000, threaded=True)
