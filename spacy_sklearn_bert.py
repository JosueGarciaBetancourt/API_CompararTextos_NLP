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

# Pesos
PESO_SIMILITUD_COMPARACION_SEMANTICA_BERT = 0.80
PESO_SIMILITUD_COMPARACION_SEMANTICA_JACCARD = 0.20

PESO_TITULO_UNIDAD = 0.10
PESO_APRENDIZAJE_UNIDAD = 0.20
PESO_TEMAS_UNIDAD = 0.70

PESO_SIMILITUD_BUSQUEDA_SEMANTICA_BERT = 0.70
PESO_SIMILITUD_BUSQUEDA_SEMANTICA_JACCARD = 0.30

# Umbrales
UMBRAL_SIMILITUD_BUSQUEDA_SEMANTICA = 0.50 # Mínima similitud para considerar comparación
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
print(f"🖥️  Dispositivo de inferencia: {device}")

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
# 📌 Funciones de Utilidad y Caché
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
    if not texto:
        return ""

    texto = str(texto).lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode("utf-8")
    texto = clean_pattern.sub('', texto)
    resultado = space_pattern.sub(' ', texto).strip()

    return resultado


# ===============================================
# 📌 Funciones de Embeddings (Individual y Batch)
# ===============================================

def obtener_embeddings_bert(texto):
    """Obtiene el embedding de un solo texto (usado en /comparar_cursos)"""
    with cache_lock:
        if texto in embedding_cache:
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

    return embeddings


def obtener_embeddings_batch(textos):
    """
    Verdadero Tensor Batching: Procesa un array de textos simultáneamente en PyTorch.
    (Usado para búsquedas masivas en /busqueda_semantica)
    """
    if not textos:
        return {}

    resultados = {}
    textos_a_procesar = []

    # 1. Revisar qué textos ya están en la memoria caché
    with cache_lock:
        for texto in textos:
            if texto in embedding_cache:
                resultados[texto] = embedding_cache[texto]
            else:
                textos_a_procesar.append(texto)

    if not textos_a_procesar:
        return resultados

    # 2. Procesar los faltantes en lotes de 32 para no saturar la RAM/VRAM
    batch_size = 32
    for i in range(0, len(textos_a_procesar), batch_size):
        batch_textos = textos_a_procesar[i:i + batch_size]

        inputs = tokenizer(
            batch_textos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Extraer token CLS de todo el lote de una vez
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Guardar en caché y en el diccionario de resultados
        with cache_lock:
            for texto, emb in zip(batch_textos, batch_embeddings):
                embedding_cache[texto] = emb
                resultados[texto] = emb

    return resultados


def extraer_terminos_clave_avanzado(texto1, texto2):
    """Extracción rápida de términos en común basada en conjuntos (Jaccard)"""
    palabras1 = set(re.findall(r'\w+', str(texto1).lower()))
    palabras2 = set(re.findall(r'\w+', str(texto2).lower()))

    palabras1 = {p for p in palabras1 if p not in stopwords_personalizadas and len(p) > 1}
    palabras2 = {p for p in palabras2 if p not in stopwords_personalizadas and len(p) > 1}

    terminos_comunes = list(palabras1 & palabras2)
    ponderacion_contextual = len(terminos_comunes) / max(len(palabras1), len(palabras2)) if max(len(palabras1),
                                                                                                len(palabras2)) > 0 else 0

    return {
        "terminos_comunes": terminos_comunes[:15],
        "ponderacion_contextual": float(ponderacion_contextual)
    }


# ===============================================
# 📌 Funciones para Evaluación Detallada (/comparar_cursos)
# ===============================================

def calcular_similitud(vector1, vector2):
    return float(cosine_similarity([vector1], [vector2])[0][0])


def procesar_texto(texto):
    cache_key = hash(texto)
    with cache_lock:
        if cache_key in text_processing_cache:
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
    return result


def identificar_temas_comunes(texto1, texto2, umbral=UMBRAL_TEMAS_COMUNES):
    temas1 = [tema.strip() for tema in split_pattern.split(texto1) if tema.strip()]
    temas2 = [tema.strip() for tema in split_pattern.split(texto2) if tema.strip()]

    if not temas1 or not temas2:
        return []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        temas1_emb = {tema: emb for tema, emb in zip(temas1, executor.map(obtener_embeddings_bert, temas1))}
        temas2_emb = {tema: emb for tema, emb in zip(temas2, executor.map(obtener_embeddings_bert, temas2))}

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
    return temas_comunes


"""def calcular_similitud_seccion(texto1, texto2):
    if not texto1 or not texto2:
        return 0.0
    texto1_norm, _ = procesar_texto(texto1)
    texto2_norm, _ = procesar_texto(texto2)
    emb1 = obtener_embeddings_bert(texto1_norm)
    emb2 = obtener_embeddings_bert(texto2_norm)
    return float(calcular_similitud(emb1, emb2))"""


"""def calcular_similitud_seccion(texto1, texto2):
    if not texto1 or not texto2:
        return 0.0

    texto1_norm, _ = procesar_texto(texto1)
    texto2_norm, _ = procesar_texto(texto2)

    # 1. Similitud Semántica (BERT - El contexto)
    emb1 = obtener_embeddings_bert(texto1_norm)
    emb2 = obtener_embeddings_bert(texto2_norm)
    sim_bert = float(calcular_similitud(emb1, emb2))

    # 2. Similitud Léxica (Jaccard - Palabras clave exactas)
    terminos_result = extraer_terminos_clave_avanzado(texto1_norm, texto2_norm)
    sim_jaccard = float(terminos_result["ponderacion_contextual"])

    # 3. Fusión Híbrida Estricta (Usa tus pesos de 50/50)
    sim_final = (sim_bert * PESO_SIMILITUD_COMPARACION_SEMANTICA_BERT) + (
                sim_jaccard * PESO_SIMILITUD_COMPARACION_SEMANTICA_JACCARD)

    return float(sim_final)"""


def calcular_similitud_seccion(texto1, texto2):
    if not texto1 or not texto2:
        return 0.0

    texto1_norm, _ = procesar_texto(texto1)
    texto2_norm, _ = procesar_texto(texto2)

    # 1. Similitud Semántica (BERT - Entiende sinónimos y contexto)
    emb1 = obtener_embeddings_bert(texto1_norm)
    emb2 = obtener_embeddings_bert(texto2_norm)
    sim_bert = float(calcular_similitud(emb1, emb2))

    # 2. Similitud Léxica (Jaccard - Solo palabras exactas)
    terminos_result = extraer_terminos_clave_avanzado(texto1_norm, texto2_norm)
    sim_jaccard = float(terminos_result["ponderacion_contextual"])

    # 3. COMPUERTA LÉXICA (Penalización inteligente)
    # Si BERT dice que hay similitud, pero NO comparten vocabulario técnico (Jaccard casi nulo):
    if sim_bert > 0.40 and sim_jaccard < 0.03:
        # Castigo: reducimos la confianza de BERT
        sim_final = sim_bert * 0.50
    else:
        # Cálculo normal usando tus constantes configurables
        sim_final = (sim_bert * PESO_SIMILITUD_COMPARACION_SEMANTICA_BERT) + (
                    sim_jaccard * PESO_SIMILITUD_COMPARACION_SEMANTICA_JACCARD)

    return float(min(sim_final, 1.0))

def comparar_unidades(unidad_origen, unidad_destino):
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

    similitud_ponderada = (
            similitud_titulo * PESO_TITULO_UNIDAD +
            similitud_aprendizaje * PESO_APRENDIZAJE_UNIDAD +
            similitud_temas * PESO_TEMAS_UNIDAD
    )

    return {
        "id_unidad_origen": unidad_origen.get("idUnidad"),
        "id_unidad_destino": unidad_destino.get("idUnidad"),
        "similitud_titulo": float(similitud_titulo),
        "similitud_aprendizaje": float(similitud_aprendizaje),
        "similitud_temas": float(similitud_temas),
        "similitud_ponderada": float(similitud_ponderada),
        "temas_comunes": temas_comunes
    }


def calcular_similitud_unidades(unidades1, unidades2):
    if not unidades1 or not unidades2:
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

    similitud_global = np.mean(
        [u["similitud_ponderada"] for u in unidades_emparejadas]) if unidades_emparejadas else 0.0

    return {
        "unidades_emparejadas": unidades_emparejadas,
        "unidades_sin_par_origen": unidades_sin_par_origen,
        "unidades_sin_par_destino": unidades_sin_par_destino,
        "similitud_global": float(similitud_global),
        "porcentaje_emparejamiento_unidades": float(len(unidades_emparejadas) / max(len(unidades1), len(unidades2)))
    }


def calcular_similitud_bibliografia(biblio1, biblio2):
    if not biblio1 or not biblio2:
        return 0.0
    textos1 = [ref.get("referencia", "") for ref in biblio1]
    textos2 = [ref.get("referencia", "") for ref in biblio2]
    texto_combinado1 = " ".join(textos1)
    texto_combinado2 = " ".join(textos2)
    return float(calcular_similitud_seccion(texto_combinado1, texto_combinado2))


# ===============================================
# 📌 ENDPOINTS DE LA API
# ===============================================

@app.route('/busqueda_semantica', methods=['POST'])
def busqueda_semantica():
    """
    ENDPOINT OPTIMIZADO PARA BATCHING DE TENSOSRES
    Recibe múltiples cursos y busca coincidencias en lote cruzando origen vs destino.
    """
    try:
        tiempo_inicio = time.perf_counter()

        # Limpieza periódica de cachés si se llenan mucho
        with cache_lock:
            if len(embedding_cache) > 2000:
                embedding_cache.clear()
            if len(text_processing_cache) > 2000:
                text_processing_cache.clear()

        data = request.get_json()
        comparaciones = data.get("comparaciones", [])

        if not comparaciones:
            return jsonify({"status": "success", "comparaciones": []})

        # 1. RECOLECTAR TODOS LOS TEXTOS ÚNICOS (Evita procesar el mismo curso N veces)
        textos_unicos = set()
        pares_a_comparar = []

        for comp in comparaciones:
            orig = comp.get("cursoOrigen", {})
            dest = comp.get("cursoDestino", {})

            # Normalizamos antes de procesar para mejorar el hit-rate del caché
            t_o = normalizar_texto(f"{orig.get('nombre', '')} {orig.get('silabo', {}).get('sumilla', '')}")
            t_d = normalizar_texto(f"{dest.get('nombre', '')} {dest.get('silabo', {}).get('sumilla', '')}")

            textos_unicos.add(t_o)
            textos_unicos.add(t_d)

            pares_a_comparar.append({
                "orig_obj": orig,
                "dest_obj": dest,
                "txt_o": t_o,
                "txt_d": t_d
            })

        # 2. TENSOR BATCHING: Procesar todos los embeddings en paralelo a nivel de hardware (GPU/CPU)
        print(f"\n🚀 Procesando lote de {len(textos_unicos)} textos únicos en BERT...")
        embeddings_dict = obtener_embeddings_batch(list(textos_unicos))

        # 3. CÁLCULO DE SIMILITUDES EN MEMORIA
        resultados = []
        for par in pares_a_comparar:
            emb_o = embeddings_dict[par["txt_o"]]
            emb_d = embeddings_dict[par["txt_d"]]

            # CALCULO CORRECTO DEL COSENO ENTRE ORIGEN Y DESTINO
            similitud_embeddings = float(cosine_similarity([emb_o], [emb_d])[0][0])

            # Cálculo léxico de términos
            terminos_result = extraer_terminos_clave_avanzado(par["txt_o"], par["txt_d"])
            similitud_terminos = terminos_result["ponderacion_contextual"]

            # Fusión 60% Semántica Contextual / 40% Coincidencia Léxica
            similitud_contextual = ((similitud_embeddings * PESO_SIMILITUD_BUSQUEDA_SEMANTICA_BERT) +
                                    (similitud_terminos * PESO_SIMILITUD_BUSQUEDA_SEMANTICA_JACCARD))

            # Umbral de descarte para limpieza de resultados irrelevantes
            if similitud_contextual < UMBRAL_SIMILITUD_BUSQUEDA_SEMANTICA:
                continue

            resultados.append({
                "cursoOrigen": {
                    "idCurso": int(par["orig_obj"].get("idCurso", 0)),
                    "nombre": str(par["orig_obj"].get("nombre", ""))
                },
                "cursoDestino": {
                    "idCurso": int(par["dest_obj"].get("idCurso", 0)),
                    "nombre": str(par["dest_obj"].get("nombre", ""))
                },
                "resultado_resumido": {
                    "similitud_global": float(round(similitud_contextual, 4)),
                    "terminos_comunes": terminos_result["terminos_comunes"]
                }
            })

        tiempo_total = float(round(time.perf_counter() - tiempo_inicio, 2))
        print(f"✅ Lote completado. Tiempo total: {tiempo_total}s | Encontrados: {len(resultados)}")

        # Ordenar resultados de mayor a menor similitud
        resultados_ordenados = sorted(resultados, key=lambda x: -x["resultado_resumido"]["similitud_global"])

        return jsonify(convert_floats({
            "status": "success",
            "tiempo_procesamiento_total_s": tiempo_total,
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


@app.route('/comparar_cursos', methods=['POST'])
def comparar_cursos():
    """
    ENDPOINT PARA EVALUACIÓN EXHAUSTIVA DE UNIDADES Y BIBLIOGRAFÍAS
    Ahora optimizado con Pre-computación Batch para velocidad extrema.
    """
    try:
        tiempo_inicio = time.perf_counter()

        with cache_lock:
            if len(embedding_cache) > 2000:
                embedding_cache.clear()
            if len(text_processing_cache) > 2000:
                text_processing_cache.clear()

        data = request.get_json()
        comparaciones = data.get("comparaciones", [])

        if not comparaciones:
            return jsonify({"status": "success", "comparaciones": []})

        # =======================================================
        # 🚀 1. PRE-COMPUTACIÓN BATCH (EL TRUCO MAESTRO)
        # =======================================================
        textos_a_preprocesar = set()

        for comp in comparaciones:
            orig = comp.get("cursoOrigen", {}).get("silabo", {})
            dest = comp.get("cursoDestino", {}).get("silabo", {})

            # Recolectar Sumillas y Aprendizajes
            textos_a_preprocesar.add(normalizar_texto(orig.get("sumilla", "")))
            textos_a_preprocesar.add(normalizar_texto(dest.get("sumilla", "")))
            textos_a_preprocesar.add(normalizar_texto(orig.get("aprendizaje_general", "")))
            textos_a_preprocesar.add(normalizar_texto(dest.get("aprendizaje_general", "")))

            # Recolectar Bibliografías combinadas
            bib_o = " ".join([ref.get("referencia", "") for ref in orig.get("bibliografias", [])])
            bib_d = " ".join([ref.get("referencia", "") for ref in dest.get("bibliografias", [])])
            textos_a_preprocesar.add(normalizar_texto(bib_o))
            textos_a_preprocesar.add(normalizar_texto(bib_d))

            # Recolectar Unidades y temas individuales de Origen
            for u in orig.get("unidades", []):
                textos_a_preprocesar.add(normalizar_texto(u.get("titulo", "")))
                textos_a_preprocesar.add(normalizar_texto(u.get("aprendizajes", "")))
                textos_a_preprocesar.add(normalizar_texto(u.get("temas", "")))
                # Extraer temas individuales para la función identificar_temas_comunes
                for tema in split_pattern.split(u.get("temas", "")):
                    if tema.strip(): textos_a_preprocesar.add(tema.strip())

            # Recolectar Unidades y temas individuales de Destino
            for u in dest.get("unidades", []):
                textos_a_preprocesar.add(normalizar_texto(u.get("titulo", "")))
                textos_a_preprocesar.add(normalizar_texto(u.get("aprendizajes", "")))
                textos_a_preprocesar.add(normalizar_texto(u.get("temas", "")))
                for tema in split_pattern.split(u.get("temas", "")):
                    if tema.strip(): textos_a_preprocesar.add(tema.strip())

        # Limpiar vacíos
        textos_a_preprocesar.discard("")

        print(f"\n🧠 Pre-computando lote de {len(textos_a_preprocesar)} fragmentos de texto en BERT...")
        # ¡Magia! Generamos todos los embeddings de golpe y quedan en la caché
        obtener_embeddings_batch(list(textos_a_preprocesar))

        # =======================================================
        # ⚙️ 2. PROCESAMIENTO MATEMÁTICO (Rápido gracias al Caché)
        # =======================================================
        resultados = []
        tiempo_total_inicio = time.time()

        for comparacion in comparaciones:
            tiempo_comparacion_inicio = time.time()
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
                    "porcentaje_emparejamiento_unidades": float(
                        resultado_unidades["porcentaje_emparejamiento_unidades"])
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

        tiempo_total = (time.time() - tiempo_total_inicio) * 1000
        print(f"\n📊 Comparaciones exhaustivas: {len(comparaciones)}")
        print(f"⏱ Tiempo Total: {tiempo_total:.2f} ms")

        return jsonify(convert_floats({
            "status": "success",
            "tiempo_procesamiento_total_ms": float(tiempo_total),
            "comparaciones": resultados
        }))

    except Exception as e:
        tiempo_fin = time.perf_counter()
        print(f"\n❌ Error en comparar_cursos: {str(e)}")
        traceback.print_exc()

        return jsonify(convert_floats({
            "status": "error",
            "error_message": str(e),
            "stack_trace": traceback.format_exc()
        })), 500


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
print("✅ Servicio de Inteligencia Artificial listo para recibir peticiones\n")

if __name__ == '__main__':
    app.run(port=5000, threaded=True)