from flask import Flask, request, jsonify
import spacy
from spacy.lang.es.stop_words import STOP_WORDS as STOP_WORDS_ES
import logging
import unicodedata

app = Flask(__name__)

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load('es_core_news_lg')

stop_words = STOP_WORDS_ES.union({
    "de", "a", "el", "la", "las", "y", "en", "los", "del", "que", "con", "por", "para",
    "se", "su", "al", "como", "lo", "es", "más", "o", "u", "una", "uno", "le", "les",
    "me", "mi", "mí", "nos", "nosotros", "vosotros", "su", "sus", "nuestro", "nuestra",
    "nuestros", "nuestras", "vuestra", "vuestro", "vuestras", "vuestros", "ser", "fue",
    "soy", "eres", "son", "era", "fui", "fuiste", "fueron", "ese", "esa", "esos", "esas",
    "este", "esta", "estos", "estas", "."
})


def cleanText(text):
    if not text:
        return "", []

    text_without_tildes = ''.join(
        c for c in unicodedata.normalize('NFC', text)
        if ord(c) < 128 or (192 <= ord(c) <= 255)
    )

    doc = nlp(text_without_tildes)
    tokens = [
        token.lemma_.lower() for token in doc
        if (token.is_alpha or token.is_digit or '+' in token.text or token.text.lower() == 'python')
           and token.text.lower() not in stop_words
    ]
    removed_words = [token.text.lower() for token in doc if token.text.lower() in stop_words]
    return ' '.join(tokens), removed_words


def calculate_similarity(text1, text2):
    cleaned_text1, _ = cleanText(text1)
    cleaned_text2, _ = cleanText(text2)
    docText1 = nlp(cleaned_text1)
    docText2 = nlp(cleaned_text2)
    return docText1.similarity(docText2), cleaned_text1, cleaned_text2

def find_similar_words(tokens1, tokens2, threshold=0.1):
    """ Encuentra palabras similares en significado entre dos listas de tokens. """
    similar_pairs = []
    for word1 in tokens1:
        for word2 in tokens2:
            similarity = nlp(word1).similarity(nlp(word2))
            if similarity >= threshold:
                similar_pairs.append((word1, word2, similarity))
    return sorted(similar_pairs, key=lambda x: x[2], reverse=True)

def detect_possible_error(curso_similarity, silabo_similarity):
    if curso_similarity > 0.7 and (curso_similarity - silabo_similarity) > 0.5:
        return True, ('Alta similitud en el nombre del curso, pero baja similitud en el contenido del sílabo. '
                      'Revise que el contenido del curso sea coherente con su nombre.')
    elif curso_similarity < 0.5 and (curso_similarity - silabo_similarity) < 0.7:
        return True, ('Baja similitud en el nombre del curso, pero cierta coherencia en el contenido del sílabo. '
                      'Revise los nombres de los cursos y su contenido para asegurar la consistencia.')
    return False, None

def calculate_totalSimilarity(curso_similarity, silabo_similarity):
    thresholds = {
        (0.0, 0.15): 0.35,
        (0.15, 0.25): 0.28,
        (0.25, 0.35): 0.22,
        (0.35, 0.45): 0.18,
        (0.45, 0.55): 0.12,
        (0.55, 0.65): -0.02,
        (0.65, 0.75): -0.04,
        (0.75, 0.85): -0.07,
        (0.85, 0.95): -0.03,
        (0.95, 1.0): 0  # Sin ajuste en similitudes muy altas
    }

    # Obtiene el valor de ajuste umbral según el rango de `curso_similarity`
    umbral = next((value for (lower, upper), value in thresholds.items() if lower <= curso_similarity < upper), 0)
    print(curso_similarity, umbral, silabo_similarity - umbral)
    return silabo_similarity - umbral

"""@app.route('/', methods=['POST'])
def getData():
    try:
        data = request.get_json(force=True)
        logger.info('Datos recibidos: %s', data)  # Registro de datos recibidos

        # Verificación de que los parámetros existen y son del tipo correcto
        required_params = ['cursoLocal', 'silaboLocal', 'cursoPostulante', 'silaboPostulante']
        missing_params = [param for param in required_params if param not in data]

        if missing_params:
            logger.error('Faltan parámetros en la solicitud: %s', ', '.join(missing_params))
            return jsonify({'error': 'Faltan parámetros: {}'.format(', '.join(missing_params))}), 400

        # Validar que los parámetros no están vacíos
        for param in required_params:
            if not isinstance(data[param], str) or not data[param].strip():
                logger.error('El parámetro "%s" no es válido o está vacío', param)
                return jsonify({'error': 'El parámetro "{}" no es válido o está vacío'.format(param)}), 400

        # Calcular similitudes y obtener detalles
        curso_similarity, cleaned_curso_local, cleaned_curso_postulante = calculate_similarity(data['cursoLocal'], data['cursoPostulante'])
        silabo_similarity, cleaned_silabo_local, cleaned_silabo_postulante = calculate_similarity(data['silaboLocal'], data['silaboPostulante'])
        total_similarity = calculate_totalSimilarity(curso_similarity, silabo_similarity)

        # Capturar las palabras similares en significado
        curso_tokens_local = cleaned_curso_local.split()
        curso_tokens_postulante = cleaned_curso_postulante.split()
        silabo_tokens_local = cleaned_silabo_local.split()
        silabo_tokens_postulante = cleaned_silabo_postulante.split()

        similar_words_curso = find_similar_words(curso_tokens_local, curso_tokens_postulante)
        similar_words_silabo = find_similar_words(silabo_tokens_local, silabo_tokens_postulante)

        possible_error, error_message = detect_possible_error(curso_similarity, silabo_similarity)

        return jsonify({
            'possible_error': possible_error,
            'error_message': error_message,
            'curso_similarity': curso_similarity,
            'silabo_similarity': silabo_similarity,
            'total_similarity': total_similarity,
            'cleaned_curso_local': cleaned_curso_local,
            'cleaned_curso_postulante': cleaned_curso_postulante,
            'cleaned_silabo_local': cleaned_silabo_local,
            'cleaned_silabo_postulante': cleaned_silabo_postulante,
            'similar_words_curso': similar_words_curso,
            'similar_words_silabo': similar_words_silabo,
        })
    except ValueError as e:
        logger.error('Error de valor: %s', str(e))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('Error inesperado: %s', str(e))
        return jsonify({'error': 'Ha ocurrido un error inesperado'}), 500
"""
@app.route('/', methods=['POST'])
def getData():
    try:
        # Recibe la solicitud como una lista de diccionarios JSON
        data_list = request.get_json(force=True)

        if not isinstance(data_list, list):
            return jsonify({'error': 'El cuerpo de la solicitud debe ser una lista de objetos JSON.'}), 400

        results = []

        for data in data_list:
            logger.info('Datos recibidos: %s', data)  # Registro de datos recibidos

            # Verificación de que los parámetros existen y son del tipo correcto
            required_params = ['cursoLocal', 'silaboLocal', 'cursoPostulante', 'silaboPostulante']
            missing_params = [param for param in required_params if param not in data]

            if missing_params:
                logger.error('Faltan parámetros en la solicitud: %s', ', '.join(missing_params))
                results.append({
                    'error': 'Faltan parámetros: {}'.format(', '.join(missing_params)),
                    'status': 'error'
                })
                continue

            # Validar que los parámetros no están vacíos
            for param in required_params:
                if not isinstance(data[param], str) or not data[param].strip():
                    logger.error('El parámetro "%s" no es válido o está vacío', param)
                    results.append({
                        'error': 'El parámetro "{}" no es válido o está vacío'.format(param),
                        'status': 'error'
                    })
                    continue

            # Calcular similitudes y obtener detalles
            curso_similarity, cleaned_curso_local, cleaned_curso_postulante = calculate_similarity(data['cursoLocal'],
                                                                                                   data[
                                                                                                       'cursoPostulante'])
            silabo_similarity, cleaned_silabo_local, cleaned_silabo_postulante = calculate_similarity(
                data['silaboLocal'], data['silaboPostulante'])
            total_similarity = calculate_totalSimilarity(curso_similarity, silabo_similarity)

            # Capturar las palabras similares en significado
            curso_tokens_local = cleaned_curso_local.split()
            curso_tokens_postulante = cleaned_curso_postulante.split()
            silabo_tokens_local = cleaned_silabo_local.split()
            silabo_tokens_postulante = cleaned_silabo_postulante.split()

            similar_words_curso = find_similar_words(curso_tokens_local, curso_tokens_postulante)
            similar_words_silabo = find_similar_words(silabo_tokens_local, silabo_tokens_postulante)

            possible_error, error_message = detect_possible_error(curso_similarity, silabo_similarity)

            # Agregar el resultado individual al arreglo de resultados
            results.append({
                'possible_error': possible_error,
                'error_message': error_message,
                'curso_similarity': curso_similarity,
                'silabo_similarity': silabo_similarity,
                'total_similarity': total_similarity,
                'cleaned_curso_local': cleaned_curso_local,
                'cleaned_curso_postulante': cleaned_curso_postulante,
                'cleaned_silabo_local': cleaned_silabo_local,
                'cleaned_silabo_postulante': cleaned_silabo_postulante,
                'similar_words_curso': similar_words_curso,
                'similar_words_silabo': similar_words_silabo,
                'status': 'success'
            })

        # Devolver todos los resultados en una lista JSON
        return jsonify(results)

    except ValueError as e:
        logger.error('Error de valor: %s', str(e))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('Error inesperado: %s', str(e))
        return jsonify({'error': 'Ha ocurrido un error inesperado'}), 500


if __name__ == '__main__':
    app.run(debug=True)