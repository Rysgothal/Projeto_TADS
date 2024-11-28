import logging
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os


app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

# Carregar modelo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Diretório atual do app.py

MODEL_PATH = os.path.join(BASE_DIR, '..', "model", "flower_model.h5")
model = load_model(MODEL_PATH)

# Classes (modifique de acordo com o dataset)
CLASS_NAMES = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

# # Caminho para o dataset de validação
# DATASET_PATH = "flower_photos"  # Altere se necessário

# # Criar geradores globalmente
# def get_data_generators():
#     data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    
#     train_gen = data_gen.flow_from_directory(
#         DATASET_PATH,
#         target_size=(150, 150),
#         batch_size=32,
#         subset="training",
#         class_mode="sparse"
#     )

#     val_gen = data_gen.flow_from_directory(
#         DATASET_PATH,
#         target_size=(150, 150),
#         batch_size=32,
#         subset="validation",
#         class_mode="sparse",
#         shuffle=False  # Importante para manter a ordem dos rótulos
#     )
#     return train_gen, val_gen

# train_gen, val_gen = get_data_generators()

# @app.route('/status', methods=['GET'])
# def status():
#     # Resetar o gerador de validação
#     val_gen.reset()

#     # Previsões
#     y_true = val_gen.classes
#     y_pred = np.argmax(model.predict(val_gen), axis=-1)

#     # Calcular métricas
#     accuracy = np.mean(y_true == y_pred)
#     errors = np.sum(y_true != y_pred)
#     total = len(y_true)

#     return jsonify({
#         "accuracy": round(accuracy, 4),
#         "errors": int(errors),
#         "total_samples": total
#     })
@app.route('/ping', methods=['GET'])
def ping():
    app.logger.debug("Ping endpoint foi chamado")
    return jsonify({"message": "pong"})

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Processar a imagem
    img = load_img(file, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fazer previsão
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    return jsonify({"class": predicted_class, "confidence": float(confidence)})

if __name__ == '__main__':
    app.logger.debug("Iniciando o servidor Flask")
    app.run(host='0.0.0.0', port=5000)
