import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carregar o dataset de flores
flowers_dataset = tf.keras.utils.get_file(
    "flower_photos", 
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz", 
    untar=True
)
print("Dataset path:", flowers_dataset)
print("Subdirectories:", os.listdir(flowers_dataset))
flowers_dataset = os.path.join(flowers_dataset, "flower_photos")
print("Dataset path:", flowers_dataset)
print("Subdirectories:", os.listdir(flowers_dataset))

data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.5)

train_gen = data_gen.flow_from_directory(
    flowers_dataset,
    target_size=(150, 150),
    batch_size=32,
    subset="training",
    class_mode="sparse"
)

val_gen = data_gen.flow_from_directory(
    flowers_dataset,
    target_size=(150, 150),
    batch_size=32,
    subset="validation",
    class_mode="sparse"
)

# Construção do modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_gen.class_indices), activation='softmax')  # Saída com base nas classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback para exibir métricas durante o treinamento
class TrainingMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        print(f"Epoch {epoch + 1}:")
        print(f"  - loss: {loss:.4f} - accuracy: {acc:.4f}")
        print(f"  - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

# Treinar o modelo com o callback
model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=10,
    callbacks=[TrainingMetrics()]
)

# Salvar o modelo
model.save('/app/model/flower_model.h5')

# Avaliação no conjunto de validação
val_gen.reset()
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=-1)

# Relatório de Classificação
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))
