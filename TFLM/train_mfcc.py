import tensorflow as tf
import numpy as np
import librosa
import os
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def load_data(data_path):
    labels = ['shalom', 'unknown', 'noise', 'lehit']  # Replace with your labels
    X = []
    y = []
    for label in labels:
        folder = os.path.join(data_path, label)
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                wav_path = os.path.join(folder, filename)
                audio, sr = librosa.load(wav_path, sr=16000)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfcc = np.mean(mfcc.T, axis=0)  # Average over time frames
                X.append(mfcc)
                y.append(labels.index(label))
    return np.array(X), np.array(y)

def build_model():
    model = models.Sequential([
        layers.Input(shape=(13,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')  # 4 labels
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def test_model_accuracy(data_path):
    # Load dataset with MFCC features and labels
    X, y = load_data(data_path)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build your model architecture
    model = build_model()

    # Train the model
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

    # Evaluate on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Optionally, print predictions and probabilities for some test samples
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    for i in range(min(10, len(X_test))):
        print(f"Sample {i}: True label={y_test[i]}, Predicted={predicted_classes[i]}, Probabilities={predictions[i]}")

    # Return accuracy for further use if needed
    return accuracy


data_path = 'TFLM/data'  # Put your dataset folder path here
X, y = load_data(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = build_model()
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Convert and save TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('wake_word_model.tflite', 'wb') as f:
    f.write(tflite_model)

test_model_accuracy(data_path)

print("Model training and conversion done.")
