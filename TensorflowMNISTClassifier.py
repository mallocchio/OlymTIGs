import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import load_model
from torchvision import datasets, transforms

class TensorflowMNISTClassifier:
    def __init__(self):
        self.num_classes = 10
        self.model = self.build_model()
        self.trainded_model = None

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, 3, input_shape=(28, 28, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(3, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_ds, epochs=5):
        report = self.model.fit(train_ds, epochs=epochs)
        return report

    def evaluate(self, test_ds):
        return self.model.evaluate(test_ds)

    def save_model(self, model_path):
        self.model.save(model_path + '.h5')


class TrainedTensorflowClassifier:

    def __init__(self, model_path):
        self.trained_model = load_model(model_path)
        self.tensor = None

    def preprocess_image(self, image):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        self.tensor = transform(image)


    def get_prediction(self, image):
        self.preprocess_image(image)
        modified_tensor = tf.expand_dims(image, 0)
        ps = self.trained_model.predict(modified_tensor)
        logps = tf.exp(ps)
        predicted_digit = np.argmax(logps[0])
        return self.tensor, ps, logps, predicted_digit

    #def get_prediction(self, tensor):

if __name__ == "__main__":
    train_ds, test_ds = tfds.load(
        'mnist',
        split=['train', 'test'],
        as_supervised=True,
        batch_size=8,
    )

    classifier = TensorflowMNISTClassifier()
    classifier.build_model()
    classifier.model.summary()

    report = classifier.train(train_ds, epochs=5)

    pd.DataFrame(report.history).plot()

    classifier.evaluate(test_ds)

    model_path = 'mnist_classifier_model.h5'
    classifier.save_model(model_path)