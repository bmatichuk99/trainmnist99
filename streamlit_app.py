import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np

# Function to load and preprocess the MNIST dataset
@st.cache(allow_output_mutation=True)
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Function to create the model
def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Streamlit app
st.title("MNIST Training App")

st.write("This app trains a simple neural network on the MNIST dataset.")

# Load data
(x_train, y_train), (x_test, y_test) = load_data()

# Display data shape
st.write("Training data shape: ", x_train.shape)
st.write("Test data shape: ", x_test.shape)

# Train model button
if st.button("Train Model"):
    model = create_model()
    with st.spinner('Training the model...'):
        history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
    st.success('Model training completed!')

    # Display training history
    st.write("Training accuracy:", history.history['accuracy'][-1])
    st.write("Validation accuracy:", history.history['val_accuracy'][-1])

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    st.write(f"Test accuracy: {test_acc:.4f}")

    # Save the model
    model.save("mnist_model.h5")
    st.write("Model saved as mnist_model.h5")

# Function to display sample images
def display_sample_images(x, y, num_samples=5):
    indices = np.random.choice(len(x), num_samples, replace=False)
    for i in indices:
        st.image(x[i], width=100, caption=f"Label: {np.argmax(y[i])}")

# Display sample images
st.write("Sample training images:")
display_sample_images(x_train, y_train)
