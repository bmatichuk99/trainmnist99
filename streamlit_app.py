import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image = np.array(image)
    # Invert the colors (MNIST images are white on black)
    # image = 255 - image
    # Normalize the image
    image = image - 100
    # Reshape the image to fit the model input
    image = image.reshape(1, 28, 28, 1)
    return image

# Function to build and train the model
def train_model(progress_bar, status_text):
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Custom callback to update progress bar and status text
    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / self.params['epochs']
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[StreamlitCallback()])

    # Save the model
    model.save('mnist_model.h5')

    return model

# Load the model if it exists
try:
    model = tf.keras.models.load_model('mnist_model.h5')
except:
    model = None

st.title("MNIST Digit Recognizer")

# Buttons for training and assessing
train_button = st.button("Train Model")
assess_button = st.button("Assess Drawing")

# Training the model
if train_button:
    with st.spinner('Training the model...'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        model = train_model(progress_bar, status_text)
    st.success("Model trained successfully!")

# Create a canvas to draw on
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Assess the drawing if the model is available and the assess button is clicked
if model and assess_button:
    if canvas_result.image_data is not None:
        # Extract the image data from the canvas
        image_data = canvas_result.image_data

        # Convert the image data to a PIL image
        image = Image.fromarray((image_data[:, :, :3] * 255).astype(np.uint8))

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Debug: Display the preprocessed image
        st.image(processed_image.reshape(28, 28), width=100, caption='Preprocessed Image')

        # Make a prediction
        predictions = model.predict(processed_image)
        predicted_digit = np.argmax(predictions)

        st.write(f"Predicted Digit: {predicted_digit}")
        st.bar_chart(predictions[0])
    else:
        st.warning("Please draw something on the canvas!")
elif not model and assess_button:
    st.warning("Please train the model first!")
