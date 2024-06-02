import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from streamlit_drawable_canvas import st_canvas
import os

MODEL_PATH = 'mnist_model.h5'

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    grayscale_image = ImageOps.grayscale(image)
    
    # Resize the image to 28x28 pixels
    resized_image = grayscale_image.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array and normalize the image to have values between 0 and 1
    image_array = np.array(resized_image).astype('float32') / 255.0

    # Apply a less aggressive threshold
    threshold = 0.25  # Lower threshold value to make it less aggressive
    image_array = np.where(image_array > threshold, image_array, 0.0)

    # Reshape the image to fit the model input
    reshaped_image = image_array.reshape(1, 28, 28, 1)
    return reshaped_image, image_array

# Function to build and train the model
def train_model(progress_bar, status_text):
    # Load and preprocess the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # Data augmentation
    datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=10,
        zoom_range=0.1
    )
    datagen.fit(x_train)

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

    # Train the model using the augmented data
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=5, 
              validation_data=(x_test, y_test),
              callbacks=[StreamlitCallback()])

    # Save the model
    model.save(MODEL_PATH)

    return model

# Load the model if it exists
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

# Initialize the model
model = load_model()

st.title("MNIST Digit Recognizer")

# Create a wider container for the layout
with st.container():
    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
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

    with col2:
        # Create a canvas to draw on
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,  # Increased size to match the aspect ratio
            width=280,
            drawing_mode="freedraw",
            key="canvas"
        )

    # Display the results in columns
    if model and assess_button:
        if canvas_result.image_data is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                # Extract the image data from the canvas
                image_data = canvas_result.image_data[:, :, :3]  # Use RGB channels

                # Convert the image data to grayscale by averaging the RGB channels
                image_data = np.mean(image_data, axis=2).astype(np.uint8)

                # Convert the image data to a PIL image
                image = Image.fromarray(image_data)

                # Preprocess the image
                processed_image, final_image = preprocess_image(image)

                # Display the final preprocessed image
                st.image(final_image, caption='Final Preprocessed Image', use_column_width=True, clamp=True)

            with col2:
                # Make a prediction
                predictions = model.predict(processed_image)
                predicted_digit = np.argmax(predictions)

                st.write(f"Predicted Digit: {predicted_digit}")
                st.bar_chart(predictions[0])
        else:
            st.warning("Please draw something on the canvas!")
    elif not model and assess_button:
        st.warning("Please train the model first!")
