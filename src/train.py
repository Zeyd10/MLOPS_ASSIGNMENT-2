import os
import numpy as np
import pandas as pd
import yaml

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def main():

    # Load training parameters
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)["train"]

    dense_units = params["dense_units"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    # Load processed data
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    # Build ANN
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(10, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # Make sure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save model
    model.save("models/model.h5")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("models/history.csv", index=False)

    print("Training completed successfully.")
    print("Model saved to models/model.h5")
    print("Training history saved to models/history.csv")


if __name__ == "__main__":
    main()