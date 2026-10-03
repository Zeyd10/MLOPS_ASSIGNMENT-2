
import os
import numpy as np
import yaml


def main():
    # Read preprocessing settings
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)["preprocess"]

    validation_size = params["test_size"]
    seed = params["seed"]

    # Load raw arrays
    x_train = np.load("data/raw/x_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 128.0
    x_test = x_test.astype("float32") / 128.0

    # Reproducibly shuffle and split the training data
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train))

    split_index = int(len(x_train) * (1 - validation_size))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    x_train_processed = x_train[train_indices]
    y_train_processed = y_train[train_indices]

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]

    # Save processed arrays
    os.makedirs("data/processed", exist_ok=True)

    np.save("data/processed/x_train.npy", x_train_processed)
    np.save("data/processed/y_train.npy", y_train_processed)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test)
    np.save("data/processed/y_test.npy", y_test)

    print("Preprocessing completed successfully.")
    print("Training images:", x_train_processed.shape)
    print("Validation images:", x_val.shape)
    print("Test images:", x_test.shape)
    print("Pixel range:", x_train_processed.min(), "to",
          x_train_processed.max())


if __name__ == "__main__":
    main()