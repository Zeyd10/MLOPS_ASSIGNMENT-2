import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def main():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save("data/raw/x_train.npy", x_train)
    np.save("data/raw/y_train.npy", y_train)
    np.save("data/raw/x_test.npy", x_test)
    np.save("data/raw/y_test.npy", y_test)

    print("Fashion-MNIST raw data saved successfully.")


if __name__ == "__main__":
    main()