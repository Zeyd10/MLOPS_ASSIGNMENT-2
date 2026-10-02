import json
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():

    # Load trained model
    model = load_model("models/model.h5")

    # Load test data
    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Generate predictions
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("models", exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot()
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.savefig("models/confusion_matrix.png")
    plt.close()

    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Confusion matrix saved to models/confusion_matrix.png")
    print("Metrics saved to metrics.json")


if __name__ == "__main__":
    main()