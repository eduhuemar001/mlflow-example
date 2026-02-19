import mlflow
import mlflow.pyfunc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():

    model_uri = input("Paste Model URI (runs:/.../model): ")

    mlflow.set_experiment("mlflow-demo")

    X, y = load_iris(return_X_y=True)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load model
    model = mlflow.pyfunc.load_model(model_uri)

    with mlflow.start_run():

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_param("evaluated_model", model_uri)

        print("Test accuracy:", acc)


if __name__ == "__main__":
    main()
