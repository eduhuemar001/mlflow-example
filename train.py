import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():

    # 1️⃣ Create / select experiment
    mlflow.set_experiment("mlflow-demo")

    # 2️⃣ Load simple dataset
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3️⃣ Start MLflow run
    with mlflow.start_run() as run:

        # 4️⃣ Define model (VERY SIMPLE)
        model = LogisticRegression(max_iter=200)

        # 5️⃣ Train
        model.fit(X_train, y_train)

        # 6️⃣ Predict
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # 7️⃣ Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 200)

        # 8️⃣ Log metric
        mlflow.log_metric("accuracy", acc)

        # 9️⃣ Log simple artifact (plot)
        plt.hist(preds)
        plt.title("Prediction Distribution")
        plt.savefig("pred_plot.png")
        plt.close()
        mlflow.log_artifact("pred_plot.png")

        # 🔟 Log model
        mlflow.sklearn.log_model(model, "model")

        print("Run ID:", run.info.run_id)
        print("Model URI:", f"runs:/{run.info.run_id}/model")


if __name__ == "__main__":
    main()
