import argparse
import subprocess
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ---------- Git helpers ----------
def _run_git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_commit() -> str:
    return _run_git(["git", "rev-parse", "HEAD"])


def git_branch() -> str:
    return _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def git_worktree_state() -> str:
    status = _run_git(["git", "status", "--porcelain"])
    if status == "unknown":
        return "unknown"
    return "dirty" if status else "clean"


# ---------- MLflow helper ----------
def log_code_snapshot(artifact_path: str = "code") -> None:
    files = [
        "train.py",
        "evaluate.py",
        "MLproject",
        "python_env.yaml",
        "requirements.txt",
        ".gitignore",
        "README.md",
    ]
    for f in files:
        if Path(f).exists():
            mlflow.log_artifact(f, artifact_path=artifact_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, default="mlflow-demo")
    parser.add_argument("--model-uri", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    # Same dataset & split settings as train.py so "test set" is consistent
    X, y = load_iris(return_X_y=True)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = mlflow.pyfunc.load_model(args.model_uri)

    with mlflow.start_run(run_name="evaluate") as run:
        # Tags
        mlflow.set_tag("git_commit", git_commit())
        mlflow.set_tag("git_branch", git_branch())
        mlflow.set_tag("git_worktree", git_worktree_state())
        mlflow.set_tag("entrypoint", "evaluate.py")

        # Params
        mlflow.log_param("model_uri", args.model_uri)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("dataset", "sklearn_iris")

        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        mlflow.log_metric("test_accuracy", acc)

        # Artifact: tiny plot
        plt.figure()
        preds_arr = np.asarray(preds)
        plt.hist(preds_arr, bins=np.arange(preds_arr.min(), preds_arr.max() + 2) - 0.5)
        plt.title("Test prediction distribution")
        plt.xlabel("Predicted class")
        plt.ylabel("Count")
        plot_path = "test_pred_distribution.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")

        # Code snapshot
        log_code_snapshot(artifact_path="code")

        print("\n=== EVALUATE DONE ===")
        print("Run ID:", run.info.run_id)
        print("test_accuracy:", acc)


if __name__ == "__main__":
    main()