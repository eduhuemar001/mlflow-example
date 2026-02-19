import argparse
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
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
    # clean = no uncommitted changes, dirty = uncommitted changes present
    status = _run_git(["git", "status", "--porcelain"])
    if status == "unknown":
        return "unknown"
    return "dirty" if status else "clean"


# ---------- MLflow helper ----------
def log_code_snapshot(artifact_path: str = "code") -> None:
    """
    Logs a minimal snapshot of important project files into the run artifacts.
    This makes each run reproducible even without GitHub access.
    """
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
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--register-name",
        type=str,
        default="",
        help='Optional: register model under this name (e.g. "IrisClassifier")',
    )
    args = parser.parse_args()

    # Create/select experiment
    mlflow.set_experiment(args.experiment_name)

    # Load dataset
    X, y = load_iris(return_X_y=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Simple model (avoid sklearn version pitfalls)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter)

    with mlflow.start_run(run_name="train") as run:
        # ---- Tags: link run ↔ code ----
        mlflow.set_tag("git_commit", git_commit())
        mlflow.set_tag("git_branch", git_branch())
        mlflow.set_tag("git_worktree", git_worktree_state())
        mlflow.set_tag("entrypoint", "train.py")

        # ---- Params ----
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("dataset", "sklearn_iris")

        # ---- Train + metrics ----
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = float(accuracy_score(y_val, preds))
        mlflow.log_metric("val_accuracy", acc)

        # ---- Artifact: tiny plot ----
        plt.figure()
        preds_arr = np.asarray(preds)
        plt.hist(preds_arr, bins=np.arange(preds_arr.min(), preds_arr.max() + 2) - 0.5)
        plt.title("Validation prediction distribution")
        plt.xlabel("Predicted class")
        plt.ylabel("Count")
        plot_path = "val_pred_distribution.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        mlflow.log_artifact(plot_path, artifact_path="plots")

        # ---- Artifact: code snapshot (important!) ----
        log_code_snapshot(artifact_path="code")

        # ---- Model logging (and optional registry) ----
        if args.register_name.strip():
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=args.register_name.strip(),
            )
            mlflow.set_tag("registered_model_name", args.register_name.strip())
        else:
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        model_uri = f"runs:/{run.info.run_id}/model"

        print("\n=== TRAIN DONE ===")
        print("Run ID:", run.info.run_id)
        print("Model URI:", model_uri)
        print("val_accuracy:", acc)
        print("git_commit:", git_commit())
        print("\nNext:")
        print(f'  python evaluate.py --model-uri "{model_uri}"')


if __name__ == "__main__":
    main()