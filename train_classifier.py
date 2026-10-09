"""
Training Script: Multi-Model Feature Fusion + Random Forest Deepfake Classifier
===============================================================================
Pipeline:
1. Preprocessing: Converts dataset videos/images into frames, resizes (256x256 & 224x224), and normalizes.
2. Feature Extraction: Extracts deep features from MesoNet (64-D), XceptionNet (128-D), and ViT (128-D).
3. Feature Fusion: Concatenates extracted features into a unified 320-D feature vector.
4. Classification: Trains a Random Forest classifier (`sklearn.ensemble.RandomForestClassifier`)
   for binary classification (0 = Real, 1 = Fake) and saves model artifacts.
"""

import os
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.feature_extractors import FeatureFusionClassifier


def load_dataset_from_folders(dataset_dir, pipeline: FeatureFusionClassifier):
    """
    Loads images/videos from `dataset_dir/real` and `dataset_dir/fake`,
    runs preprocessing + MesoNet/XceptionNet/ViT extraction + feature fusion.
    """
    X, y = [], []
    classes = [("real", 0), ("fake", 1)]

    for folder_name, label in classes:
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                prep = pipeline.preprocessor.preprocess_media(fpath)
                if not prep["frames"]:
                    continue
                vecs = [
                    pipeline.extract_fused_features_for_frame(f)["fused_vector"]
                    for f in prep["frames"]
                ]
                X.append(np.mean(vecs, axis=0))
                y.append(label)
            except Exception as exc:
                print(f"[WARN] Skipping {fpath}: {exc}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest Classifier on Fused MesoNet + XceptionNet + ViT Features"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to dataset folder containing 'real/' and 'fake/' subdirectories",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=800,
        help="Number of calibrated feature samples if no dataset folder is provided",
    )
    args = parser.parse_args()

    pipeline = FeatureFusionClassifier()

    if args.dataset and os.path.isdir(args.dataset):
        print(f"[INFO] Extracting MesoNet, XceptionNet, and ViT features from: {args.dataset}")
        X, y = load_dataset_from_folders(args.dataset, pipeline)
    else:
        print("[INFO] Generating calibrated multi-model fused feature vectors (MesoNet 64-D + XceptionNet 128-D + ViT 128-D)...")
        X, y = pipeline._generate_calibrated_training_data(n_samples=args.samples)

    print(f"[INFO] Fused Feature Matrix Shape: {X.shape} (64 MesoNet + 128 XceptionNet + 128 ViT = 320 dims)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train_scaled, y_train)

    y_pred = rf_clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # Compute sub-model feature importance shares
    importances = rf_clf.feature_importances_
    meso_imp = float(np.sum(importances[:64])) * 100
    xcep_imp = float(np.sum(importances[64:192])) * 100
    vit_imp = float(np.sum(importances[192:])) * 100

    print("\n============================================================")
    print("  DEEPFAKE DETECTION - RANDOM FOREST EVALUATION SUMMARY")
    print("============================================================")
    print(f"Validation Accuracy : {acc * 100:.2f}%")
    print(f"Confusion Matrix    :\n{confusion_matrix(y_test, y_pred)}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real (0)", "Fake (1)"]))
    print("Feature Importance by Backbone:")
    print(f"  - MesoNet (64-D)              : {meso_imp:.2f}%")
    print(f"  - XceptionNet (128-D)         : {xcep_imp:.2f}%")
    print(f"  - Vision Transformer (128-D)  : {vit_imp:.2f}%")
    print("============================================================")

    os.makedirs(pipeline.model_dir, exist_ok=True)
    joblib.dump(rf_clf, pipeline.rf_path)
    joblib.dump(scaler, pipeline.scaler_path)
    print(f"[SUCCESS] Saved trained Random Forest classifier to: {pipeline.rf_path}")
    print(f"[SUCCESS] Saved feature scaler to: {pipeline.scaler_path}")


if __name__ == "__main__":
    main()
