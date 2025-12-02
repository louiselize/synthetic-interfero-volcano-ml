from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

def run_random_forest_classification(
    X,
    y,
    experiment_name="",
    test_size=0.3,
    random_state=42,
    n_estimators=200,
    max_depth=None,
    top_k_features=10,
    plot_example_tree=True,
    tree_max_depth=4,
):
    """
    Train + evaluate a RandomForestClassifier on given X, y.

    Prints metrics, plots confusion matrix and a sample tree,
    and prints top-k most important features.

    Returns:
        rf: fitted RandomForestClassifier
        results: dict with confusion matrix, y_test, y_pred, feature_importances, classes, feature_names
    """
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    # 3. Predictions and metrics
    y_pred = rf.predict(X_test)

    classes = rf.classes_
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(f"\n=== {experiment_name or 'RandomForest experiment'} ===")
    print("Classes:", classes)
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, target_names=classes))

    # 4. Confusion matrix plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix - {experiment_name or 'RandomForest'}")
    plt.tight_layout()
    plt.show()

    # 5. Feature importances
    importances = rf.feature_importances_
    feature_names = np.array(X.columns)

    idx = np.argsort(importances)[::-1][:top_k_features]
    print(f"\nTop {top_k_features} features:")
    for name, imp in zip(feature_names[idx], importances[idx]):
        print(f"{name:20s} {imp:.3f}")

    # 6. Plot one tree from the forest (optional)
    if plot_example_tree:
        estimator = rf.estimators_[0]
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(
            estimator,
            feature_names=feature_names,
            class_names=classes,
            filled=False,
            impurity=False,
            max_depth=tree_max_depth,
            fontsize=6,
            ax=ax,
        )
        plt.tight_layout()
        plt.show()

    results = {
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importances": importances,
        "classes": classes,
        "feature_names": feature_names,
    }
    return rf, results




from sklearn.tree import plot_tree

def run_random_forest_regression(
    X,
    y,
    target_name="",
    experiment_name="",
    test_size=0.3,
    random_state=42,
    n_estimators=200,
    max_depth=None,
    plot_scatter=True,
    plot_example_tree=False,
    tree_max_depth=3,
):
    """
    Train + evaluate a RandomForestRegressor on given X, y.

    Prints metrics (R2, RMSE, MAE) and optionally:
      - plots true vs predicted values (scatter)
      - plots one example tree from the forest

    Returns:
        rf: fitted RandomForestRegressor
        results: dict with y_test, y_pred, metrics, feature_importances, feature_names
    """

    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Random Forest regressor
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    # 3. Predictions & metrics
    y_pred = rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n=== {experiment_name or 'RF regression'} | target = {target_name} ===")
    print(f"R²   : {r2:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")

    # 4. Scatter plot true vs predicted
    if plot_scatter:
        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.5)
        min_v = min(y_test.min(), y_pred.min())
        max_v = max(y_test.max(), y_pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], "--")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        plt.title(f"{experiment_name or 'RF regression'} – {target_name}")
        plt.tight_layout()
        plt.show()

    feature_names = np.array(X.columns)
    importances = rf.feature_importances_

    # 5. Optional: plot one tree from the forest
    if plot_example_tree:
        estimator = rf.estimators_[0]
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(
            estimator,
            feature_names=feature_names,
            filled=False,
            impurity=False,
            max_depth=tree_max_depth,
            fontsize=6,
            ax=ax,
        )
        plt.tight_layout()
        plt.show()

    results = {
        "y_test": y_test,
        "y_pred": y_pred,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "feature_importances": importances,
        "feature_names": feature_names,
    }
    return rf, results
