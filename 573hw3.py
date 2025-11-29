import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import zero_one_loss, roc_auc_score

tempvar = np.load("/Users/ankushidutta/Downloads/datahw3.npz")

train_dat_x = tempvar["train_dat_x"]
val_dat_x = tempvar["val_dat_x"]

# Dataset D1
train_dat_y1 = tempvar["train_dat_y"]
val_dat_y1 = tempvar["val_dat_y"]

# Dataset D2
train_dat_y2 = tempvar["train_dat_y2"]
val_dat_y2 = tempvar["val_dat_y2"]

est_list = [2, 10, 50, 75, 100]

results = {"D1": {"Bagging": [], "Boosting": []},
           "D2": {"Bagging": [], "Boosting": []}}


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    zol = zero_one_loss(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    return zol, auc


def run_bagging(train_dat_x, train_y, val_dat_x, val_dat_y, dataset_key):
    for est in est_list:
        base = SVC(probability=True, random_state=0)

        model = BaggingClassifier(
            estimator=base,           
            n_estimators=est,
            random_state=0
        )

        model.fit(train_dat_x, train_y)
        zol, auc = evaluate(model, val_dat_x, val_dat_y)

        results[dataset_key]["Bagging"].append((est, zol, auc))
        print(f"[Bagging][{dataset_key}] Est={est}: Zero-One={zol:.4f}, AUC={auc:.4f}")


def run_boosting(train_dat_x, train_dat_y, val_dat_x, val_dat_y, dataset_key):
    for est in est_list:
        base = DecisionTreeClassifier(max_depth=1, random_state=0)

        model = AdaBoostClassifier(
            estimator=base,          
            n_estimators=est,
            random_state=0
        )

        model.fit(train_dat_x, train_dat_y)
        zol, auc = evaluate(model, val_dat_x, val_dat_y)

        results[dataset_key]["Boosting"].append((est, zol, auc))
        print(f"[Boosting][{dataset_key}] Est={est}: Zero-One={zol:.4f}, AUC={auc:.4f}")

print("\nDataset D1")
run_bagging(train_dat_x, train_dat_y1, val_dat_x, val_dat_y1, "D1")
run_boosting(train_dat_x, train_dat_y1, val_dat_x, val_dat_y1, "D1")

print("\nDataset D2")
run_bagging(train_dat_x, train_dat_y2, val_dat_x, val_dat_y2, "D2")
run_boosting(train_dat_x, train_dat_y2, val_dat_x, val_dat_y2, "D2")