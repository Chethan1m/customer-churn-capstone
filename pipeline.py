import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
RES, RS = {}, 42
FIG = "figs/"

# ---------------- 1. Data acquisition ----------------
URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
df = pd.read_csv("telco.csv")          # downloaded from URL above
RES["shape_raw"] = list(df.shape)
RES["dtypes_object"] = int((df.dtypes == "object").sum())

# ---------------- 2. Cleaning ----------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
RES["missing_total_charges"] = int(df["TotalCharges"].isna().sum())
RES["missing_tenure_of_those"] = df.loc[df["TotalCharges"].isna(), "tenure"].tolist()
df["TotalCharges"] = df["TotalCharges"].fillna(0)
RES["duplicates"] = int(df.duplicated().sum())
RES["dup_ids"] = int(df["customerID"].duplicated().sum())
df = df.drop(columns="customerID")
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
RES["churn_rate"] = round(df["Churn"].mean() * 100, 2)
RES["churn_counts"] = df["Churn"].value_counts().to_dict()

# Feature engineering
svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumAddOns"] = (df[svc] == "Yes").sum(axis=1)
df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
RES["shape_clean"] = list(df.shape)
RES["describe"] = df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2).to_dict()

# ---------------- 3. EDA ----------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().plot.bar(ax=ax[0], color=["#4C72B0", "#C44E52"], rot=0)
ax[0].set_title("Churn class distribution"); ax[0].set_ylabel("Customers")
ax[1].pie(df["Churn"].value_counts(), labels=["Stayed", "Churned"], autopct="%1.1f%%", colors=["#4C72B0", "#C44E52"])
ax[1].set_title("Churn share")
plt.tight_layout(); plt.savefig(FIG + "eda_target.png", dpi=150); plt.close()

cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "PaperlessBilling", "SeniorCitizen"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
churn_by = {}
for a, c in zip(axes.ravel(), cats):
    r = df.groupby(c)["Churn"].mean().mul(100).sort_values(ascending=False)
    churn_by[c] = r.round(1).to_dict()
    sns.barplot(x=r.values, y=r.index, ax=a, color="#C44E52")
    a.set_title(f"Churn % by {c}"); a.set_xlabel("Churn rate (%)"); a.set_ylabel("")
    for i, v in enumerate(r.values):
        a.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIG + "eda_categorical.png", dpi=150); plt.close()
RES["churn_by"] = churn_by

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for a, c in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.kdeplot(data=df, x=c, hue=df["Churn"].map({0: "Stayed", 1: "Churned"}), fill=True, common_norm=False, ax=a)
    a.set_title(f"{c} by churn")
plt.tight_layout(); plt.savefig(FIG + "eda_numeric.png", dpi=150); plt.close()
RES["numeric_means"] = df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges", "NumAddOns"]].mean().round(2).to_dict()

plt.figure(figsize=(6.5, 5))
num = df[["tenure", "MonthlyCharges", "TotalCharges", "NumAddOns", "AvgMonthlySpend", "Churn"]]
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix"); plt.tight_layout(); plt.savefig(FIG + "eda_corr.png", dpi=150); plt.close()
RES["corr_with_churn"] = num.corr()["Churn"].round(3).to_dict()

# ---------------- 4. Supervised modelling ----------------
X, y = df.drop(columns=["Churn", "TenureGroup"]), df["Churn"]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
RES["num_cols"], RES["n_cat_cols"] = num_cols, len(cat_cols)
pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
RES["split"] = [len(Xtr), len(Xte)]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=RS, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
}
skf = StratifiedKFold(5, shuffle=True, random_state=RS)
fitted, rows, probs = {}, [], {}
for n, m in models.items():
    p = Pipeline([("pre", pre), ("clf", m)])
    cv = cross_val_score(p, Xtr, ytr, cv=skf, scoring="roc_auc").mean()
    p.fit(Xtr, ytr); fitted[n] = p
    pr = p.predict_proba(Xte)[:, 1]; probs[n] = pr; pd_ = (pr >= 0.5).astype(int)
    rows.append(dict(Model=n, CV_AUC=round(cv, 4), Accuracy=round(accuracy_score(yte, pd_), 4),
                     Precision=round(precision_score(yte, pd_), 4), Recall=round(recall_score(yte, pd_), 4),
                     F1=round(f1_score(yte, pd_), 4), ROC_AUC=round(roc_auc_score(yte, pr), 4)))
res = pd.DataFrame(rows); RES["model_table"] = rows
best = res.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
RES["best_model"] = best

# hyper-parameter tuning on best (small grid)
from sklearn.model_selection import GridSearchCV
grid = {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__n_estimators": [100, 200], "clf__max_depth": [2, 3]}
gs = GridSearchCV(Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=RS))]),
                  grid, cv=skf, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
RES["gs_best_params"] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
RES["gs_best_cv_auc"] = round(gs.best_score_, 4)
tuned = gs.best_estimator_
pt = tuned.predict_proba(Xte)[:, 1]; probs["Tuned Gradient Boosting"] = pt
p5 = (pt >= 0.5).astype(int)
RES["tuned_row"] = dict(Model="Tuned Gradient Boosting", CV_AUC=round(gs.best_score_, 4),
                        Accuracy=round(accuracy_score(yte, p5), 4), Precision=round(precision_score(yte, p5), 4),
                        Recall=round(recall_score(yte, p5), 4), F1=round(f1_score(yte, p5), 4),
                        ROC_AUC=round(roc_auc_score(yte, pt), 4))

# threshold tuning (maximise F1 on train OOF would be cleaner; here use PR curve on test for discussion)
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(tuned, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
pp, rr, tt = precision_recall_curve(ytr, oof)
f1s = 2 * pp * rr / (pp + rr + 1e-9)
thr = float(tt[np.argmax(f1s[:-1])])
pth = (pt >= thr).astype(int)
RES["threshold"] = round(thr, 3)
RES["thr_row"] = dict(Accuracy=round(accuracy_score(yte, pth), 4), Precision=round(precision_score(yte, pth), 4),
                      Recall=round(recall_score(yte, pth), 4), F1=round(f1_score(yte, pth), 4))
RES["cm_thr"] = confusion_matrix(yte, pth).tolist()
RES["cm_50"] = confusion_matrix(yte, p5).tolist()

# ROC
plt.figure(figsize=(6, 5))
for n, pr in probs.items():
    f, t, _ = roc_curve(yte, pr); plt.plot(f, t, label=f"{n} (AUC={roc_auc_score(yte, pr):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves (test set)"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG + "roc.png", dpi=150); plt.close()

# confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, cm, t in zip(ax, [RES["cm_50"], RES["cm_thr"]], ["Threshold 0.50", f"Tuned threshold {thr:.2f}"]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    a.set_title(t); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.tight_layout(); plt.savefig(FIG + "cm.png", dpi=150); plt.close()

# feature importance
names = tuned.named_steps["pre"].get_feature_names_out()
imp = pd.Series(tuned.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(12)
imp.index = [i.replace("num__", "").replace("cat__", "") for i in imp.index]
plt.figure(figsize=(7, 4.5)); sns.barplot(x=imp.values, y=imp.index, color="#4C72B0")
plt.title("Top 12 feature importances (tuned Gradient Boosting)"); plt.tight_layout(); plt.savefig(FIG + "importance.png", dpi=150); plt.close()
RES["top_features"] = {k: round(float(v), 4) for k, v in imp.head(8).items()}

# Logistic regression coefficients (interpretability)
lr = fitted["Logistic Regression"]
co = pd.Series(lr.named_steps["clf"].coef_[0], index=[i.replace("num__", "").replace("cat__", "") for i in lr.named_steps["pre"].get_feature_names_out()])
RES["lr_top_pos"] = co.sort_values(ascending=False).head(5).round(2).to_dict()
RES["lr_top_neg"] = co.sort_values().head(5).round(2).to_dict()

# ---------------- 5. Unsupervised ----------------
cl_cols = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddOns"]
Z = StandardScaler().fit_transform(df[cl_cols])
ks, inertia, sil = range(2, 9), [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=RS).fit(Z)
    inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_, sample_size=3000, random_state=RS))
RES["k_sil"] = {int(k): round(s, 4) for k, s in zip(ks, sil)}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, "o-", color="#C44E52"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(FIG + "kmeans_select.png", dpi=150); plt.close()

K = 4
km = KMeans(K, n_init=10, random_state=RS).fit(Z)
df["Cluster"] = km.labels_
RES["k_used"] = K; RES["sil_used"] = round(silhouette_score(Z, km.labels_, sample_size=3000, random_state=RS), 4)
prof = df.groupby("Cluster").agg(Customers=("Churn", "size"), Tenure=("tenure", "mean"), MonthlyCharges=("MonthlyCharges", "mean"),
                                 TotalCharges=("TotalCharges", "mean"), AddOns=("NumAddOns", "mean"), ChurnRate=("Churn", "mean")).round(2)
prof["ChurnRate"] = (prof["ChurnRate"] * 100).round(1)
RES["profile"] = prof.reset_index().to_dict("records")
RES["cluster_contract"] = (pd.crosstab(df["Cluster"], df["Contract"], normalize="index") * 100).round(1).to_dict("index")
pc = PCA(2, random_state=RS).fit(Z); P = pc.transform(Z)
RES["pca_var"] = [round(float(v) * 100, 1) for v in pc.explained_variance_ratio_]
plt.figure(figsize=(7, 5))
sns.scatterplot(x=P[:, 0], y=P[:, 1], hue=df["Cluster"], palette="deep", s=12, alpha=0.7)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Customer segments (PCA projection of K-Means clusters)")
plt.tight_layout(); plt.savefig(FIG + "clusters.png", dpi=150); plt.close()

json.dump(RES, open("results.json", "w"), indent=1, default=str)
print(json.dumps(RES, indent=1, default=str))
