import os
import io
import uuid
import time
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, abort, flash
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    confusion_matrix,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# ======================================
# APP INITIALIZATION
# ======================================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, "static", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================
# HELPERS
# ======================================
def unique_name(prefix):
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"

def save_figure(fig):
    name = unique_name("out")
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return name, path

def score_assessment(sil):
    if sil is None:
        return "Unknown"
    if sil >= 0.5:
        return "Good (well-separated clusters)"
    if sil >= 0.25:
        return "Moderate"
    return "Poor (overlapping clusters)"

def detect_elbow(var_ratio):
    """
    Very small 'knee' heuristic:
    choose the component where the marginal gain drops the most.
    """
    if len(var_ratio) < 3:
        return 1
    diffs = np.diff(var_ratio)
    # elbow ~ index after the largest drop
    elbow_at = int(np.argmax(-diffs) + 1)
    return elbow_at

# ======================================
# PLOT HELPERS
# ======================================
def plot_pca_scatter_with_hulls(X_pca, labels):
    fig, ax = plt.subplots(figsize=(7, 5))
    unique = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10")
    for i, c in enumerate(unique):
        idx = labels == c
        pts = X_pca[idx]
        ax.scatter(pts[:, 0], pts[:, 1], label=f"Cluster {c}", s=40, alpha=0.85, color=cmap(i))
        if pts.shape[0] >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            poly = pts[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], alpha=0.12, color=cmap(i))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA scatter with convex hull zones")
    ax.legend(frameon=False)
    return save_figure(fig)

def plot_pca_scree(pca):
    fig, ax = plt.subplots(figsize=(6, 4))
    var = pca.explained_variance_ratio_ * 100
    ax.bar(range(1, len(var) + 1), var)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot")
    for i, v in enumerate(var):
        ax.text(i + 1, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    return save_figure(fig)

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return save_figure(fig), cm

def plot_decision_tree(model, feature_names=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_tree(model, feature_names=feature_names, filled=True, ax=ax, fontsize=8)
    ax.set_title("Decision Tree Structure")
    plt.tight_layout()
    return save_figure(fig)

def plot_svm_decision_boundary_2d(X2d, y, svc_model):
    x_min, x_max = X2d[:, 0].min() - 1, X2d[:, 0].max() + 1
    y_min, y_max = X2d[:, 1].min() - 1, X2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
    )
    Z = svc_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.contourf(xx, yy, Z, alpha=0.15, cmap="tab10")
    ax.scatter(X2d[:, 0], X2d[:, 1], c=y, cmap="tab10", s=35, edgecolors="w")
    ax.set_title("SVM Decision Boundaries (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return save_figure(fig)

def plot_svm_3d(X_pca, labels):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    if X_pca.shape[1] < 3:
        pad = np.zeros((X_pca.shape[0], 3 - X_pca.shape[1]))
        X3 = np.hstack([X_pca, pad])
    else:
        X3 = X_pca[:, :3]
    cmap = plt.cm.get_cmap("tab10")
    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(
            X3[mask, 0], X3[mask, 1], X3[mask, 2],
            label=f"Cluster {c}", s=30, alpha=0.8, color=cmap(int(c) % 10)
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA 3D Projection")
    ax.legend()
    return save_figure(fig)

# ======================================
# ROUTES
# ======================================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # ---- read & validate
    if "csvFile" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("dashboard"))
    f = request.files["csvFile"]
    if f.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("dashboard"))
    try:
        df = pd.read_csv(f)
    except Exception as e:
        flash(f"Error reading CSV: {e}", "error")
        return redirect(url_for("dashboard"))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        flash("CSV must have at least 2 numeric columns", "error")
        return redirect(url_for("dashboard"))

    # ---- optional label column
    label_col = None
    for col in df.columns:
        if col.lower() in ["label", "behavior", "target", "class"]:
            label_col = col
            break

    # ---- clean & scale
    df_clean = df.dropna(subset=numeric_cols).reset_index(drop=True)
    X = df_clean[numeric_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---- choose best k by silhouette
    best_k, best_score, best_labels = 2, -1, None
    for k in range(2, min(7, len(Xs))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(Xs)
        try:
            sil = silhouette_score(Xs, labs)
        except Exception:
            sil = -1
        if sil > best_score:
            best_k, best_score, best_labels = k, sil, labs

    if best_labels is None:
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_labels = km.fit_predict(Xs)
        try:
            best_score = silhouette_score(Xs, best_labels)
        except Exception:
            best_score = None

    # ---- metrics
    try:
        dbi = davies_bouldin_score(Xs, best_labels)
    except Exception:
        dbi = None
    try:
        ch = calinski_harabasz_score(Xs, best_labels)
    except Exception:
        ch = None

    # ---- PCA
    pca = PCA(n_components=min(6, Xs.shape[1]))
    X_pca = pca.fit_transform(Xs)

    outputs = {}
    descriptions = {}

    # Common facts for descriptions
    pc1_var = float(pca.explained_variance_ratio_[0] * 100 if pca.n_components_ >= 1 else 0)
    pc2_var = float(pca.explained_variance_ratio_[1] * 100 if pca.n_components_ >= 2 else 0)
    cum2 = pc1_var + pc2_var
    unique_labs, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes = ", ".join([f"C{int(c)}: {int(n)}" for c, n in zip(unique_labs, counts)])

    # Some separation heuristics in 2D PCA
    means = []
    radii = []
    for c in unique_labs:
        pts = X_pca[best_labels == c, :2]
        means.append(pts.mean(axis=0))
        radii.append(np.median(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    means = np.array(means)
    inter_centroid_min = float(np.inf) if len(means) < 2 else float(np.min(
        [np.linalg.norm(means[i] - means[j]) for i in range(len(means)) for j in range(i+1, len(means))]
    ))
    avg_radius = float(np.mean(radii)) if len(radii) else 0.0

    # ---------- PCA scatter + convex hulls
    try:
        import scipy.spatial  # noqa
        name, path = plot_pca_scatter_with_hulls(X_pca[:, :2], best_labels)
        outputs["pca_hull"] = name
        sep_hint = (
            "well-separated" if inter_centroid_min > 2.5 * max(1e-9, avg_radius)
            else "partially overlapping" if inter_centroid_min > 1.2 * max(1e-9, avg_radius)
            else "heavily overlapping"
        )
        descriptions["pca_hull"] = (
            f"Scatter on PC1/PC2 (variance PC1 {pc1_var:.1f}%, PC2 {pc2_var:.1f}%, cumulative {cum2:.1f}%). "
            f"KMeans chose k={best_k} with silhouette={best_score:.3f} → {score_assessment(best_score).lower()}. "
            f"Convex hulls outline clusters; centroids are {sep_hint} relative to typical cluster radii "
            f"(min inter-centroid distance {inter_centroid_min:.2f}, avg radius {avg_radius:.2f}). "
            f"Cluster sizes → {cluster_sizes}."
        )
    except Exception:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap="tab10", s=40, edgecolors="w")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA scatter")
        name, path = save_figure(fig)
        outputs["pca_hull"] = name
        descriptions["pca_hull"] = (
            f"PCA scatter on PC1/PC2 (variance {pc1_var:.1f}% / {pc2_var:.1f}%). k={best_k}."
        )

    # ---------- Scree + elbow
    name, path = plot_pca_scree(pca)
    outputs["pca_scree"] = name
    var_ratio = (pca.explained_variance_ratio_ * 100)
    elbow = detect_elbow(var_ratio)
    cum = np.cumsum(var_ratio)
    n80 = int(np.argmax(cum >= 80) + 1) if (cum >= 80).any() else None
    tail = f"About {n80} PC(s) reach ≥80% cumulative variance. " if n80 else ""
    descriptions["pca_scree"] = (
        f"Scree plot of explained variance. PC1={pc1_var:.1f}%, PC2={pc2_var:.1f}% "
        f"(cum {cum2:.1f}%). Elbow detected near PC{elbow}. {tail}"
        f"Consider using PCs up to the elbow where marginal gains flatten."
    )

    # ---------- Build 2D feature set for supervised models
    X_use = X_pca[:, :2]
    y_true = df_clean[label_col].values if label_col else best_labels
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_use, y_true, test_size=0.25, random_state=42, stratify=y_true
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X_use, y_true, test_size=0.25, random_state=42
        )

    # ---------- SVM
    svm = SVC(kernel="rbf", random_state=42)
    try:
        svm.fit(X_train, y_train)
        preds_svm = svm.predict(X_test)

        (name, path), cm_svm = plot_confusion(y_test, preds_svm, "SVM Confusion Matrix")
        outputs["svm_cm"] = name
        acc = float((preds_svm == y_test).mean())
        # most confused pair
        cm_tmp = cm_svm.copy()
        np.fill_diagonal(cm_tmp, 0)
        i, j = np.unravel_index(np.argmax(cm_tmp), cm_tmp.shape)
        confused_note = f"Most confusion: {i}↔{j} ({int(cm_tmp[i,j])} cases)." if cm_tmp.sum() > 0 else "Minimal confusion."
        descriptions["svm_cm"] = (
            f"SVM (RBF) on PC1/PC2 with test accuracy ≈ {acc:.3f}. {confused_note} "
            f"Diagonal cells are correct predictions; off-diagonals are misclassifications."
        )

        name, path = plot_svm_decision_boundary_2d(
            np.vstack([X_train, X_test]), np.hstack([y_train, y_test]), svm
        )
        outputs["svm_boundary_2d"] = name
        n_sv = int(np.sum(svm.n_support_)) if hasattr(svm, "n_support_") else None
        sv_txt = f"{n_sv} support vectors" if n_sv is not None else "support vectors not available"
        descriptions["svm_boundary_2d"] = (
            f"Decision regions over PC1–PC2. The background shows model predictions; "
            f"points are samples. Model uses {sv_txt}; compact regions imply clearer margins."
        )
    except Exception:
        outputs["svm_cm"] = outputs["svm_boundary_2d"] = None

    # ---------- Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    try:
        dt.fit(X_train, y_train)
        preds_dt = dt.predict(X_test)

        (name, path), cm_dt = plot_confusion(y_test, preds_dt, "Decision Tree Confusion Matrix")
        outputs["dt_cm"] = name
        acc_dt = float((preds_dt == y_test).mean())
        imps = getattr(dt, "feature_importances_", None)
        imp_txt = (
            f" PC importances — PC1: {imps[0]:.2f}, PC2: {imps[1]:.2f}."
            if imps is not None else ""
        )
        descriptions["dt_cm"] = (
            f"Decision Tree on PC1/PC2 with test accuracy ≈ {acc_dt:.3f}.{imp_txt} "
            "Trees capture simple threshold rules; errors cluster where splits are least informative."
        )

        name, path = plot_decision_tree(dt, ["PC1", "PC2"])
        outputs["dt_tree"] = name
        depth = int(getattr(dt, "get_depth")() if hasattr(dt, "get_depth") else 0)
        root_feat = "PC1" if (imps is not None and imps[0] >= imps[1]) else "PC2"
        descriptions["dt_tree"] = (
            f"Tree structure (max depth=5, fitted depth={depth}). The root split likely uses {root_feat}, "
            "creating interpretable rules along principal axes."
        )
    except Exception:
        outputs["dt_cm"] = outputs["dt_tree"] = None

    # ---------- Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)

        (name, path), cm_rf = plot_confusion(y_test, preds_rf, "Random Forest Confusion Matrix")
        outputs["rf_cm"] = name
        acc_rf = float((preds_rf == y_test).mean())
        fi = getattr(rf, "feature_importances_", None)
        fi_txt = (
            f" PC importances — PC1: {fi[0]:.2f}, PC2: {fi[1]:.2f}."
            if fi is not None else ""
        )
        descriptions["rf_cm"] = (
            f"Random Forest (100 trees) on PC1/PC2. Test accuracy ≈ {acc_rf:.3f}.{fi_txt} "
            "Ensembling stabilizes performance versus a single tree."
        )
    except Exception:
        outputs["rf_cm"] = None

    # ---------- 3D projection
    try:
        name, path = plot_svm_3d(X_pca, best_labels)
        outputs["svm_3d"] = name
        pc3_var = float(pca.explained_variance_ratio_[2] * 100) if pca.n_components_ >= 3 else 0.0
        # Does adding PC3 improve separation? (compare silhouette on 2D vs 3D with same labels)
        try:
            sil2 = silhouette_score(X_pca[:, :2], best_labels)
            sil3 = silhouette_score(X_pca[:, :3], best_labels) if X_pca.shape[1] >= 3 else sil2
            imp = "improves" if sil3 - sil2 > 0.02 else "does not noticeably change"
            sep_line = f" Adding PC3 {imp} the silhouette ({sil2:.3f}→{sil3:.3f})."
        except Exception:
            sep_line = ""
        descriptions["svm_3d"] = (
            f"3-D PCA projection (PC1 {pc1_var:.1f}%, PC2 {pc2_var:.1f}%, PC3 {pc3_var:.1f}%)."
            + sep_line
        )
    except Exception:
        outputs["svm_3d"] = None

    metrics = {
        "silhouette": round(best_score, 4) if best_score is not None else None,
        "davies_bouldin": round(dbi, 4) if dbi is not None else None,
        "calinski_harabasz": round(ch, 4) if ch is not None else None,
        "n_clusters": int(best_k),
        "assessment": score_assessment(best_score),
    }

    return render_template("result.html", metrics=metrics, outputs=outputs, descriptions=descriptions)

@app.route("/static/outputs/<path:filename>")
def download_output(filename):
    filepath = os.path.join(OUT_DIR, filename)
    if not os.path.exists(filepath):
        abort(404)
    return send_file(filepath, as_attachment=True)

@app.route("/report.pdf")
def generate_pdf():
    files = sorted(
        [os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR) if f.endswith(".png")],
        key=os.path.getmtime,
    )
    if not files:
        abort(404, description="No output images to include in report.")
    out_pdf = os.path.join(OUT_DIR, f"report_{int(time.time())}.pdf")
    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.8, "Predix - Clustering Report", ha="center", fontsize=20, weight="bold")
        fig.text(0.5, 0.7, f"Generated: {datetime.now().isoformat()}", ha="center", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        for img in files:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_subplot(111)
            ax.axis("off")
            im = plt.imread(img)
            ax.imshow(im)
            pdf.savefig(fig)
            plt.close(fig)
    return send_file(out_pdf, as_attachment=True, download_name=os.path.basename(out_pdf))

# ======================================
# RUN APP
# ======================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9000)
