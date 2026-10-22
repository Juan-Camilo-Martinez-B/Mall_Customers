"""Entrena la segmentación de clientes del Mall Customers Dataset.

Recorre los pasos del taller: exploración, StandardScaler, método del codo,
coeficiente de silueta, K-Means, perfiles de negocio y persistencia en joblib.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Mall_Customers.csv"
MODEL_PATH = ROOT / "models" / "kmeans_mall.joblib"
ANALYSIS_PATH = ROOT / "artifacts" / "analysis.json"
FIG_DIR = ROOT / "artifacts" / "figuras"

INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"
AGE = "Age"
RANDOM_STATE = 42
K_MIN = 1
K_MAX = 10

FEATURE_SETS = {
    "ingreso_gasto": [INCOME, SPENDING],
    "edad_ingreso_gasto": [AGE, INCOME, SPENDING],
}

PROFILE_CATALOG = {
    "alto_valor": {
        "nombre": "Clientes de alto valor",
        "resumen": (
            "Ingreso alto y puntaje de gasto alto. Compran con frecuencia y "
            "tienen capacidad de pago. Son el núcleo de margen del centro comercial."
        ),
        "diferencia": (
            "Se separan del resto por combinar poder adquisitivo y disposición "
            "real de compra. No dependen de una promoción para visitar la tienda."
        ),
        "estrategias": [
            "Programa de lealtad con beneficios experienciales, no solo descuento.",
            "Acceso anticipado a colecciones y atención personalizada.",
            "Venta cruzada de categorías premium, protegiendo el margen.",
            "Medir recompra y ticket, no solo la tasa de respuesta a cupones.",
        ],
        "oportunidad_crecimiento": False,
    },
    "potencial": {
        "nombre": "Clientes con potencial",
        "resumen": (
            "Ingreso alto y puntaje de gasto bajo. Tienen capacidad de compra "
            "que hoy no se está capturando en el centro comercial."
        ),
        "diferencia": (
            "Su ingreso se parece al de los clientes de alto valor, pero su "
            "gasto se parece al de los clientes de bajo valor. Esa brecha es la oportunidad."
        ),
        "estrategias": [
            "Entender la barrera: surtido, experiencia, horario o relevancia de marca.",
            "Campañas de descubrimiento y prueba de producto, con asesoría.",
            "Invitaciones a categorías afines a su perfil, sin descuentos agresivos.",
            "Seguimiento del segundo ticket: el objetivo es convertir visita en hábito.",
        ],
        "oportunidad_crecimiento": True,
    },
    "promedio": {
        "nombre": "Clientes promedio",
        "resumen": (
            "Ingreso y gasto cercanos a la media del mall. Son la base estable "
            "de tráfico y sostienen la operación cotidiana."
        ),
        "diferencia": (
            "No extremizan ninguna variable. Forman el grupo central y más "
            "homogéneo alrededor de los promedios de ingreso y gasto."
        ),
        "estrategias": [
            "Comunicación de valor y promociones estacionales moderadas.",
            "Programa de puntos para aumentar la frecuencia sin erosionar precio.",
            "Recomendaciones de complemento según la categoría que ya compran.",
            "Cuidar la experiencia básica: filas, disponibilidad y claridad de precio.",
        ],
        "oportunidad_crecimiento": False,
    },
    "bajo_valor": {
        "nombre": "Clientes de bajo valor",
        "resumen": (
            "Ingreso bajo y puntaje de gasto bajo. Su contribución actual al "
            "ticket es limitada y el costo de adquirirlos debe mantenerse bajo."
        ),
        "diferencia": (
            "Ocupan la esquina de menor ingreso y menor gasto. No se confunden "
            "con quienes gastan mucho a pesar de un ingreso ajustado."
        ),
        "estrategias": [
            "Formatos de entrada, presentaciones pequeñas y comunicación de precio.",
            "Canales digitales de bajo costo en lugar de asesoría presencial larga.",
            "Ofertas puntuales de tráfico, separadas de la comunicación premium.",
            "No asignar el mismo presupuesto de retención que al segmento de alto valor.",
        ],
        "oportunidad_crecimiento": False,
    },
    "entusiastas": {
        "nombre": "Clientes entusiastas",
        "resumen": (
            "Ingreso bajo o moderado y puntaje de gasto alto. Compran con "
            "intensidad pese a una restricción de presupuesto."
        ),
        "diferencia": (
            "Su gasto se parece al de los clientes de alto valor, pero su "
            "ingreso no. Son sensibles a la oferta y muy activos."
        ),
        "estrategias": [
            "Bundles, cuotas y recompensas por frecuencia de visita.",
            "Promociones con fecha, evitando que el precio pleno deje de existir.",
            "Programas de referidos: suelen recomendar lo que compran.",
            "Vigilar el margen por campaña para no subsidiar todo el ticket.",
        ],
        "oportunidad_crecimiento": False,
    },
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    expected = {"CustomerID", "Gender", AGE, INCOME, SPENDING}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Al dataset le faltan columnas: {sorted(missing)}")
    return df


def iqr_outliers(series: pd.Series) -> dict:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask = (series < low) | (series > high)
    values = [float(v) for v in series.loc[mask].tolist()]
    return {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "limite_inferior": round(low, 2),
        "limite_superior": round(high, 2),
        "n": int(mask.sum()),
        "valores": values,
    }


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    model.fit(X)
    return model


def knee_k(ks: list[int], inertias: list[float]) -> int:
    """Punto del codo: máxima distancia perpendicular a la recta entre extremos."""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    start = np.array([x_n[0], y_n[0]])
    end = np.array([x_n[-1], y_n[-1]])
    line = end - start
    length = np.linalg.norm(line)
    distances = []
    for i in range(len(x_n)):
        point = np.array([x_n[i], y_n[i]])
        cross = line[0] * (start[1] - point[1]) - (start[0] - point[0]) * line[1]
        distances.append(abs(cross) / length)
    return int(ks[int(np.argmax(distances))])


def evaluate_feature_set(df: pd.DataFrame, features: list[str]) -> dict:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    curve = []
    models: dict[int, KMeans] = {}
    for k in range(K_MIN, K_MAX + 1):
        model = fit_kmeans(scaled, k)
        models[k] = model
        item = {"k": k, "inertia": round(float(model.inertia_), 4)}
        if k >= 2:
            labels = model.labels_
            item["silhouette"] = round(float(silhouette_score(scaled, labels)), 4)
        else:
            item["silhouette"] = None
        curve.append(item)

    elbow = knee_k([c["k"] for c in curve], [c["inertia"] for c in curve])
    sil_curve = [c for c in curve if c["silhouette"] is not None]
    best_sil = max(sil_curve, key=lambda c: c["silhouette"])
    return {
        "features": features,
        "curve": curve,
        "elbow_k": elbow,
        "best_silhouette_k": int(best_sil["k"]),
        "best_silhouette": best_sil["silhouette"],
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "scaled": scaled,
        "models": models,
    }


def level(z: float) -> str:
    if z >= 0.55:
        return "alto"
    if z <= -0.55:
        return "bajo"
    return "medio"


def profile_key(income_z: float, spending_z: float) -> str:
    income_level = level(income_z)
    spending_level = level(spending_z)
    mapping = {
        ("alto", "alto"): "alto_valor",
        ("alto", "bajo"): "potencial",
        ("alto", "medio"): "potencial",
        ("bajo", "bajo"): "bajo_valor",
        ("bajo", "alto"): "entusiastas",
        ("bajo", "medio"): "bajo_valor",
        ("medio", "alto"): "entusiastas",
        ("medio", "bajo"): "potencial",
        ("medio", "medio"): "promedio",
    }
    return mapping[(income_level, spending_level)]


def build_clusters(df: pd.DataFrame, labels: np.ndarray, centroids_original: np.ndarray) -> list[dict]:
    inc_mean = float(df[INCOME].mean())
    spd_mean = float(df[SPENDING].mean())
    inc_std = float(df[INCOME].std(ddof=0))
    spd_std = float(df[SPENDING].std(ddof=0))
    framed = df.copy()
    framed["cluster"] = labels
    clusters = []
    for cluster_id in sorted(np.unique(labels)):
        part = framed.loc[framed["cluster"] == cluster_id]
        income_c = float(centroids_original[cluster_id, 0])
        spending_c = float(centroids_original[cluster_id, 1])
        key = profile_key((income_c - inc_mean) / inc_std, (spending_c - spd_mean) / spd_std)
        catalog = PROFILE_CATALOG[key]
        gender = part["Gender"].value_counts(normalize=True)
        clusters.append(
            {
                "id": int(cluster_id),
                "profile_key": key,
                "nombre": catalog["nombre"],
                "resumen": catalog["resumen"],
                "diferencia": catalog["diferencia"],
                "estrategias": catalog["estrategias"],
                "oportunidad_crecimiento": catalog["oportunidad_crecimiento"],
                "n": int(len(part)),
                "pct": round(100 * len(part) / len(framed), 1),
                "centroid": {
                    "annual_income": round(income_c, 2),
                    "spending_score": round(spending_c, 2),
                },
                "stats": {
                    "age_mean": round(float(part[AGE].mean()), 2),
                    "age_min": int(part[AGE].min()),
                    "age_max": int(part[AGE].max()),
                    "income_mean": round(float(part[INCOME].mean()), 2),
                    "income_min": float(part[INCOME].min()),
                    "income_max": float(part[INCOME].max()),
                    "spending_mean": round(float(part[SPENDING].mean()), 2),
                    "spending_min": float(part[SPENDING].min()),
                    "spending_max": float(part[SPENDING].max()),
                    "female_pct": round(100 * float(gender.get("Female", 0)), 1),
                    "male_pct": round(100 * float(gender.get("Male", 0)), 1),
                },
            }
        )
    return clusters


def describe_frame(df: pd.DataFrame) -> dict:
    numeric = df[[AGE, INCOME, SPENDING]]
    summary = numeric.describe().round(2)
    table = {"estadistico": summary.index.tolist()}
    for column in summary.columns:
        table[column] = [float(v) for v in summary[column].tolist()]
    return table


def customers_payload(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    rows = []
    for idx in range(len(df)):
        record = df.iloc[idx]
        rows.append(
            {
                "customer_id": int(record["CustomerID"]),
                "gender": record["Gender"],
                "age": int(record[AGE]),
                "annual_income": float(record[INCOME]),
                "spending_score": int(record[SPENDING]),
                "cluster": int(labels[idx]),
            }
        )
    return rows


def build_answers(context: dict) -> dict:
    corr = context["corr"]
    chosen = context["chosen"]
    other = context["other"]
    outliers = context["outliers"]
    selected_k = context["selected_k"]
    sil = context["silhouette"]
    coincide = chosen["elbow_k"] == chosen["best_silhouette_k"]
    outlier_bits = []
    for name, info in outliers.items():
        if info["n"] == 0:
            outlier_bits.append(f"{name} no tiene valores fuera de las vallas de Tukey")
        else:
            outlier_bits.append(
                f"{name} tiene {info['n']} valores fuera del rango "
                f"[{info['limite_inferior']}, {info['limite_superior']}]: {info['valores']}"
            )
    coincide_text = (
        f"Sí. El codo y la silueta señalan k = {selected_k} para ingreso y gasto."
        if coincide
        else (
            f"No de forma exacta. El codo geométrico señala k = {chosen['elbow_k']} "
            f"y la silueta máxima está en k = {chosen['best_silhouette_k']} "
            f"({chosen['best_silhouette']}). Se adopta k = {selected_k} porque la silueta "
            "mide separación entre grupos y no solo la caída de la inercia. "
            "La inercia sigue bajando al aumentar k, pero el salto de calidad se aplana "
            "después del k elegido."
        )
    )
    return {
        "variables": (
            "Las variables más adecuadas para segmentar son el ingreso anual (k$) "
            "y el puntaje de gasto (1-100). Describen capacidad de pago y comportamiento "
            f"de compra. Su correlación lineal es {corr[INCOME][SPENDING]:.3f}: casi no hay "
            "una recta que las una, y justamente por eso aparecen grupos en el plano. "
            f"La edad correlaciona {corr[AGE][SPENDING]:.3f} con el gasto y "
            f"{corr[AGE][INCOME]:.3f} con el ingreso; aporta contexto, pero al meterla "
            f"en K-Means la mejor silueta baja de {chosen['best_silhouette']:.3f} "
            f"(k = {chosen['best_silhouette_k']}) a {other['best_silhouette']:.3f} "
            f"(k = {other['best_silhouette_k']}). El género es categórico y no entra en "
            "la distancia euclidiana; CustomerID solo identifica la fila."
        ),
        "atipicos": (
            "Con la regla del rango intercuartílico, "
            + "; ".join(outlier_bits)
            + ". Los ingresos de 137 k$ corresponden a clientes reales de cola alta, "
            "no a errores de captura, así que se conservan: en marketing esa cola es "
            "parte del segmento de alto ingreso."
        ),
        "relaciones": (
            f"Edad y gasto se mueven en sentido contrario (r = {corr[AGE][SPENDING]:.3f}): "
            "los puntajes altos se concentran en clientes más jóvenes. Edad e ingreso "
            f"casi no se alinean (r = {corr[AGE][INCOME]:.3f}). Ingreso y gasto tampoco "
            f"(r = {corr[INCOME][SPENDING]:.3f}). La relación útil no es lineal: en el "
            "plano ingreso-gasto se ven cinco nubes, no una pendiente."
        ),
        "escalar": (
            "K-Means asigna cada punto al centroide más cercano con distancia euclidiana. "
            "Si una variable tiene mayor dispersión o está en unidades más grandes, "
            "domina esa distancia y el resto casi no influye. StandardScaler resta la "
            "media y divide por la desviación estándar, de modo que cada variable queda "
            "con media 0 y desviación 1. En este archivo el ingreso ya viene en miles "
            "de dólares y su escala se parece a la del puntaje de gasto, así que el "
            "efecto numérico es moderado; igual se escala, porque es la condición para "
            "que las dos variables pesen lo mismo y porque al probar la edad (otra "
            "unidad y otra dispersión) la comparación entre modelos sería inválida "
            "sin esa normalización."
        ),
        "inercia_cero": (
            "La inercia es la suma de distancias al cuadrado de cada cliente a su "
            "centroide. Un valor cercano a cero indica que los puntos quedaron muy "
            "pegados a su centro: el ajuste dentro del grupo es alto. No conviene "
            "perseguir el cero a cualquier costo: con k igual al número de clientes "
            "la inercia es cero y el modelo no resume nada. El codo busca el k donde "
            "la inercia ya bajó bastante y los siguientes grupos aportan poco."
        ),
        "punto_optimo": (
            "El punto óptimo es el codo de la curva: se traza la inercia para k de "
            f"{K_MIN} a {K_MAX} y se localiza donde la curva deja de caer en picada "
            "y pasa a un descenso suave. De forma operativa se midió la distancia "
            "de cada punto a la recta que une el primer y el último k. El k más "
            f"alejado de esa recta es {chosen['elbow_k']}."
        ),
        "silueta_mide": (
            "El coeficiente de silueta compara, para cada cliente, qué tan cerca está "
            "de su propio grupo (cohesión, a) y qué tan lejos está del grupo vecino "
            "más próximo (separación, b). El valor del cliente es (b - a) / max(a, b) "
            "y el coeficiente del modelo es el promedio."
        ),
        "silueta_uno": (
            "Un valor cercano a 1 significa que los clientes están juntos dentro de "
            "su segmento y lejos de los demás. Los grupos se distinguen con claridad."
        ),
        "silueta_cero": (
            "Un valor cercano a 0 indica solapamiento: la distancia al propio centro "
            "y al grupo vecino son parecidas, así que la frontera es débil. Un valor "
            "negativo indica que, en promedio, hay clientes más cerca de otro grupo "
            "que del suyo; la asignación es pobre."
        ),
        "coincide": (
            f"{coincide_text} La silueta del modelo elegido es {sil:.4f}. "
            "Quedarse en un k menor mezclaría nubes que el plano muestra separadas "
            "(en especial a quien gasta mucho y a quien gasta poco dentro del mismo "
            "nivel de ingreso). Subir de k parte grupos que ya son compactos y baja "
            "o estanca la silueta."
        ),
    }


def save_figures(df: pd.DataFrame, labels: np.ndarray, clusters: list[dict], chosen: dict, other: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    colors = {
        "alto_valor": "#0e6b63",
        "potencial": "#b5522a",
        "promedio": "#2f5d8c",
        "bajo_valor": "#6e6258",
        "entusiastas": "#a33b5a",
    }
    color_of = {c["id"]: colors[c["profile_key"]] for c in clusters}

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ks = [c["k"] for c in chosen["curve"]]
    inertias = [c["inertia"] for c in chosen["curve"]]
    ax.plot(ks, inertias, marker="o", color="#0e6b63")
    ax.axvline(chosen["elbow_k"], color="#b5522a", ls="--", label=f"codo k={chosen['elbow_k']}")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("Método del codo — ingreso y gasto")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "codo.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sil_items = [c for c in chosen["curve"] if c["silhouette"] is not None]
    ax.bar([c["k"] for c in sil_items], [c["silhouette"] for c in sil_items], color="#2f5d8c")
    ax.set_xlabel("k")
    ax.set_ylabel("Coeficiente de silueta")
    ax.set_title("Silueta para ingreso y gasto")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silueta.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for cluster in clusters:
        part = df.loc[labels == cluster["id"]]
        ax.scatter(
            part[INCOME],
            part[SPENDING],
            s=36,
            color=color_of[cluster["id"]],
            label=f"{cluster['nombre']} (n={cluster['n']})",
            alpha=0.9,
        )
        ax.scatter(
            cluster["centroid"]["annual_income"],
            cluster["centroid"]["spending_score"],
            s=160,
            marker="X",
            c="black",
        )
    ax.set_xlabel("Ingreso anual (k$)")
    ax.set_ylabel("Puntaje de gasto (1-100)")
    ax.set_title("Segmentos K-Means")
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clusters.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5))
    sns.heatmap(
        df[[AGE, INCOME, SPENDING]].corr(),
        annot=True,
        fmt=".2f",
        cmap="BrBG",
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlación entre variables numéricas")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlacion.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4))
    for ax, curve, title in (
        (axes[0], chosen, "Ingreso y gasto"),
        (axes[1], other, "Edad, ingreso y gasto"),
    ):
        items = [c for c in curve["curve"] if c["silhouette"] is not None]
        ax.plot([c["k"] for c in items], [c["silhouette"] for c in items], marker="o")
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.set_ylabel("Silueta")
    fig.suptitle("Comparación de conjuntos de variables")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparacion_silueta.png", dpi=140)
    plt.close(fig)


def strip_runtime(result: dict) -> dict:
    return {
        "features": result["features"],
        "curve": result["curve"],
        "elbow_k": result["elbow_k"],
        "best_silhouette_k": result["best_silhouette_k"],
        "best_silhouette": result["best_silhouette"],
        "scaler_mean": result["scaler_mean"],
        "scaler_scale": result["scaler_scale"],
    }


def main() -> None:
    df = load_data()
    numeric = df[[AGE, INCOME, SPENDING]]
    corr = numeric.corr().round(4)
    outliers = {column: iqr_outliers(df[column]) for column in (AGE, INCOME, SPENDING)}

    evaluated = {name: evaluate_feature_set(df, features) for name, features in FEATURE_SETS.items()}
    chosen_name = "ingreso_gasto"
    chosen = evaluated[chosen_name]
    other = evaluated["edad_ingreso_gasto"]

    if chosen["elbow_k"] == chosen["best_silhouette_k"]:
        selected_k = chosen["elbow_k"]
    else:
        selected_k = chosen["best_silhouette_k"]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=selected_k,
                    init="k-means++",
                    n_init=10,
                    max_iter=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    labels = pipeline.fit_predict(df[[INCOME, SPENDING]])
    scaler = pipeline.named_steps["scaler"]
    kmeans = pipeline.named_steps["kmeans"]
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    silhouette = float(silhouette_score(scaler.transform(df[[INCOME, SPENDING]]), labels))
    clusters = build_clusters(df, labels, centroids)

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": [INCOME, SPENDING],
            "k": selected_k,
            "random_state": RANDOM_STATE,
            "cluster_ids": [c["id"] for c in clusters],
        },
        MODEL_PATH,
    )

    growth = next(c for c in clusters if c["oportunidad_crecimiento"])
    answers = build_answers(
        {
            "corr": corr,
            "chosen": chosen,
            "other": other,
            "outliers": outliers,
            "selected_k": selected_k,
            "silhouette": silhouette,
        }
    )
    save_figures(df, labels, clusters, chosen, other)

    analysis = {
        "dataset": {
            "nombre": "Mall Customers",
            "archivo": "data/Mall_Customers.csv",
            "fuente": "https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python",
            "n_filas": int(len(df)),
            "n_columnas": int(df.shape[1]),
            "columnas": df.columns.tolist(),
            "nulos": {column: int(df[column].isna().sum()) for column in df.columns},
            "genero": {key: int(value) for key, value in df["Gender"].value_counts().items()},
            "descriptivos": describe_frame(df),
            "correlacion": {
                "columnas": [AGE, INCOME, SPENDING],
                "matriz": corr.values.round(4).tolist(),
            },
            "atipicos": outliers,
        },
        "conjuntos": {
            "ingreso_gasto": strip_runtime(chosen),
            "edad_ingreso_gasto": strip_runtime(other),
        },
        "features": [INCOME, SPENDING],
        "selected_k": selected_k,
        "inertia": round(float(kmeans.inertia_), 4),
        "silhouette": round(silhouette, 4),
        "scaler": {
            "features": [INCOME, SPENDING],
            "mean": [round(float(v), 4) for v in scaler.mean_],
            "scale": [round(float(v), 4) for v in scaler.scale_],
        },
        "clusters": clusters,
        "customers": customers_payload(df, labels),
        "crecimiento": {
            "cluster_id": growth["id"],
            "profile_key": growth["profile_key"],
            "nombre": growth["nombre"],
            "razon": (
                "Es el grupo con ingreso alto y gasto bajo. Cada punto de gasto "
                "que se les active vale más que el mismo esfuerzo sobre quien ya "
                "gasta mucho o sobre quien tiene poco margen de ingreso."
            ),
        },
        "respuestas": answers,
        "modelo": "models/kmeans_mall.joblib",
    }
    ANALYSIS_PATH.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo guardado en {MODEL_PATH}")
    print(f"k = {selected_k} | silueta = {silhouette:.4f} | inercia = {kmeans.inertia_:.2f}")
    for cluster in clusters:
        print(
            f"  [{cluster['id']}] {cluster['nombre']}: n={cluster['n']} "
            f"ingreso={cluster['centroid']['annual_income']} "
            f"gasto={cluster['centroid']['spending_score']}"
        )


if __name__ == "__main__":
    main()
