"""Despliegue del taller: segmentación de clientes con el K-Means guardado."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "kmeans_mall.joblib"
ANALYSIS_PATH = ROOT / "artifacts" / "analysis.json"
INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"

PALETTE = {
    "alto_valor": "#0e6b63",
    "potencial": "#b5522a",
    "promedio": "#2f5d8c",
    "bajo_valor": "#6e6258",
    "entusiastas": "#a33b5a",
}

SECCIONES = [
    "Exploración",
    "Preprocesamiento",
    "Método del codo",
    "Silueta",
    "Segmentos",
    "Negocio",
    "Clasificar",
]

EJEMPLOS = [
    {"etiqueta": "Alto valor", "gender": "Female", "age": 32, "annual_income": 90, "spending_score": 85},
    {"etiqueta": "Potencial", "gender": "Male", "age": 40, "annual_income": 95, "spending_score": 18},
    {"etiqueta": "Promedio", "gender": "Female", "age": 45, "annual_income": 55, "spending_score": 50},
    {"etiqueta": "Bajo valor", "gender": "Male", "age": 48, "annual_income": 25, "spending_score": 18},
    {"etiqueta": "Entusiasta", "gender": "Female", "age": 23, "annual_income": 25, "spending_score": 82},
]


st.set_page_config(page_title="Segmentación de clientes", layout="wide")


@st.cache_resource
def cargar_modelo() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def cargar_analisis() -> dict:
    return json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))


def color_cluster(cluster: dict) -> str:
    return PALETTE[cluster["profile_key"]]


def etiqueta_genero(valor: str) -> str:
    return "Mujer" if valor == "Female" else "Hombre"


def plano(texto: str) -> str:
    """Evita que Streamlit interprete el signo de dólares como fórmula."""
    return str(texto).replace("$", r"\$")


def respuesta(titulo: str, texto: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{plano(titulo)}**")
        st.markdown(plano(texto))


def figura_codo(curve: list[dict], selected_k: int):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([item["k"] for item in curve], [item["inertia"] for item in curve], marker="o", color="#0e6b63")
    ax.axvline(selected_k, color="#b5522a", ls="--", label=f"k = {selected_k}")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia (WCSS)")
    ax.set_xticks([item["k"] for item in curve])
    ax.legend()
    fig.tight_layout()
    return fig


def figura_silueta(curve: list[dict], selected_k: int):
    items = [item for item in curve if item["silhouette"] is not None]
    colores = ["#0e6b63" if item["k"] == selected_k else "#8eabc8" for item in items]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([item["k"] for item in items], [item["silhouette"] for item in items], color=colores)
    ax.set_xlabel("k")
    ax.set_ylabel("Coeficiente de silueta")
    ax.set_xticks([item["k"] for item in items])
    fig.tight_layout()
    return fig


def figura_clusters(customers: list[dict], clusters: list[dict], nuevo: dict | None = None):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    por_id = {cluster["id"]: cluster for cluster in clusters}
    for cluster in clusters:
        puntos = [row for row in customers if row["cluster"] == cluster["id"]]
        ax.scatter(
            [row["annual_income"] for row in puntos],
            [row["spending_score"] for row in puntos],
            s=36,
            color=color_cluster(cluster),
            label=f"{cluster['nombre']} (n={cluster['n']})",
            alpha=0.9,
        )
        ax.scatter(
            cluster["centroid"]["annual_income"],
            cluster["centroid"]["spending_score"],
            marker="X",
            s=140,
            c="black",
        )
    if nuevo is not None:
        perfil = por_id[nuevo["cluster"]]
        ax.scatter(
            [nuevo["entrada"]["annual_income"]],
            [nuevo["entrada"]["spending_score"]],
            s=160,
            facecolors="none",
            edgecolors=color_cluster(perfil),
            linewidths=2.4,
            label="Cliente nuevo",
            zorder=5,
        )
    ax.set_xlabel("Ingreso anual (k$)")
    ax.set_ylabel("Puntaje de gasto (1-100)")
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def clasificar(modelo: dict, analisis: dict, gender: str, age: int, income: float, spending: float) -> dict:
    pipeline = modelo["pipeline"]
    features = modelo["features"]
    frame = pd.DataFrame([{INCOME: income, SPENDING: spending}], columns=features)
    cluster_id = int(pipeline.predict(frame)[0])
    scaled = pipeline.named_steps["scaler"].transform(frame)
    centro = pipeline.named_steps["kmeans"].cluster_centers_[cluster_id]
    distancia = float(np.linalg.norm(scaled[0] - centro))
    perfil = next(item for item in analisis["clusters"] if item["id"] == cluster_id)
    return {
        "cluster": cluster_id,
        "distancia": round(distancia, 4),
        "perfil": perfil,
        "entrada": {
            "gender": gender,
            "age": age,
            "annual_income": income,
            "spending_score": spending,
        },
    }


def seccion_exploracion(analisis: dict) -> None:
    datos = analisis["dataset"]
    descriptivos = datos["descriptivos"]
    st.subheader("Exploración de datos")
    st.write(
        f"El archivo tiene {datos['n_filas']} filas y {datos['n_columnas']} columnas, sin valores nulos. "
        "CustomerID solo identifica al cliente. La distancia del modelo usa ingreso y gasto."
    )
    columnas = st.columns(5)
    columnas[0].metric("Mujeres", datos["genero"]["Female"])
    columnas[1].metric("Hombres", datos["genero"]["Male"])
    columnas[2].metric("Edad media", descriptivos["Age"][1])
    columnas[3].metric("Ingreso medio (k$)", descriptivos[INCOME][1])
    columnas[4].metric("Gasto medio", descriptivos[SPENDING][1])

    tabla = pd.DataFrame(
        {
            "Estadístico": descriptivos["estadistico"],
            "Edad": descriptivos["Age"],
            "Ingreso anual (k$)": descriptivos[INCOME],
            "Puntaje de gasto": descriptivos[SPENDING],
        }
    )
    st.dataframe(tabla, hide_index=True, width="stretch")

    corr = pd.DataFrame(
        datos["correlacion"]["matriz"],
        index=["Edad", "Ingreso", "Gasto"],
        columns=["Edad", "Ingreso", "Gasto"],
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="BrBG", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlación")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    atipicos = pd.DataFrame(
        [
            {
                "Variable": nombre,
                "Límite inferior": info["limite_inferior"],
                "Límite superior": info["limite_superior"],
                "Cantidad": info["n"],
                "Valores": ", ".join(str(valor) for valor in info["valores"]) or "—",
            }
            for nombre, info in datos["atipicos"].items()
        ]
    )
    st.markdown("**Valores atípicos (rango intercuartílico)**")
    st.dataframe(atipicos, hide_index=True, width="stretch")
    respuesta("¿Qué variables son más adecuadas para segmentar clientes?", analisis["respuestas"]["variables"])
    respuesta("¿Se identifican valores atípicos?", analisis["respuestas"]["atipicos"])
    respuesta("¿Qué tipo de relaciones se observan entre las variables?", analisis["respuestas"]["relaciones"])


def seccion_escala(analisis: dict) -> None:
    st.subheader("StandardScaler")
    st.write(
        "Antes de medir distancias, ingreso y gasto quedan con media 0 y desviación 1. "
        "Esas medias viajan dentro del pipeline y se aplican al clasificar un cliente nuevo."
    )
    scaler = analisis["scaler"]
    st.dataframe(
        pd.DataFrame(
            {
                "Variable": scaler["features"],
                "Media de entrenamiento": scaler["mean"],
                "Desviación estándar": scaler["scale"],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    elegido = analisis["conjuntos"]["ingreso_gasto"]
    alterno = analisis["conjuntos"]["edad_ingreso_gasto"]
    izquierda, derecha = st.columns(2)
    izquierda.metric("Silueta ingreso y gasto", f"{elegido['best_silhouette']:.3f}")
    izquierda.caption(f"Mejor k = {elegido['best_silhouette_k']}. Codo en k = {elegido['elbow_k']}.")
    derecha.metric("Silueta con la edad", f"{alterno['best_silhouette']:.3f}")
    derecha.caption(f"Mejor k = {alterno['best_silhouette_k']}. Codo en k = {alterno['elbow_k']}.")
    respuesta("¿Por qué es necesario escalar las variables en K-Means?", analisis["respuestas"]["escalar"])


def seccion_codo(analisis: dict) -> None:
    curva = analisis["conjuntos"]["ingreso_gasto"]["curve"]
    st.subheader("Método del codo")
    st.write(
        f"La inercia baja rápido hasta k = {analisis['selected_k']} y después cada cluster adicional recorta poco error."
    )
    fig = figura_codo(curva, analisis["selected_k"])
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(
        pd.DataFrame({"k": [item["k"] for item in curva], "Inercia": [item["inertia"] for item in curva]}),
        hide_index=True,
        width="stretch",
    )
    respuesta("¿Por qué valores de inercia cercanos a cero indican mejor ajuste?", analisis["respuestas"]["inercia_cero"])
    respuesta("¿Cómo se identifica el punto óptimo de k?", analisis["respuestas"]["punto_optimo"])


def seccion_silueta(analisis: dict) -> None:
    conjunto = analisis["conjuntos"]["ingreso_gasto"]
    st.subheader("Coeficiente de silueta")
    st.write(
        f"El máximo está en k = {conjunto['best_silhouette_k']}, con {conjunto['best_silhouette']:.3f}. "
        f"El modelo guardado alcanza {analisis['silhouette']:.3f}."
    )
    fig = figura_silueta(conjunto["curve"], analisis["selected_k"])
    st.pyplot(fig)
    plt.close(fig)
    respuesta("¿Qué mide el coeficiente de silueta?", analisis["respuestas"]["silueta_mide"])
    respuesta("¿Qué significa un valor cercano a 1?", analisis["respuestas"]["silueta_uno"])
    respuesta("¿Qué implica un valor cercano a 0 o negativo?", analisis["respuestas"]["silueta_cero"])
    respuesta("¿Coincide el k óptimo de la silueta con el método del codo?", analisis["respuestas"]["coincide"])


def seccion_segmentos(analisis: dict, nuevo: dict | None) -> None:
    st.subheader("Modelo K-Means")
    st.write("Cada punto es un cliente. La X negra es el centroide, de vuelta en las unidades originales.")
    fig = figura_clusters(analisis["customers"], analisis["clusters"], nuevo)
    st.pyplot(fig)
    plt.close(fig)
    filas = []
    for cluster in analisis["clusters"]:
        filas.append(
            {
                "Perfil": cluster["nombre"],
                "Cluster": cluster["id"],
                "Clientes": cluster["n"],
                "Porcentaje": cluster["pct"],
                "Ingreso centroide": cluster["centroid"]["annual_income"],
                "Gasto centroide": cluster["centroid"]["spending_score"],
                "Edad media": cluster["stats"]["age_mean"],
                "Mujeres %": cluster["stats"]["female_pct"],
            }
        )
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")


def seccion_negocio(analisis: dict) -> None:
    st.subheader("Interpretación de negocio")
    crecimiento = analisis["crecimiento"]
    st.info(f"Mayor oportunidad de crecimiento: {crecimiento['nombre']}. {crecimiento['razon']}")
    for cluster in analisis["clusters"]:
        with st.container(border=True):
            st.markdown(f"### {cluster['nombre']}")
            st.write(cluster["resumen"])
            st.write(cluster["diferencia"])
            for estrategia in cluster["estrategias"]:
                st.markdown(f"- {estrategia}")


def seccion_clasificar(modelo: dict, analisis: dict) -> dict | None:
    st.subheader("Clasificar un cliente nuevo")
    st.write(
        "El pipeline aplica el mismo StandardScaler y asigna el centroide más cercano. "
        "Edad y género quedan en la ficha; no entran en la distancia."
    )
    if "gender" not in st.session_state:
        st.session_state.gender = "Female"
        st.session_state.age = 32
        st.session_state.annual_income = 70.0
        st.session_state.spending_score = 60.0

    st.caption("Probar un perfil conocido")
    columnas = st.columns(len(EJEMPLOS))
    for columna, ejemplo in zip(columnas, EJEMPLOS):
        if columna.button(ejemplo["etiqueta"], width="stretch"):
            st.session_state.gender = ejemplo["gender"]
            st.session_state.age = ejemplo["age"]
            st.session_state.annual_income = float(ejemplo["annual_income"])
            st.session_state.spending_score = float(ejemplo["spending_score"])
            st.rerun()

    with st.form("cliente_nuevo"):
        izquierda, derecha = st.columns(2)
        gender = izquierda.selectbox("Género", ["Female", "Male"], format_func=etiqueta_genero, key="gender")
        age = derecha.number_input("Edad", min_value=15, max_value=100, step=1, key="age")
        income = izquierda.number_input("Ingreso anual (k$)", min_value=0.0, max_value=300.0, step=1.0, key="annual_income")
        spending = derecha.number_input("Puntaje de gasto (1-100)", min_value=1.0, max_value=100.0, step=1.0, key="spending_score")
        enviar = st.form_submit_button("Asignar segmento", type="primary")

    if enviar:
        st.session_state.prediccion = clasificar(modelo, analisis, gender, int(age), float(income), float(spending))

    prediccion = st.session_state.get("prediccion")
    if prediccion:
        perfil = prediccion["perfil"]
        entrada = prediccion["entrada"]
        st.success(perfil["nombre"])
        st.markdown(
            plano(
                f"{etiqueta_genero(entrada['gender'])}, {entrada['age']} años, "
                f"ingreso {entrada['annual_income']} k$, gasto {entrada['spending_score']}. "
                f"Distancia al centroide en el espacio escalado: {prediccion['distancia']}."
            )
        )
        st.write(perfil["resumen"])
        fig = figura_clusters(analisis["customers"], analisis["clusters"], prediccion)
        st.pyplot(fig)
        plt.close(fig)
    return prediccion


def main() -> None:
    if not MODEL_PATH.exists() or not ANALYSIS_PATH.exists():
        st.error("Falta el modelo o el análisis. Ejecuta primero: python ml/train.py")
        st.stop()

    modelo = cargar_modelo()
    analisis = cargar_analisis()
    st.sidebar.markdown("### Mall Customers")
    st.sidebar.caption(f"{analisis['dataset']['n_filas']} clientes · k = {analisis['selected_k']} · silueta {analisis['silhouette']:.3f}")
    seccion = st.sidebar.radio("Taller", SECCIONES)
    st.title("Segmentación de clientes con K-Means")

    if seccion == "Exploración":
        seccion_exploracion(analisis)
    elif seccion == "Preprocesamiento":
        seccion_escala(analisis)
    elif seccion == "Método del codo":
        seccion_codo(analisis)
    elif seccion == "Silueta":
        seccion_silueta(analisis)
    elif seccion == "Segmentos":
        seccion_segmentos(analisis, st.session_state.get("prediccion"))
    elif seccion == "Negocio":
        seccion_negocio(analisis)
    else:
        seccion_clasificar(modelo, analisis)


if __name__ == "__main__":
    main()
