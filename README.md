# Segmentación de clientes con K-Means

Taller final de minería de datos. El dataset queda en `data/Mall_Customers.csv`. El modelo entrenado está en `models/kmeans_mall.joblib` (pipeline de `StandardScaler` + `KMeans`). El cuaderno `notebooks/segmentacion_clientes.ipynb` recorre las siete partes del enunciado.

## Despliegue con Streamlit

La parte 7 del taller corre en `streamlit_app.py`. Usa el pipeline guardado en `models/kmeans_mall.joblib`.

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

La aplicación queda en `http://localhost:8501`. Ahí se ven el codo, la silueta, los clusters y el formulario que clasifica un cliente nuevo.

Para publicarla en Streamlit Community Cloud hace falta un repositorio de GitHub con este proyecto. En [share.streamlit.io](https://share.streamlit.io) se crea la app con el archivo principal `streamlit_app.py`. No hay secretos ni base de datos. `requirements.txt` es el que instala la nube.

## Cómo correr la API y la otra interfaz

```powershell
.\.venv\Scripts\python.exe ml\train.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
```

En otra terminal, dentro de `frontend`:

```powershell
npm install
npm run dev
```

La interfaz queda en `http://localhost:5173` y la API en `http://localhost:8000`.

## Docker

Desde la raíz del proyecto, con Docker Desktop encendido:

```powershell
docker compose up --build
```

- Interfaz web: `http://localhost:8080`
- Streamlit: `http://localhost:8501`
- API: `http://localhost:8000/api/health`

La imagen del backend incluye el modelo y `artifacts/analysis.json`. Si se vuelve a entrenar, hay que reconstruir la imagen.

## Qué hace falta para publicarlo

No hay claves, base de datos ni servicios externos. Para dejarlo fuera de esta máquina hace falta:

1. Un servidor con Docker y Docker Compose (un VPS, una máquina del laboratorio o un servicio que acepte `docker compose`).
2. Los puertos 8080 y 8000 abiertos, o un proxy que publique solo el puerto 80 hacia el servicio `frontend`.
3. Si se quiere un dominio, el DNS apuntando a esa máquina.

No hace falta volver a entrenar en el servidor: el `joblib` viaja dentro de la imagen.
