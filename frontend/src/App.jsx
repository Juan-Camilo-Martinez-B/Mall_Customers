import { useEffect, useState } from "react";
import { getAnalysis, getCustomers, predictClient } from "./api";
import { ClusterScatter, ElbowChart, SilhouetteChart, colorOf } from "./components/Charts";

const NAV = [
  ["exploracion", "01", "Exploración"],
  ["escala", "02", "Preprocesamiento"],
  ["codo", "03", "Método del codo"],
  ["silueta", "04", "Silueta"],
  ["segmentos", "05", "Segmentos"],
  ["negocio", "06", "Negocio"],
  ["clasificar", "07", "Clasificar"],
];

const QUESTIONS = [
  ["variables", "¿Qué variables son más adecuadas para segmentar clientes?"],
  ["atipicos", "¿Se identifican valores atípicos?"],
  ["relaciones", "¿Qué tipo de relaciones se observan entre las variables?"],
  ["escalar", "¿Por qué es necesario escalar las variables en K-Means?"],
  ["inercia_cero", "¿Por qué valores de inercia cercanos a cero indican mejor ajuste?"],
  ["punto_optimo", "¿Cómo se identifica el punto óptimo de k?"],
  ["silueta_mide", "¿Qué mide el coeficiente de silueta?"],
  ["silueta_uno", "¿Qué significa un valor cercano a 1?"],
  ["silueta_cero", "¿Qué implica un valor cercano a 0 o negativo?"],
  ["coincide", "¿Coincide el k óptimo de la silueta con el método del codo?"],
];

const EXAMPLES = [
  { label: "Alto valor", gender: "Female", age: 32, annual_income: 90, spending_score: 85 },
  { label: "Potencial", gender: "Male", age: 40, annual_income: 95, spending_score: 18 },
  { label: "Promedio", gender: "Female", age: 45, annual_income: 55, spending_score: 50 },
  { label: "Bajo valor", gender: "Male", age: 48, annual_income: 25, spending_score: 18 },
  { label: "Entusiasta", gender: "Female", age: 23, annual_income: 25, spending_score: 82 },
];

const EMPTY_FORM = {
  gender: "Female",
  age: "32",
  annual_income: "70",
  spending_score: "60",
};

function genderLabel(value) {
  return value === "Female" ? "Mujer" : "Hombre";
}

export default function App() {
  const [section, setSection] = useState("exploracion");
  const [analysis, setAnalysis] = useState(null);
  const [customers, setCustomers] = useState([]);
  const [error, setError] = useState("");
  const [activeId, setActiveId] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [prediction, setPrediction] = useState(null);
  const [predictError, setPredictError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    Promise.all([getAnalysis(), getCustomers()])
      .then(([analysisData, customerData]) => {
        setAnalysis(analysisData);
        setCustomers(customerData.customers);
      })
      .catch((err) => setError(err.message));
  }, []);

  async function onPredict(event) {
    event.preventDefault();
    setBusy(true);
    setPredictError("");
    try {
      const result = await predictClient({
        gender: form.gender,
        age: Number(form.age),
        annual_income: Number(form.annual_income),
        spending_score: Number(form.spending_score),
      });
      setPrediction(result);
      setActiveId(result.cluster);
      setSection("clasificar");
    } catch (err) {
      setPrediction(null);
      setPredictError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (error) {
    return (
      <main className="gate">
        <p className="eyebrow">Backend</p>
        <h1>No hay conexión con el modelo</h1>
        <p>{error}. Levanta la API y vuelve a cargar esta página.</p>
      </main>
    );
  }

  if (!analysis) {
    return (
      <main className="gate">
        <p className="eyebrow">Mall Customers</p>
        <h1>Cargando la segmentación…</h1>
      </main>
    );
  }

  const chosen = analysis.conjuntos.ingreso_gasto;
  const alternate = analysis.conjuntos.edad_ingreso_gasto;
  const descriptivos = analysis.dataset.descriptivos;

  return (
    <div className="shell">
      <aside className="rail">
        <p className="brand">Mall Customers</p>
        <h1>Segmentación con K-Means</h1>
        <p className="rail-note">
          {analysis.dataset.n_filas} clientes · k = {analysis.selected_k} · silueta {analysis.silhouette.toFixed(3)}
        </p>
        <nav>
          {NAV.map(([id, number, label]) => (
            <button key={id} className={section === id ? "nav active" : "nav"} onClick={() => setSection(id)}>
              <span>{number}</span>
              {label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="stage">
        {section === "exploracion" && (
          <section>
            <p className="eyebrow">Parte 1</p>
            <h2>Exploración de datos</h2>
            <p className="lead">
              El archivo tiene {analysis.dataset.n_filas} filas y {analysis.dataset.n_columnas} columnas, sin valores nulos.
              CustomerID identifica al cliente y no se usa para agrupar. El género se describe, pero la distancia del modelo se calcula con ingreso y gasto.
            </p>
            <div className="stats">
              <article>
                <span>Mujeres</span>
                <strong>{analysis.dataset.genero.Female}</strong>
              </article>
              <article>
                <span>Hombres</span>
                <strong>{analysis.dataset.genero.Male}</strong>
              </article>
              <article>
                <span>Edad media</span>
                <strong>{descriptivos.Age[1]}</strong>
              </article>
              <article>
                <span>Ingreso medio (k$)</span>
                <strong>{descriptivos[analysis.features[0]][1]}</strong>
              </article>
              <article>
                <span>Gasto medio</span>
                <strong>{descriptivos[analysis.features[1]][1]}</strong>
              </article>
            </div>

            <h3>Descriptivos</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Estadístico</th>
                    <th>Edad</th>
                    <th>Ingreso anual (k$)</th>
                    <th>Puntaje de gasto</th>
                  </tr>
                </thead>
                <tbody>
                  {descriptivos.estadistico.map((name, index) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{descriptivos.Age[index]}</td>
                      <td>{descriptivos["Annual Income (k$)"][index]}</td>
                      <td>{descriptivos["Spending Score (1-100)"][index]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h3>Correlación</h3>
            <Correlation matrix={analysis.dataset.correlacion} />

            <h3>Valores atípicos (rango intercuartílico)</h3>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Variable</th>
                    <th>Límite inferior</th>
                    <th>Límite superior</th>
                    <th>Cantidad</th>
                    <th>Valores</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(analysis.dataset.atipicos).map(([name, info]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{info.limite_inferior}</td>
                      <td>{info.limite_superior}</td>
                      <td>{info.n}</td>
                      <td>{info.valores.length ? info.valores.join(", ") : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Answer title={QUESTIONS[0][1]} text={analysis.respuestas.variables} />
            <Answer title={QUESTIONS[1][1]} text={analysis.respuestas.atipicos} />
            <Answer title={QUESTIONS[2][1]} text={analysis.respuestas.relaciones} />
          </section>
        )}

        {section === "escala" && (
          <section>
            <p className="eyebrow">Parte 2</p>
            <h2>StandardScaler</h2>
            <p className="lead">
              Antes de medir distancias, ingreso y gasto se centran en cero y se llevan a desviación estándar uno. Esas medias y escalas quedan guardadas dentro del pipeline y se vuelven a aplicar al clasificar un cliente nuevo.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Variable</th>
                    <th>Media de entrenamiento</th>
                    <th>Desviación estándar</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.scaler.features.map((feature, index) => (
                    <tr key={feature}>
                      <td>{feature}</td>
                      <td>{analysis.scaler.mean[index]}</td>
                      <td>{analysis.scaler.scale[index]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <h3>Por qué no entró la edad al modelo final</h3>
            <div className="compare">
              <article>
                <span>Ingreso y gasto</span>
                <strong>silueta {chosen.best_silhouette.toFixed(3)}</strong>
                <p>Mejor k = {chosen.best_silhouette_k}. Codo en k = {chosen.elbow_k}.</p>
              </article>
              <article>
                <span>Edad, ingreso y gasto</span>
                <strong>silueta {alternate.best_silhouette.toFixed(3)}</strong>
                <p>Mejor k = {alternate.best_silhouette_k}. Codo en k = {alternate.elbow_k}.</p>
              </article>
            </div>
            <Answer title={QUESTIONS[3][1]} text={analysis.respuestas.escalar} />
          </section>
        )}

        {section === "codo" && (
          <section>
            <p className="eyebrow">Parte 3</p>
            <h2>Método del codo</h2>
            <p className="lead">
              La inercia baja rápido hasta k = {chosen.elbow_k} y después cada cluster adicional recorta poco error. La línea vertical marca ese codo.
            </p>
            <ElbowChart curve={chosen.curve} selectedK={analysis.selected_k} />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>k</th>
                    {chosen.curve.map((item) => (
                      <th key={item.k}>{item.k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Inercia</td>
                    {chosen.curve.map((item) => (
                      <td key={item.k}>{item.inertia.toFixed(1)}</td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
            <Answer title={QUESTIONS[4][1]} text={analysis.respuestas.inercia_cero} />
            <Answer title={QUESTIONS[5][1]} text={analysis.respuestas.punto_optimo} />
          </section>
        )}

        {section === "silueta" && (
          <section>
            <p className="eyebrow">Parte 4</p>
            <h2>Coeficiente de silueta</h2>
            <p className="lead">
              El máximo está en k = {chosen.best_silhouette_k}, con {chosen.best_silhouette.toFixed(3)}. El modelo guardado usa ese k y alcanza una silueta de {analysis.silhouette.toFixed(3)}.
            </p>
            <SilhouetteChart curve={chosen.curve} selectedK={analysis.selected_k} />
            <Answer title={QUESTIONS[6][1]} text={analysis.respuestas.silueta_mide} />
            <Answer title={QUESTIONS[7][1]} text={analysis.respuestas.silueta_uno} />
            <Answer title={QUESTIONS[8][1]} text={analysis.respuestas.silueta_cero} />
            <Answer title={QUESTIONS[9][1]} text={analysis.respuestas.coincide} />
          </section>
        )}

        {section === "segmentos" && (
          <section>
            <p className="eyebrow">Parte 5</p>
            <h2>Modelo K-Means</h2>
            <p className="lead">
              Cada punto es un cliente. El cuadrado negro es el centroide, de vuelta en las unidades originales. Pulsa un perfil para aislar su grupo.
            </p>
            <div className="legend">
              <button className={activeId === null ? "chip active" : "chip"} onClick={() => setActiveId(null)}>
                Todos
              </button>
              {analysis.clusters.map((cluster) => (
                <button
                  key={cluster.id}
                  className={activeId === cluster.id ? "chip active" : "chip"}
                  onClick={() => setActiveId(cluster.id)}
                >
                  <i style={{ background: colorOf(cluster) }} />
                  {cluster.nombre}
                </button>
              ))}
            </div>
            <ClusterScatter customers={customers} clusters={analysis.clusters} prediction={prediction} activeId={activeId} />
            <div className="cards">
              {analysis.clusters.map((cluster) => (
                <article key={cluster.id} className="profile-card">
                  <header>
                    <i style={{ background: colorOf(cluster) }} />
                    <div>
                      <h3>{cluster.nombre}</h3>
                      <p>
                        Cluster {cluster.id} · {cluster.n} clientes ({cluster.pct}%)
                      </p>
                    </div>
                  </header>
                  <dl>
                    <div>
                      <dt>Centroide ingreso</dt>
                      <dd>{cluster.centroid.annual_income} k$</dd>
                    </div>
                    <div>
                      <dt>Centroide gasto</dt>
                      <dd>{cluster.centroid.spending_score}</dd>
                    </div>
                    <div>
                      <dt>Edad media</dt>
                      <dd>{cluster.stats.age_mean}</dd>
                    </div>
                    <div>
                      <dt>Mujeres / hombres</dt>
                      <dd>
                        {cluster.stats.female_pct}% / {cluster.stats.male_pct}%
                      </dd>
                    </div>
                  </dl>
                  <p>{cluster.diferencia}</p>
                </article>
              ))}
            </div>
          </section>
        )}

        {section === "negocio" && (
          <section>
            <p className="eyebrow">Parte 6</p>
            <h2>Interpretación de negocio</h2>
            <p className="lead">
              La mayor oportunidad de crecimiento es <strong>{analysis.crecimiento.nombre}</strong>. {analysis.crecimiento.razon}
            </p>
            <div className="cards">
              {analysis.clusters.map((cluster) => (
                <article key={cluster.id} className="profile-card">
                  <header>
                    <i style={{ background: colorOf(cluster) }} />
                    <div>
                      <h3>{cluster.nombre}</h3>
                      {cluster.oportunidad_crecimiento && <em>Mayor oportunidad</em>}
                    </div>
                  </header>
                  <p>{cluster.resumen}</p>
                  <ul>
                    {cluster.estrategias.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          </section>
        )}

        {section === "clasificar" && (
          <section>
            <p className="eyebrow">Parte 7</p>
            <h2>Clasificar un cliente nuevo</h2>
            <p className="lead">
              El pipeline aplica el mismo StandardScaler y asigna el centroide más cercano. Edad y género quedan en la ficha; no modifican la distancia.
            </p>
            <div className="classify">
              <form onSubmit={onPredict}>
                <label>
                  Género
                  <select value={form.gender} onChange={(event) => setForm({ ...form, gender: event.target.value })}>
                    <option value="Female">Mujer</option>
                    <option value="Male">Hombre</option>
                  </select>
                </label>
                <label>
                  Edad
                  <input
                    type="number"
                    min="15"
                    max="100"
                    value={form.age}
                    onChange={(event) => setForm({ ...form, age: event.target.value })}
                    required
                  />
                </label>
                <label>
                  Ingreso anual (k$)
                  <input
                    type="number"
                    min="0"
                    max="300"
                    step="0.1"
                    value={form.annual_income}
                    onChange={(event) => setForm({ ...form, annual_income: event.target.value })}
                    required
                  />
                </label>
                <label>
                  Puntaje de gasto (1-100)
                  <input
                    type="number"
                    min="1"
                    max="100"
                    step="1"
                    value={form.spending_score}
                    onChange={(event) => setForm({ ...form, spending_score: event.target.value })}
                    required
                  />
                </label>
                <button className="primary" type="submit" disabled={busy}>
                  {busy ? "Clasificando…" : "Asignar segmento"}
                </button>
                {predictError && <p className="form-error">{predictError}</p>}
              </form>
              <div className="examples">
                <p>Probar un perfil conocido</p>
                {EXAMPLES.map((example) => (
                  <button
                    key={example.label}
                    type="button"
                    onClick={() =>
                      setForm({
                        gender: example.gender,
                        age: String(example.age),
                        annual_income: String(example.annual_income),
                        spending_score: String(example.spending_score),
                      })
                    }
                  >
                    {example.label}
                  </button>
                ))}
              </div>
            </div>

            {prediction && (
              <article className="result">
                <p className="eyebrow">Resultado</p>
                <h3 style={{ color: colorOf(prediction.perfil) }}>{prediction.perfil.nombre}</h3>
                <p>
                  {genderLabel(prediction.entrada.gender)}, {prediction.entrada.age} años, ingreso {prediction.entrada.annual_income} k$, gasto {prediction.entrada.spending_score}. Distancia al centroide en el espacio escalado: {prediction.distancia_al_centroide}.
                </p>
                <p>{prediction.perfil.resumen}</p>
                <p className="fine">{prediction.nota}</p>
              </article>
            )}

            <ClusterScatter customers={customers} clusters={analysis.clusters} prediction={prediction} activeId={activeId} />
          </section>
        )}
      </main>
    </div>
  );
}

function Correlation({ matrix }) {
  const labels = ["Edad", "Ingreso", "Gasto"];
  const cells = [<span key="corner" />, ...labels.map((label) => <strong key={`h-${label}`}>{label}</strong>)];
  matrix.matriz.forEach((row, rowIndex) => {
    cells.push(<strong key={`l-${rowIndex}`}>{labels[rowIndex]}</strong>);
    row.forEach((value, colIndex) => {
      cells.push(
        <span key={`${rowIndex}-${colIndex}`} style={{ background: heatColor(value) }}>
          {Number(value).toFixed(2)}
        </span>
      );
    });
  });
  return (
    <div className="heat" style={{ gridTemplateColumns: `120px repeat(${labels.length}, 1fr)` }}>
      {cells}
    </div>
  );
}

function heatColor(value) {
  const amount = Math.min(1, Math.abs(value));
  if (value >= 0) {
    return `rgba(14, 107, 99, ${0.12 + amount * 0.75})`;
  }
  return `rgba(181, 82, 42, ${0.12 + amount * 0.75})`;
}

function Answer({ title, text }) {
  return (
    <article className="answer">
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}
