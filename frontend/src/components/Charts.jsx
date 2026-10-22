const PALETTE = {
  alto_valor: "#0e6b63",
  potencial: "#b5522a",
  promedio: "#2f5d8c",
  bajo_valor: "#6e6258",
  entusiastas: "#a33b5a",
};

export function colorOf(cluster) {
  return PALETTE[cluster.profile_key] || "#1a2332";
}

function ticks(min, max, count) {
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, i) => min + step * i);
}

export function ElbowChart({ curve, selectedK }) {
  const width = 680;
  const height = 320;
  const pad = { l: 56, r: 18, t: 22, b: 42 };
  const minK = Math.min(...curve.map((item) => item.k));
  const maxK = Math.max(...curve.map((item) => item.k));
  const minY = 0;
  const maxY = Math.max(...curve.map((item) => item.inertia)) * 1.08;
  const x = (k) => pad.l + ((k - minK) / (maxK - minK || 1)) * (width - pad.l - pad.r);
  const y = (value) => pad.t + (1 - (value - minY) / (maxY - minY)) * (height - pad.t - pad.b);
  const path = curve
    .map((item, index) => `${index === 0 ? "M" : "L"} ${x(item.k)} ${y(item.inertia)}`)
    .join(" ");
  const yTicks = ticks(minY, maxY, 5);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img" aria-label="Método del codo">
      {yTicks.map((tick) => (
        <g key={tick}>
          <line x1={pad.l} x2={width - pad.r} y1={y(tick)} y2={y(tick)} className="grid" />
          <text x={pad.l - 8} y={y(tick) + 4} className="tick" textAnchor="end">
            {Math.round(tick)}
          </text>
        </g>
      ))}
      <line x1={x(selectedK)} x2={x(selectedK)} y1={pad.t} y2={height - pad.b} className="marker-line" />
      <path d={path} className="series" />
      {curve.map((item) => (
        <g key={item.k}>
          <circle cx={x(item.k)} cy={y(item.inertia)} r={item.k === selectedK ? 6 : 4} className={item.k === selectedK ? "dot selected" : "dot"} />
          <text x={x(item.k)} y={height - 16} className="tick" textAnchor="middle">
            {item.k}
          </text>
        </g>
      ))}
      <text x={width / 2} y={height - 2} className="axis-title" textAnchor="middle">
        k
      </text>
      <text x={16} y={height / 2} className="axis-title" textAnchor="middle" transform={`rotate(-90 16 ${height / 2})`}>
        Inercia
      </text>
    </svg>
  );
}

export function SilhouetteChart({ curve, selectedK }) {
  const items = curve.filter((item) => item.silhouette !== null);
  const width = 680;
  const height = 320;
  const pad = { l: 48, r: 18, t: 18, b: 42 };
  const innerW = width - pad.l - pad.r;
  const slot = innerW / items.length;
  const maxY = Math.max(0.7, ...items.map((item) => item.silhouette));
  const y = (value) => pad.t + (1 - value / maxY) * (height - pad.t - pad.b);
  const base = height - pad.b;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img" aria-label="Coeficiente de silueta">
      {[0, 0.25, 0.5, 0.75].filter((tick) => tick <= maxY).map((tick) => (
        <g key={tick}>
          <line x1={pad.l} x2={width - pad.r} y1={y(tick)} y2={y(tick)} className="grid" />
          <text x={pad.l - 8} y={y(tick) + 4} className="tick" textAnchor="end">
            {tick.toFixed(2)}
          </text>
        </g>
      ))}
      {items.map((item, index) => {
        const barX = pad.l + index * slot + slot * 0.22;
        const barW = slot * 0.56;
        const barY = y(item.silhouette);
        return (
          <g key={item.k}>
            <rect x={barX} y={barY} width={barW} height={base - barY} className={item.k === selectedK ? "bar selected" : "bar"} />
            <text x={barX + barW / 2} y={barY - 6} className="tick" textAnchor="middle">
              {item.silhouette.toFixed(3)}
            </text>
            <text x={barX + barW / 2} y={height - 16} className="tick" textAnchor="middle">
              {item.k}
            </text>
          </g>
        );
      })}
      <text x={width / 2} y={height - 2} className="axis-title" textAnchor="middle">
        k
      </text>
    </svg>
  );
}

export function ClusterScatter({ customers, clusters, prediction, activeId }) {
  const width = 700;
  const height = 460;
  const pad = { l: 52, r: 20, t: 16, b: 44 };
  const incomes = customers.map((item) => item.annual_income);
  const spends = customers.map((item) => item.spending_score);
  if (prediction) {
    incomes.push(prediction.entrada.annual_income);
    spends.push(prediction.entrada.spending_score);
  }
  const minX = 0;
  const maxX = Math.max(150, ...incomes);
  const minY = 0;
  const maxY = 100;
  const x = (value) => pad.l + ((value - minX) / (maxX - minX)) * (width - pad.l - pad.r);
  const y = (value) => pad.t + (1 - (value - minY) / (maxY - minY)) * (height - pad.t - pad.b);
  const byId = Object.fromEntries(clusters.map((cluster) => [cluster.id, cluster]));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart" role="img" aria-label="Clusters de clientes">
      {[0, 25, 50, 75, 100].map((tick) => (
        <g key={`y-${tick}`}>
          <line x1={pad.l} x2={width - pad.r} y1={y(tick)} y2={y(tick)} className="grid" />
          <text x={pad.l - 8} y={y(tick) + 4} className="tick" textAnchor="end">
            {tick}
          </text>
        </g>
      ))}
      {ticks(minX, maxX, 6).map((tick) => (
        <text key={`x-${tick}`} x={x(tick)} y={height - 18} className="tick" textAnchor="middle">
          {Math.round(tick)}
        </text>
      ))}
      {customers.map((customer) => {
        const cluster = byId[customer.cluster];
        const dim = activeId !== null && customer.cluster !== activeId;
        return (
          <circle
            key={customer.customer_id}
            cx={x(customer.annual_income)}
            cy={y(customer.spending_score)}
            r={5}
            fill={colorOf(cluster)}
            opacity={dim ? 0.15 : 0.88}
          >
            <title>
              {`Cliente ${customer.customer_id}: ingreso ${customer.annual_income}, gasto ${customer.spending_score}`}
            </title>
          </circle>
        );
      })}
      {clusters.map((cluster) => (
        <g key={`c-${cluster.id}`}>
          <rect
            x={x(cluster.centroid.annual_income) - 6}
            y={y(cluster.centroid.spending_score) - 6}
            width="12"
            height="12"
            transform={`rotate(45 ${x(cluster.centroid.annual_income)} ${y(cluster.centroid.spending_score)})`}
            fill="#1a2332"
          />
        </g>
      ))}
      {prediction && (
        <g>
          <circle
            cx={x(prediction.entrada.annual_income)}
            cy={y(prediction.entrada.spending_score)}
            r="11"
            className="new-point-ring"
          />
          <circle
            cx={x(prediction.entrada.annual_income)}
            cy={y(prediction.entrada.spending_score)}
            r="5.5"
            fill={colorOf(prediction.perfil)}
            stroke="#fff"
            strokeWidth="2"
          />
        </g>
      )}
      <text x={width / 2} y={height - 2} className="axis-title" textAnchor="middle">
        Ingreso anual (k$)
      </text>
      <text x={14} y={height / 2} className="axis-title" textAnchor="middle" transform={`rotate(-90 14 ${height / 2})`}>
        Puntaje de gasto
      </text>
    </svg>
  );
}
