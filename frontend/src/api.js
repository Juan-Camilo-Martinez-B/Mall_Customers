export async function getAnalysis() {
  const response = await fetch("/api/analysis");
  if (!response.ok) {
    throw new Error("No se pudo cargar el análisis");
  }
  return response.json();
}

export async function getCustomers() {
  const response = await fetch("/api/customers");
  if (!response.ok) {
    throw new Error("No se pudo cargar los clientes");
  }
  return response.json();
}

export async function predictClient(payload) {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => item.msg).join(" ")
      : detail || "No se pudo clasificar";
    throw new Error(message);
  }
  return data;
}
