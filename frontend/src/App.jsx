import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8001";

const FIELDS = [
  { name: "age", label: "Age (years)", type: "number", min: 18, max: 100 },
  {
    name: "gender", label: "Gender", type: "select",
    options: [{ value: 1, label: "Female" }, { value: 2, label: "Male" }],
  },
  { name: "height", label: "Height (cm)", type: "number", min: 120, max: 220 },
  { name: "weight", label: "Weight (kg)", type: "number", min: 30, max: 200 },
  { name: "ap_hi", label: "Systolic BP (mmHg)", type: "number", min: 80, max: 250 },
  { name: "ap_lo", label: "Diastolic BP (mmHg)", type: "number", min: 40, max: 200 },
  {
    name: "cholesterol", label: "Cholesterol", type: "select",
    options: [{ value: 1, label: "Normal" }, { value: 2, label: "Above normal" }, { value: 3, label: "Well above normal" }],
  },
  {
    name: "gluc", label: "Glucose", type: "select",
    options: [{ value: 1, label: "Normal" }, { value: 2, label: "Above normal" }, { value: 3, label: "Well above normal" }],
  },
  {
    name: "smoke", label: "Smoker", type: "select",
    options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }],
  },
  {
    name: "alco", label: "Alcohol intake", type: "select",
    options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }],
  },
  {
    name: "active", label: "Physically active", type: "select",
    options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }],
  },
];

const DEFAULT_FORM = {
  age: "", gender: "", height: "", weight: "", ap_hi: "", ap_lo: "",
  cholesterol: "", gluc: "", smoke: "", alco: "", active: "",
};

function ConsentScreen({ onAgree }) {
  const [checked, setChecked] = useState(false);
  const [nickname, setNickname] = useState("");

  return (
    <div className="card">
      <h1>Clinical Risk Prediction</h1>
      <p className="subtitle">Cardiovascular risk demo — SHAP-explained ML predictions</p>

      <div className="disclaimer-banner">
        <strong>Not medical advice.</strong>
        This is a portfolio/educational project, not a medical device. It does not
        diagnose disease and should never replace professional medical advice.
        If you have a real health concern, please see a doctor.
      </div>

      <div className="field">
        <label>Nickname (optional — never required, just for your own reference)</label>
        <input
          type="text" maxLength={50} value={nickname}
          onChange={(e) => setNickname(e.target.value)}
          placeholder="e.g. 'test run' — leave blank to stay anonymous"
        />
      </div>

      <div className="consent-box">
        <input
          type="checkbox" id="consent" checked={checked}
          onChange={(e) => setChecked(e.target.checked)}
        />
        <label htmlFor="consent">
          I understand this tool does not provide medical advice, and I consent
          to my anonymized inputs (no name or contact info required) being stored
          to power this demo's prediction history. I can decline and still get a
          one-off prediction without storage.
        </label>
      </div>

      <button className="primary" onClick={() => onAgree({ consent: true, nickname })}>
        Continue with consent
      </button>
      <button className="secondary" onClick={() => onAgree({ consent: false, nickname: "" })}>
        Continue without storing my data
      </button>
    </div>
  );
}

function PredictionForm({ consent, nickname, onResult }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (name, value) => {
    setForm((f) => ({ ...f, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const emptyField = FIELDS.find((f) => form[f.name] === "");
    if (emptyField) {
      setError(`Please fill in "${emptyField.label}".`);
      return;
    }

    const payload = {
      consent_given: consent,
      nickname: nickname || null,
      ...Object.fromEntries(
        FIELDS.map((f) => [f.name, Number(form[f.name])])
      ),
    };

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((d) => d.msg).join("; ")
          : data.detail || "Something went wrong.";
        throw new Error(detail);
      }
      onResult(data);
    } catch (err) {
      setError(err.message || "Failed to reach the prediction service.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h1>Enter your health metrics</h1>
      <p className="subtitle">
        {consent
          ? "Your inputs will be stored anonymously for this demo's history."
          : "Your inputs will NOT be stored — one-off prediction only."}
      </p>

      {error && <div className="error-box">{error}</div>}

      <form onSubmit={handleSubmit}>
        <div className="field-grid">
          {FIELDS.map((f) => (
            <div className="field" key={f.name}>
              <label htmlFor={f.name}>{f.label}</label>
              {f.type === "select" ? (
                <select
                  id={f.name} value={form[f.name]}
                  onChange={(e) => handleChange(f.name, e.target.value)}
                >
                  <option value="" disabled>Select…</option>
                  {f.options.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              ) : (
                <input
                  id={f.name} type="number" min={f.min} max={f.max}
                  value={form[f.name]}
                  onChange={(e) => handleChange(f.name, e.target.value)}
                />
              )}
            </div>
          ))}
        </div>
        <button className="primary" type="submit" disabled={loading}>
          {loading ? "Calculating…" : "Get my risk prediction"}
        </button>
      </form>
    </div>
  );
}

function ResultScreen({ result, onReset }) {
  const isHigh = result.risk_class === "high";
  return (
    <div className="card">
      <div className="risk-result">
        <div className={`risk-badge ${result.risk_class}`}>
          {isHigh ? "Higher risk" : "Lower risk"}
        </div>
        <div className="risk-probability">
          {(result.risk_probability * 100).toFixed(1)}%
        </div>
        <p className="subtitle">estimated risk probability (model: {result.model_version})</p>
      </div>

      <div className="contributions">
        <h3>What drove this prediction</h3>
        {result.top_contributions.map((c) => (
          <div className="contribution-row" key={c.feature}>
            <span>{c.feature}</span>
            <span className={`dir ${c.direction}`}>{c.direction}</span>
          </div>
        ))}
      </div>

      <div className="disclaimer-banner" style={{ marginTop: 24 }}>
        {result.disclaimer}
      </div>

      <button className="secondary" onClick={onReset}>Try another prediction</button>
    </div>
  );
}

export default function App() {
  const [stage, setStage] = useState("consent"); // consent | form | result
  const [consent, setConsent] = useState(false);
  const [nickname, setNickname] = useState("");
  const [result, setResult] = useState(null);

  return (
    <div className="app">
      {stage === "consent" && (
        <ConsentScreen
          onAgree={({ consent, nickname }) => {
            setConsent(consent);
            setNickname(nickname);
            setStage("form");
          }}
        />
      )}
      {stage === "form" && (
        <PredictionForm
          consent={consent}
          nickname={nickname}
          onResult={(data) => {
            setResult(data);
            setStage("result");
          }}
        />
      )}
      {stage === "result" && result && (
        <ResultScreen result={result} onReset={() => setStage("form")} />
      )}
    </div>
  );
}
