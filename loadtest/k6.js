import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 5 },
    { duration: "20s", target: 15 },
    { duration: "20s", target: 30 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate==0"],
    http_req_duration: ["p(95)<500"],
  },
};

function url(path) {
  const base = (__ENV.BASE_URL || "http://arvmldevopspipeline-svc:8000").replace(/\/+$/, "");
  const p = (path || "").startsWith("/") ? path : `/${path || ""}`;
  return `${base}${p}`;
}

// Default minimal payload. Override in Jenkins by setting PAYLOAD_JSON
const DEFAULT_PAYLOAD = {
  event_ts: "2026-01-24 10:00:00",
  baseline_queue_min: 12.0,
  shortage_flag: 0,
  replenishment_eta_min: 0.0,
  machine_state: "RUN",
  queue_time_min: 5.0,
  down_minutes_last_60: 0.0,
};

function getPayload() {
  if (!__ENV.PAYLOAD_JSON) return DEFAULT_PAYLOAD;
  try {
    const p = JSON.parse(__ENV.PAYLOAD_JSON);
    return p && typeof p === "object" ? p : DEFAULT_PAYLOAD;
  } catch (e) {
    return DEFAULT_PAYLOAD;
  }
}

export default function () {
  // 80–90% predict traffic, 10–20% health traffic
  // Keep it deterministic and simple: 1 in 5 iterations hits /health (~20%), others hit /predict (~80%)
  const doHealth = (__ITER % 5) === 0;

  if (doHealth) {
    const h = http.get(url("/health"));
    check(h, {
      "health status is 200": (r) => r.status === 200,
      "health has ok payload": (r) => {
        try {
          const j = r.json();
          return j && (j.status === "ok" || j.status === "OK");
        } catch (e) {
          return false;
        }
      },
    });
  } else {
    const payload = getPayload();
    const p = http.post(url("/predict"), JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    });

    check(p, {
      "predict status is 200": (r) => r.status === 200,
      "predict returns prediction": (r) => {
        try {
          const j = r.json();
          // Accept both common formats:
          // 1) {"prediction": 12.3}
          // 2) {"predictions":[12.3]} or {"predictions":[...]}
          if (j && typeof j.prediction === "number" && isFinite(j.prediction)) return true;
          if (j && Array.isArray(j.predictions) && j.predictions.length > 0) {
            const v = j.predictions[0];
            return typeof v === "number" && isFinite(v);
          }
          return false;
        } catch (e) {
          return false;
        }
      },
    });
  }

  sleep(1);
}
