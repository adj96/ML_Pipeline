import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 5 },
    { duration: "20s", target: 30 },
    { duration: "20s", target: 30 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate==0"],
    http_req_duration: ["p(95)<2000"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://arvmldevopspipeline-svc:8000";

export default function () {
  const h = http.get(`${BASE_URL}/health`);
  check(h, {
    "health status is 200": (r) => r.status === 200,
    "health has ok payload": (r) => r.json("status") === "ok",
    "health model_loaded true": (r) => r.json("model_loaded") === true,
  });

  const payload = JSON.stringify({
    event_ts: "2026-02-03T00:00:00Z",
    baseline_queue_min: 1.0,
    shortage_flag: 0,
    replenishment_eta_min: 5.0,
    machine_state: "RUN",
    queue_time_min: 2.0,
    down_minutes_last_60: 0.0,
  });

  const p = http.post(`${BASE_URL}/predict`, payload, {
    headers: { "Content-Type": "application/json" },
  });

  check(p, {
    "predict status is 200": (r) => r.status === 200,
    "predict returns prediction": (r) => typeof r.json("prediction") === "number",
  });

  sleep(1);
}
