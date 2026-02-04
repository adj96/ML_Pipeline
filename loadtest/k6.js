import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  vus: 5,
  duration: "30s",
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<1000"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://arvmldevopspipeline-svc:8000";

export default function () {
  const health = http.get(`${BASE_URL}/health`);
  check(health, {
    "health status 200": (r) => r.status === 200,
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

  const params = { headers: { "Content-Type": "application/json" } };
  const pred = http.post(`${BASE_URL}/predict`, payload, params);

  check(pred, {
    "predict status 200": (r) => r.status === 200,
  });

  sleep(1);
}
