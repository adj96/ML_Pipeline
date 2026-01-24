import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "10s", target: 5 },
    { duration: "60s", target: 30 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<800"],
  },
};

const BASE_URL = (__ENV.BASE_URL || "http://arvmldevopspipeline-svc.arvmldevopspipeline.svc.cluster.local:8000")
  .replace(/\/+$/, "");

function rndInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export default function () {
  const payload = JSON.stringify({
    event_ts: Date.now(),                 // integer-like
    shortage_flag: rndInt(0, 1),          // 0/1
    replenishment_eta_min: rndInt(0, 240),
    machine_state: rndInt(0, 3),
    down_minutes_last_60: rndInt(0, 60),
    queue_time_min: rndInt(0, 120),
    baseline_queue_min: rndInt(0, 120),
  });

  const res = http.post(`${BASE_URL}/predict`, payload, {
    headers: { "Content-Type": "application/json" },
    timeout: "5s",
  });

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  sleep(1);
}
