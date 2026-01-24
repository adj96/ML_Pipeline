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

export default function () {
  // Health check (must be 200)
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

  // Optional: basic predict smoke (only runs if you provide PAYLOAD_JSON)
  // In Jenkins: set env PAYLOAD_JSON={"feature1":1.0,"feature2":2.0,...}
  if (__ENV.PAYLOAD_JSON) {
    let payload;
    try {
      payload = JSON.parse(__ENV.PAYLOAD_JSON);
    } catch (e) {
      payload = null;
    }

    if (payload) {
      const p = http.post(url("/predict"), JSON.stringify(payload), {
        headers: { "Content-Type": "application/json" },
      });

      check(p, {
        "predict status is 200": (r) => r.status === 200,
        "predict returns prediction": (r) => {
          try {
            const j = r.json();
            return j && typeof j.prediction === "number" && isFinite(j.prediction);
          } catch (e) {
            return false;
          }
        },
      });
    }
  }

  sleep(1);
}
