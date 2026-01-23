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

const BASE_URL = (__ENV.BASE_URL || "http://mldevops:8000").replace(/\/+$/, "");

export default function () {
  // 10 features (replace numbers with realistic values/ranges)
  const payload = JSON.stringify({
    machine_age_days: 10,
    temperature_c: 25,
    pressure_kpa: 101,
    vibration_mm_s: 1.2,
    humidity_pct: 60,
    operator_experience_yrs: 3,
    shift: 2,
    material_grade: 1,
    line_speed_m_min: 120,
    inspection_interval_hrs: 8
  });

  const params = {
    headers: { "Content-Type": "application/json" },
  };

  const res = http.post(`${BASE_URL}/predict`, payload, params);

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  sleep(1);
}
