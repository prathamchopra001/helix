"""
Locust load test for the Helix inference API.

Targets the Nginx gateway (localhost:80) which proxies to the inference service.

Usage:
  # Install: pip install locust
  # Interactive UI (opens browser at http://localhost:8089):
  locust -f tests/load/locustfile.py --host http://localhost

  # Headless — 50 concurrent users, ramp 5/s, run 60s:
  locust -f tests/load/locustfile.py --host http://localhost \
    --headless -u 50 -r 5 -t 60s \
    --html tests/load/report.html

  # Headless with CSV stats:
  locust -f tests/load/locustfile.py --host http://localhost \
    --headless -u 50 -r 5 -t 120s \
    --csv tests/load/results

SLA targets (CPU/ONNX — no GPU):
  p50 latency  < 800ms
  p95 latency  < 2000ms  (batch endpoint included)
  p99 latency  < 4000ms  (dominated by batch endpoint under load)
  error rate   < 1%

  GPU/TRT targets (when running with --profile gpu):
  p50 latency  < 50ms
  p95 latency  < 100ms
  p99 latency  < 200ms
"""

import random

from locust import HttpUser, between, events, task

# API keys defined in .env — API_KEYS=dev-key-1,dev-key-2
API_KEYS = ["dev-key-1", "dev-key-2"]

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "TSLA",
    "SPY",
    "QQQ",
    "BTC-USD",
    "ETH-USD",
]


class InferenceUser(HttpUser):
    """
    Simulates a typical API consumer:
    - 80% single-ticker predict requests
    - 15% batch predict requests
    - 5% health/ready checks
    """

    wait_time = between(0.1, 0.5)  # Think time between requests (100–500ms)

    def on_start(self) -> None:
        """Pick a random API key for this user session."""
        self.api_key = random.choice(API_KEYS)
        self.headers = {"X-API-Key": self.api_key}

    @task(8)
    def predict_single(self) -> None:
        """POST /api/predict — single ticker"""
        ticker = random.choice(TICKERS)
        with self.client.post(
            "/api/predict",
            json={"ticker": ticker},
            headers=self.headers,
            name="/api/predict [single]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "score" not in data:
                    resp.failure(f"Missing 'score' in response: {resp.text[:100]}")
                elif not (0.0 <= data["score"] <= 1.0):
                    resp.failure(f"Score out of range: {data['score']}")
                else:
                    resp.success()
            elif resp.status_code == 503:
                resp.failure("Model not loaded (503)")
            elif resp.status_code == 422:
                resp.failure(f"Not enough data for {ticker} (422)")
            else:
                resp.failure(f"Unexpected status {resp.status_code}: {resp.text[:100]}")

    @task(2)
    def predict_batch(self) -> None:
        """POST /api/predict/batch — 3 random tickers"""
        tickers = random.sample(TICKERS, k=3)
        with self.client.post(
            "/api/predict/batch",
            json={"tickers": tickers},
            headers=self.headers,
            name="/api/predict/batch",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "results" not in data:
                    resp.failure(f"Missing 'results': {resp.text[:100]}")
                else:
                    resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}: {resp.text[:100]}")

    @task(1)
    def health_check(self) -> None:
        """GET /api/health — liveness probe"""
        with self.client.get(
            "/api/health",
            headers=self.headers,
            name="/api/health",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(1)
    def ready_check(self) -> None:
        """GET /api/ready — readiness probe"""
        with self.client.get(
            "/api/ready",
            headers=self.headers,
            name="/api/ready",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 503):
                resp.success()  # 503 is expected if model is loading
            else:
                resp.failure(f"Ready check unexpected status: {resp.status_code}")


class UnauthenticatedUser(HttpUser):
    """
    Simulates unauthenticated traffic — should always get 401.
    Weight=1 means 1 in 10 users is unauthenticated.
    """

    weight = 1
    wait_time = between(1, 3)

    @task
    def predict_no_key(self) -> None:
        with self.client.post(
            "/api/predict",
            json={"ticker": "AAPL"},
            name="/api/predict [unauth]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 401:
                resp.success()  # expected
            else:
                resp.failure(f"Expected 401, got {resp.status_code}")


# ── SLA verification hook ──────────────────────────────────────────────────────


@events.quitting.add_listener
def check_sla(environment, **kwargs):
    """Print SLA pass/fail summary when the test ends."""
    stats = environment.runner.stats.total
    if stats.num_requests == 0:
        return

    p50 = stats.get_response_time_percentile(0.50)
    p95 = stats.get_response_time_percentile(0.95)
    p99 = stats.get_response_time_percentile(0.99)
    error_rate = (stats.num_failures / stats.num_requests) * 100

    print("\n" + "=" * 60)
    print("SLA VERIFICATION")
    print("=" * 60)
    print(f"Requests:   {stats.num_requests:,}")
    print(f"Failures:   {stats.num_failures:,}")
    # CPU/ONNX targets. GPU/TRT targets would be ~10x tighter.
    p50_limit, p95_limit, p99_limit = 800, 2000, 4000
    print(
        f"p50 latency: {p50:>6.0f}ms  {'[PASS]' if p50 < p50_limit else f'[FAIL] target < {p50_limit}ms'}"
    )
    print(
        f"p95 latency: {p95:>6.0f}ms  {'[PASS]' if p95 < p95_limit else f'[FAIL] target < {p95_limit}ms'}"
    )
    print(
        f"p99 latency: {p99:>6.0f}ms  {'[PASS]' if p99 < p99_limit else f'[FAIL] target < {p99_limit}ms'}"
    )
    print(
        f"Error rate:  {error_rate:>6.2f}%  {'[PASS]' if error_rate < 1 else '[FAIL] target < 1%'}"
    )
    print("=" * 60)

    # Fail the test run if any SLA is breached
    if p50 >= p50_limit or p95 >= p95_limit or p99 >= p99_limit or error_rate >= 1:
        environment.process_exit_code = 1
