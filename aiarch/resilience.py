"""Discrete-event simulation of the synchronous authorization path.

A pool of inference workers serves requests first-come, first-served. Requests that the
caller abandons are NOT cancelled server-side (a common default), so they still consume
worker time. Three caller designs are compared:
  naive      : wait for the model up to a 2 s client timeout; on timeout the payment errors
  budget     : fixed time budget; on timeout use the pre-approved fallback rules
  budget_cb  : time budget plus a circuit breaker that stops calling an unhealthy model
"""
import heapq
from collections import deque
import numpy as np

POLICY_MS = 1.0      # policy-engine evaluation time
FALLBACK_MS = 2.0    # deterministic fallback rules evaluation time


class CircuitBreaker:
    def __init__(self, window=20, min_calls=10, fail_ratio=0.5, open_ms=5000.0, probes=3):
        self.window, self.min_calls, self.fail_ratio = window, min_calls, fail_ratio
        self.open_ms, self.probes = open_ms, probes
        self.state, self.recent, self.open_until = "closed", deque(maxlen=window), 0.0
        self.probes_sent = self.probes_ok = 0

    def allow(self, t):
        if self.state == "open":
            if t < self.open_until:
                return False, False
            self.state, self.probes_sent, self.probes_ok = "half", 0, 0
        if self.state == "half":
            if self.probes_sent >= self.probes:
                return False, False
            self.probes_sent += 1
            return True, True
        return True, False

    def record(self, t, ok, is_probe):
        if self.state == "closed":
            self.recent.append(ok)
            if len(self.recent) >= self.min_calls and \
                    (1 - sum(self.recent) / len(self.recent)) >= self.fail_ratio:
                self._open(t)
        elif self.state == "half" and is_probe:
            if not ok:
                self._open(t)
            else:
                self.probes_ok += 1
                if self.probes_ok >= self.probes:
                    self.state = "closed"; self.recent.clear()

    def _open(self, t):
        self.state, self.open_until = "open", t + self.open_ms
        self.recent.clear()


def simulate(arrivals_ms, service_ms, mode, budget_ms=100.0, client_timeout_ms=2000.0, workers=8):
    n = len(arrivals_ms)
    latency = np.empty(n); used_model = np.zeros(n, bool); error = np.zeros(n, bool)
    sent = np.zeros(n, bool)
    free = [0.0] * workers; heapq.heapify(free)
    cb = CircuitBreaker() if mode == "budget_cb" else None
    pending = []  # (observe_time, seq, ok, is_probe)
    limit = client_timeout_ms if mode == "naive" else budget_ms
    for i in range(n):
        t = arrivals_ms[i]
        if cb:
            while pending and pending[0][0] <= t:
                ot, _, ok, pr = heapq.heappop(pending)
                cb.record(ot, ok, pr)
            allowed, is_probe = cb.allow(t)
            if not allowed:
                latency[i] = FALLBACK_MS + POLICY_MS
                continue
        else:
            is_probe = False
        sent[i] = True
        start = max(t, heapq.heappop(free))
        finish = start + (service_ms(start, i) if callable(service_ms) else service_ms[i])
        heapq.heappush(free, finish)
        resp = finish - t
        ok = resp <= limit
        if ok:
            latency[i] = resp + POLICY_MS; used_model[i] = True
        elif mode == "naive":
            latency[i] = limit; error[i] = True
        else:
            latency[i] = limit + FALLBACK_MS + POLICY_MS
        if cb:
            heapq.heappush(pending, (t + min(resp, limit), i, ok, is_probe))
    return dict(latency_ms=latency, used_model=used_model, error=error, sent=sent)


def fault_schedule(arrivals_ms, rng, base_median_ms=15.0, sigma=0.4,
                   degraded=(200_000, 320_000), slow_factor=4.0,
                   outage=(320_000, 380_000), hang_ms=5000.0):
    """Service time depends on when a request STARTS on a worker, not when it arrived."""
    base = rng.lognormal(np.log(base_median_ms), sigma, len(arrivals_ms))

    def service(start, i):
        if outage[0] <= start < outage[1]:
            return hang_ms
        if degraded[0] <= start < degraded[1]:
            return base[i] * slow_factor
        return base[i]
    return service
