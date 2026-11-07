"""Application-local bounded Prometheus instruments."""

from prometheus_client import CollectorRegistry, Counter, Histogram


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "credit_scoring_http_requests_total",
            "Completed requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "credit_scoring_http_request_duration_seconds",
            "Request duration",
            ["method", "route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )
        self.probability = Histogram(
            "credit_scoring_probability_of_default",
            "Successful scores",
            buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
            registry=self.registry,
        )
