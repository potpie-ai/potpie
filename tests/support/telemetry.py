"""Recording product analytics sink shared by root behavior contracts."""

from dataclasses import dataclass, field

from potpie.cli.telemetry.product_analytics import ProductAnalyticsEvent


@dataclass
class RecordingAnalyticsSink:
    events: list[ProductAnalyticsEvent] = field(default_factory=list)

    def capture(self, event: ProductAnalyticsEvent) -> None:
        self.events.append(event)
