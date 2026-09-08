from __future__ import annotations

from .types import ModelEndpoint, SpendRecord, Usage, utc_now


class BudgetManager:
    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = max(limit_usd, 0.0)
        self.records: list[SpendRecord] = []

    def set_limit(self, limit_usd: float) -> None:
        self.limit_usd = max(limit_usd, 0.0)

    @property
    def total_spend_usd(self) -> float:
        return sum(record.amount_usd for record in self.records)

    @property
    def remaining_usd(self) -> float:
        return self.limit_usd - self.total_spend_usd

    def can_spend(self, endpoint: ModelEndpoint) -> bool:
        if endpoint.provider_type == "local":
            return True
        return self.remaining_usd > 0

    def record_usage(
        self, agent: str, endpoint: ModelEndpoint, usage: Usage
    ) -> SpendRecord:
        amount = (usage.prompt_tokens / 1_000_000) * endpoint.input_cost_per_million + (
            usage.completion_tokens / 1_000_000
        ) * endpoint.output_cost_per_million
        record = SpendRecord(
            timestamp=utc_now(),
            agent=agent,
            provider=endpoint.provider,
            provider_type=endpoint.provider_type,
            model=endpoint.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            amount_usd=amount,
        )
        self.records.append(record)
        return record

    def summary(self) -> dict:
        cloud_spend = sum(
            record.amount_usd
            for record in self.records
            if record.provider_type == "cloud"
        )
        local_calls = sum(
            1 for record in self.records if record.provider_type == "local"
        )
        return {
            "limit_usd": round(self.limit_usd, 4),
            "spent_usd": round(self.total_spend_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "cloud_spend_usd": round(cloud_spend, 4),
            "local_calls": local_calls,
            "records": [
                {
                    "timestamp": record.timestamp.isoformat(),
                    "agent": record.agent,
                    "provider": record.provider,
                    "provider_type": record.provider_type,
                    "model": record.model,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "total_tokens": record.total_tokens,
                    "amount_usd": round(record.amount_usd, 6),
                }
                for record in self.records[-25:]
            ],
        }
