"""Attribution: from failure signals to editable harness facets.

This is the differentiating layer. Every attribution must land on a concrete,
resolvable ``field_path``; otherwise it is not actionable and is dropped back to
the facet default.
"""

from __future__ import annotations

from collections import defaultdict

from harnessloop.attribution.labels import (
    LabelRegistry,
)
from harnessloop.attribution.labels import (
    get_registry as get_label_registry,
)
from harnessloop.attribution.plugins import AttributionRegistry, get_registry
from harnessloop.models import (
    Attribution,
    AttributionGroup,
    AttributionLabel,
    CheckResult,
    FailureSignal,
    HarnessFacet,
    HarnessState,
)


def _facet_for_path(path: str) -> HarnessFacet | None:
    if path == "system_prompt":
        return HarnessFacet.SYSTEM_PROMPT
    if path == "output_format":
        return HarnessFacet.OUTPUT_FORMAT
    if path.startswith("tools.") and path.endswith("description"):
        return HarnessFacet.TOOL_DESCRIPTION
    if path.startswith("tools.") and ".parameters" in path:
        return HarnessFacet.SCHEMA_FIELD
    return None


class Attributor:
    def __init__(
        self,
        registry: AttributionRegistry | None = None,
        labels: LabelRegistry | None = None,
    ) -> None:
        # Plugin rules run first; label defaults (built-in or custom) are fallback.
        self.registry = registry or get_registry()
        self.labels = labels or get_label_registry()

    def attribute(
        self, results: list[CheckResult], harness: HarnessState
    ) -> list[Attribution]:
        attributions: list[Attribution] = []
        for result in results:
            for signal in result.signals:
                label, facet, path = self._locate(signal, harness)
                attributions.append(
                    Attribution(
                        task_id=result.task_id,
                        label=label,
                        evidence=signal.evidence,
                        harness_facet=facet,
                        field_path=path,
                    )
                )
        return attributions

    def _locate(
        self, signal: FailureSignal, harness: HarnessState
    ) -> tuple[str, HarnessFacet, str]:
        # A custom label classifier can rename a built-in signal (e.g. a
        # RESULT_MISMATCH whose evidence says "nothing changed" is a NOOP_ACTION).
        label = self.labels.classify(signal, harness) or signal.label
        located = self.registry.resolve(signal, harness)
        if located is not None:
            return label, located[0], located[1]

        spec = self.labels.get(label)
        default_facet = spec.facet if spec is not None else HarnessFacet.SYSTEM_PROMPT
        default_path = spec.field_path if spec is not None else "system_prompt"

        candidates: list[str] = []
        # A custom label declares where it wants to land: honour that first.
        if spec is not None and not spec.builtin:
            candidates.append(spec.field_path)
        if signal.tool_name:
            if label == AttributionLabel.MISSING_PARAM:
                candidates.append(f"tools.{signal.tool_name}.parameters.required")
            elif label == AttributionLabel.EXTRA_PARAM:
                candidates.append(f"tools.{signal.tool_name}.parameters.properties")
            candidates.append(f"tools.{signal.tool_name}.description")
        candidates.append(default_path)

        for path in candidates:
            if harness.resolve_path(path):
                if spec is not None and path == spec.field_path:
                    return label, spec.facet, path
                return label, _facet_for_path(path) or default_facet, path
        if spec is not None and not spec.builtin:
            # A custom label must stay actionable even if its path is stale.
            return label, HarnessFacet.SYSTEM_PROMPT, "system_prompt"
        return label, default_facet, default_path

    def aggregate(self, attributions: list[Attribution]) -> list[AttributionGroup]:
        buckets: dict[tuple[str, str, str], list[Attribution]] = defaultdict(list)
        for attribution in attributions:
            key = (
                attribution.label,
                attribution.harness_facet.value,
                attribution.field_path or "",
            )
            buckets[key].append(attribution)

        groups: list[AttributionGroup] = []
        for items in buckets.values():
            first = items[0]
            task_ids: list[str] = []
            for item in items:
                if item.task_id not in task_ids:
                    task_ids.append(item.task_id)
            groups.append(
                AttributionGroup(
                    label=first.label,
                    count=len(items),
                    task_ids=task_ids,
                    facet=first.harness_facet,
                    field_path=first.field_path or "",
                    sample_evidence=first.evidence,
                )
            )
        groups.sort(key=lambda group: group.count, reverse=True)
        return groups
