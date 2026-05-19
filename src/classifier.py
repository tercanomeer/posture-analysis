"""Rule-based posture classification on top of biomechanical measurements.

Layered design:

- ``Severity`` — the four possible verdicts a rule can return.
- ``PostureThresholds`` — frozen dataclass exposing every tunable knob; pass a
  custom instance to ``PostureClassifier`` to override defaults without
  subclassing anything.
- ``PostureRule`` — minimal base class for a single rule. Three concrete rules
  are provided: ``ForwardHeadRule``, ``ShoulderAsymmetryRule``,
  ``SlouchingRule``. Adding a new rule means implementing one method.
- ``PostureClassifier`` — composes a list of rules into a ``PostureAssessment``.

All rules are O(1) and allocate nothing per call — safe to invoke every frame
in a realtime loop.

NaN inputs (from occluded landmarks) yield ``Severity.UNKNOWN`` so callers can
distinguish "no data" from "good posture" without try/except.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Sequence, Tuple

from .biomechanics import PostureMetrics


class Severity(Enum):
    """Verdict returned by a single posture rule."""

    OK = "ok"
    MILD = "mild"
    SEVERE = "severe"
    UNKNOWN = "unknown"


# Ordering for "worst" aggregation. UNKNOWN never wins against a real verdict.
_SEVERITY_RANK = {
    Severity.UNKNOWN: 0,
    Severity.OK: 1,
    Severity.MILD: 2,
    Severity.SEVERE: 3,
}


def _worst(a: Severity, b: Severity) -> Severity:
    return a if _SEVERITY_RANK[a] >= _SEVERITY_RANK[b] else b


@dataclass(frozen=True, slots=True)
class PostureThresholds:
    """All tunable knobs for the default rule set.

    Defaults are deliberately conservative starting points. Tighten them for
    professional ergonomics use; loosen them for casual ambient monitoring.
    """

    # Forward head posture — neck_angle drops below these values (180 = ideal).
    forward_head_mild_deg: float = 160.0
    forward_head_severe_deg: float = 145.0

    # Shoulder asymmetry — absolute shoulder slope (0 = level).
    shoulder_mild_deg: float = 5.0
    shoulder_severe_deg: float = 10.0

    # Slouching — absolute torso inclination from vertical (0 = upright). On
    # a frontal webcam this captures lateral lean; a side-view camera plus
    # this same threshold would surface forward slouch.
    slouch_mild_deg: float = 10.0
    slouch_severe_deg: float = 20.0


# ────────────────────────────────────────────────────────────────────────────
# Rules
# ────────────────────────────────────────────────────────────────────────────

class PostureRule:
    """Base class for a single posture rule. Subclasses implement ``evaluate``."""

    name: str = "rule"
    label: str = "rule"

    def evaluate(self, metrics: PostureMetrics) -> Severity:
        raise NotImplementedError


class ForwardHeadRule(PostureRule):
    """Triggers when the head sits forward of the torso (neck flexion)."""

    name = "forward_head"
    label = "Forward head"

    def __init__(self, mild_deg: float, severe_deg: float) -> None:
        if not severe_deg < mild_deg:
            raise ValueError("severe_deg must be smaller than mild_deg for forward head")
        self._mild = mild_deg
        self._severe = severe_deg

    def evaluate(self, metrics: PostureMetrics) -> Severity:
        a = metrics.neck_angle
        if math.isnan(a):
            return Severity.UNKNOWN
        if a < self._severe:
            return Severity.SEVERE
        if a < self._mild:
            return Severity.MILD
        return Severity.OK


class ShoulderAsymmetryRule(PostureRule):
    """Triggers when the shoulder line deviates from horizontal."""

    name = "shoulder_asymmetry"
    label = "Shoulder asymmetry"

    def __init__(self, mild_deg: float, severe_deg: float) -> None:
        if not mild_deg < severe_deg:
            raise ValueError("severe_deg must be larger than mild_deg for shoulder asymmetry")
        self._mild = mild_deg
        self._severe = severe_deg

    def evaluate(self, metrics: PostureMetrics) -> Severity:
        s = metrics.shoulder_slope
        if math.isnan(s):
            return Severity.UNKNOWN
        # ±180° means the LEFT/RIGHT shoulder labels arrive flipped in the
        # image (e.g. mirrored view): the line is still level, so collapse
        # the angle into [-90, 90] before magnitude comparison.
        mag = abs(_fold_to_horizontal(s))
        if mag >= self._severe:
            return Severity.SEVERE
        if mag >= self._mild:
            return Severity.MILD
        return Severity.OK


class SlouchingRule(PostureRule):
    """Triggers when the torso deviates from vertical."""

    name = "slouching"
    label = "Slouching"

    def __init__(self, mild_deg: float, severe_deg: float) -> None:
        if not mild_deg < severe_deg:
            raise ValueError("severe_deg must be larger than mild_deg for slouching")
        self._mild = mild_deg
        self._severe = severe_deg

    def evaluate(self, metrics: PostureMetrics) -> Severity:
        t = metrics.torso_inclination
        if math.isnan(t):
            return Severity.UNKNOWN
        mag = abs(t)
        if mag >= self._severe:
            return Severity.SEVERE
        if mag >= self._mild:
            return Severity.MILD
        return Severity.OK


def _fold_to_horizontal(deg: float) -> float:
    """Map any angle to the equivalent acute deviation from horizontal in [-90, 90]."""
    x = ((deg + 90.0) % 180.0) - 90.0
    return x


# ────────────────────────────────────────────────────────────────────────────
# Assessment + classifier
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PostureAssessment:
    """Per-rule verdicts plus a readable summary label."""

    findings: Tuple[Tuple[str, Severity], ...]  # ((rule_label, severity), ...)

    @property
    def overall(self) -> Severity:
        result = Severity.UNKNOWN
        for _, s in self.findings:
            result = _worst(result, s)
        return result

    @property
    def label(self) -> str:
        problems = [name for name, s in self.findings if s in (Severity.MILD, Severity.SEVERE)]
        if not problems:
            if all(s == Severity.UNKNOWN for _, s in self.findings):
                return "No detection"
            return "Good posture"
        return "; ".join(
            f"{name} ({s.value})" for name, s in self.findings if s in (Severity.MILD, Severity.SEVERE)
        )

    def __iter__(self):
        return iter(self.findings)


class PostureClassifier:
    """Applies a list of ``PostureRule``s to a ``PostureMetrics`` snapshot."""

    def __init__(
        self,
        thresholds: PostureThresholds = PostureThresholds(),
        rules: Sequence[PostureRule] | None = None,
    ) -> None:
        if rules is None:
            rules = (
                ForwardHeadRule(
                    mild_deg=thresholds.forward_head_mild_deg,
                    severe_deg=thresholds.forward_head_severe_deg,
                ),
                ShoulderAsymmetryRule(
                    mild_deg=thresholds.shoulder_mild_deg,
                    severe_deg=thresholds.shoulder_severe_deg,
                ),
                SlouchingRule(
                    mild_deg=thresholds.slouch_mild_deg,
                    severe_deg=thresholds.slouch_severe_deg,
                ),
            )
        self._rules: Tuple[PostureRule, ...] = tuple(rules)
        self.thresholds = thresholds

    @property
    def rules(self) -> Tuple[PostureRule, ...]:
        return self._rules

    def classify(self, metrics: PostureMetrics) -> PostureAssessment:
        return PostureAssessment(
            findings=tuple((rule.label, rule.evaluate(metrics)) for rule in self._rules),
        )
