"""Causal control and short-lived verification branches; no model or GT access.

Candidate IDs organize adjacent observations, never carry target identity mass.
An association gap or ambiguous connection breaks the confirmation history.
All thresholds are explicit engineering defaults, not calibrated probabilities.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import math
import numpy as np


class Mode(str, Enum):
    TRACKING = 'TRACKING'
    SEARCHING = 'SEARCHING'
    VERIFYING = 'VERIFYING'


@dataclass(frozen=True)
class Config:
    local_score_min: float = 0.5
    detector_score_min: float = 0.1
    verification_iou_min: float = 0.3
    association_iou_min: float = 0.2
    association_center_max: float = 1.0
    association_area_ratio_max: float = 4.0
    same_object_iou: float = 0.5
    confirmation_frames: int = 3
    max_candidates: int = 12

    def __post_init__(self):
        for name in ['local_score_min', 'detector_score_min', 'verification_iou_min',
                     'association_iou_min', 'same_object_iou']:
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be in (0, 1]')
        if not math.isfinite(self.association_center_max) or self.association_center_max <= 0:
            raise ValueError('association_center_max must be positive and finite')
        if not math.isfinite(self.association_area_ratio_max) or self.association_area_ratio_max < 1:
            raise ValueError('association_area_ratio_max must be finite and >= 1')
        for name, minimum in [('confirmation_frames', 2), ('max_candidates', 1)]:
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.max_candidates > 12:
            raise ValueError('The frozen detector returns at most 12 candidates')

    def to_dict(self):
        return asdict(self)


def box_array(box):
    a = np.asarray(box, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all() or (a[2:] <= 0).any():
        raise ValueError('Expected a finite, positive xywh box')
    return a.copy()


def clip_box(box, shape):
    a = box_array(box); h, w = shape[:2]
    lo = np.clip(a[:2], 0, [w, h]); hi = np.clip(a[:2] + a[2:], 0, [w, h])
    return box_array(np.r_[lo, hi - lo]).tolist()


def iou(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    inter = np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(a[2:].prod() + b[2:].prod() - inter, 1e-12))


def association_affinity(a, b, cfg):
    a = box_array(a); b = box_array(b)
    areas = [a[2:].prod(), b[2:].prod()]
    if max(areas) / min(areas) > cfg.association_area_ratio_max:
        return None
    distance = np.linalg.norm(a[:2] + a[2:] / 2 - b[:2] - b[2:] / 2) / math.sqrt(max(areas))
    overlap = iou(a, b)
    if overlap < cfg.association_iou_min and distance > cfg.association_center_max:
        return None
    # Used only for adjacent geometry association, never to rank target identity.
    return overlap + 1. / (1. + float(distance))


@dataclass
class Branch:
    id: int
    detection: dict
    birth_frame: int
    last_frame: int
    observations: int = 1
    shadow_box: list | None = None
    verified_streak: int = 0
    unique_streak: int = 0

    def clear_confirmation(self):
        self.shadow_box = None
        self.verified_streak = 0
        self.unique_streak = 0


class CandidateBank:
    """Adjacent-frame branches. Missing observations retire IDs immediately."""
    def __init__(self, config):
        self.cfg = config
        self.branches = []
        self.next_id = 1

    def update(self, candidates, frame):
        old = self.branches
        affinities = np.full((len(old), len(candidates)), -np.inf)
        for i, branch in enumerate(old):
            if branch.last_frame != frame - 1:
                continue
            for j, detection in enumerate(candidates):
                if detection.get('cls', 0) != branch.detection.get('cls', 0):
                    continue
                value = association_affinity(branch.detection['box'], detection['box'], self.cfg)
                if value is not None:
                    affinities[i, j] = value
        # Mutual unique best matches. Equal alternatives do not inherit a streak.
        matches = {}
        for j in range(len(candidates)):
            column = affinities[:, j]
            if not len(column) or not np.isfinite(column).any():
                continue
            best = column.max(); ids = np.flatnonzero(np.isclose(column, best, rtol=0, atol=1e-9))
            if len(ids) != 1:
                continue
            i = int(ids[0]); row = affinities[i]
            best_columns = np.flatnonzero(np.isclose(row, row.max(), rtol=0, atol=1e-9))
            if len(best_columns) == 1 and best_columns[0] == j:
                matches[j] = i
        current = []
        for j, detection in enumerate(candidates):
            if j in matches:
                branch = old[matches[j]]
                branch.detection = detection; branch.last_frame = frame; branch.observations += 1
            else:
                branch = Branch(self.next_id, detection, frame, frame)
                self.next_id += 1
            current.append(branch)
        retired = sorted(set(b.id for b in old) - set(b.id for b in current))
        self.branches = current
        return retired


def normalize_candidates(raw, shape, cfg):
    output = []; rejected = 0
    for item in raw:
        try:
            score = float(item['score'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid detector score')
            box = clip_box(item['box'], shape)
            cls = int(item.get('cls', 0))
        except (KeyError, TypeError, ValueError, OverflowError):
            rejected += 1
            continue
        if score >= cfg.detector_score_min:
            output.append(dict(box=box, score=score, cls=cls))
    output.sort(key=lambda c: (-c['score'], tuple(c['box']), c['cls']))
    unique = []; seen = set()
    for candidate in output:
        key = (candidate['cls'], tuple(candidate['box']))
        if key not in seen:
            unique.append(candidate); seen.add(key)
    return unique[:cfg.max_candidates], rejected


def proposal_valid(proposal, cfg):
    try:
        box_array(proposal['box'])
        score = float(proposal['score'])
        return math.isfinite(score) and cfg.local_score_min <= score <= 1
    except (ValueError, TypeError, KeyError):
        return False


def spatial_groups(evidence, cfg):
    """Complete-link groups prevent a chain of overlapping boxes merging rivals."""
    groups = []
    for item in sorted(evidence, key=lambda x: x['id']):
        fits = [g for g in groups if all(iou(item['box'], other['box']) >= cfg.same_object_iou for other in g)]
        if len(fits) == 1:
            fits[0].append(item)
        else:
            groups.append([item])
    return groups


class RecoveryController:
    def __init__(self, config):
        self.cfg = config
        self.reset()

    def reset(self):
        self.mode = Mode.TRACKING
        self.bank = CandidateBank(self.cfg)
        self.local_shadow = None
        self.local_verified_streak = 0
        self.local_unique_streak = 0

    def decide(self, image, local, detections, frame, probe):
        """Return output and commit as separate decisions. probe is read-only."""
        previous = self.mode
        candidates, rejected = normalize_candidates(detections, image.shape, self.cfg)
        retired = self.bank.update(candidates, frame)
        support = max((iou(local['box'], d['box']) for d in candidates), default=0.) if proposal_valid(local, self.cfg) else 0.
        local_good = proposal_valid(local, self.cfg) and (not candidates or support >= self.cfg.verification_iou_min)
        health_reason = 'LOCAL_SUPPORTED' if local_good and candidates else ('LOCAL_WITHOUT_DETECTION' if local_good else
                       ('LOCAL_DETECTOR_CONFLICT' if proposal_valid(local, self.cfg) else 'LOCAL_RESPONSE_UNRELIABLE'))
        branch_logs = []; evidence = []; probe_count = 0
        output = None; anchor = None; source = None
        if previous == Mode.TRACKING and local_good:
            self.mode = Mode.TRACKING
            output = list(local['box']); anchor = list(local['box']); source = 'local'
            reason = health_reason
            self.local_shadow = None; self.local_verified_streak = 0; self.local_unique_streak = 0
            for branch in self.bank.branches:
                branch.clear_confirmation()
        else:
            if previous == Mode.TRACKING:
                for branch in self.bank.branches:
                    branch.clear_confirmation()
                self.local_shadow = None; self.local_verified_streak = 0; self.local_unique_streak = 0
            # The original local path may also re-establish itself, but needs the
            # same consecutive, unambiguous confirmation as a remote candidate.
            if local_good:
                connected = self.local_shadow is not None and association_affinity(self.local_shadow, local['box'], self.cfg) is not None
                self.local_verified_streak = self.local_verified_streak + 1 if connected else 1
                if not connected:
                    self.local_unique_streak = 0
                self.local_shadow = list(local['box'])
                evidence.append(dict(id=0, box=list(local['box']), score=float(local['score']), source='local_reconfirmation'))
            else:
                self.local_shadow = None; self.local_verified_streak = 0; self.local_unique_streak = 0
            for branch in self.bank.branches:
                anchor_for_probe = branch.shadow_box if branch.shadow_box is not None else branch.detection['box']
                refined = probe(image, anchor_for_probe); probe_count += 1
                agreement = iou(refined['box'], branch.detection['box']) if proposal_valid(refined, self.cfg) else 0.
                passed = proposal_valid(refined, self.cfg) and agreement >= self.cfg.verification_iou_min
                if passed:
                    branch.shadow_box = list(refined['box']); branch.verified_streak += 1
                    evidence.append(dict(id=branch.id, box=list(refined['box']), score=float(refined['score']), source='global_verification'))
                else:
                    branch.clear_confirmation()
                branch_logs.append(dict(id=branch.id, detection=branch.detection, birth_frame=branch.birth_frame,
                    observations=branch.observations, probe_anchor=list(anchor_for_probe), proposal=refined,
                    agreement_iou=agreement, passed=bool(passed), verified_streak=branch.verified_streak))
            groups = spatial_groups(evidence, self.cfg)
            # Distinct credible detections without a passing verifier remain
            # unresolved alternatives. Do not silently discard them and promote
            # a high-response rival as the target.
            unresolved = []
            if len(groups) == 1:
                for d in candidates:
                    if not any(iou(d['box'], item['box']) >= self.cfg.verification_iou_min for item in groups[0]):
                        unresolved.append(d)
            unique = len(groups) == 1 and not unresolved
            unique_ids = {x['id'] for x in groups[0]} if unique else set()
            self.local_unique_streak = self.local_unique_streak + 1 if 0 in unique_ids else 0
            for branch in self.bank.branches:
                branch.unique_streak = branch.unique_streak + 1 if branch.id in unique_ids else 0
            streaks = {0: self.local_unique_streak, **{b.id: b.unique_streak for b in self.bank.branches}}
            ready = [item for item in evidence if item['id'] in unique_ids and streaks[item['id']] >= self.cfg.confirmation_frames]
            if ready:
                # Inside one spatial group only; no cross-object response argmax.
                chosen = min(ready, key=lambda x: x['id'])
                output = list(chosen['box']); anchor = list(chosen['box']); source = chosen['source']
                self.mode = Mode.TRACKING; reason = 'RECOVERY_CONFIRMED'
            elif evidence:
                self.mode = Mode.VERIFYING
                reason = 'AMBIGUOUS_CANDIDATES' if not unique else 'WAITING_FOR_CONFIRMATION'
            else:
                self.mode = Mode.SEARCHING; reason = 'NO_VERIFIED_CANDIDATE'
            for log in branch_logs:
                log['unique_streak'] = streaks[log['id']]
        return dict(frame=frame, previous_state=previous.value, state=self.mode.value, reason=reason,
            bbox=output, proposed_commit=anchor, output_source=source,
            localization_status='confirmed' if output is not None else 'unconfirmed',
            physical_presence='unknown', physical_absence_inferred=False,
            local_health=dict(accepted=bool(local_good), reason=health_reason, detector_agreement_iou=float(support)),
            active_candidates=candidates, rejected_invalid_candidates=rejected,
            retired_branch_ids=retired, verification_branches=branch_logs,
            local_confirmation_streak=self.local_unique_streak, probe_count=probe_count)
