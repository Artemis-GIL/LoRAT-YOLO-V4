"""V4: independent output/commit policy and shared local/global verification."""
if __package__:
    from .recovery_controller import CandidateBank, Mode, association_affinity, iou, normalize_candidates, spatial_groups
    from .reliability_v4 import SharedReferenceVerifier, usable
else:
    from recovery_controller import CandidateBank, Mode, association_affinity, iou, normalize_candidates, spatial_groups
    from reliability_v4 import SharedReferenceVerifier, usable


class RecoveryControllerV4:
    def __init__(self, config, reference_check):
        self.cfg = config
        self.verifier = SharedReferenceVerifier(config, reference_check)
        self.reset()

    def reset(self):
        self.mode = Mode.TRACKING
        self.bank = CandidateBank(self.cfg)
        self.local_shadow = None
        self.local_unique_streak = 0

    def decide(self, image, local, detections, frame, probe):
        previous = self.mode
        self.verifier.reset_frame()
        candidates, rejected = normalize_candidates(detections, image.shape, self.cfg)
        retired = self.bank.update(candidates, frame)
        local_check = self.verifier.assess(image, local)
        evidence = []; logs = []; probe_count = 0
        if local_check['passed']:
            connected = self.local_shadow is not None and association_affinity(self.local_shadow, local['box'], self.cfg) is not None
            if not connected:
                self.local_unique_streak = 0
            self.local_shadow = list(local['box'])
            evidence.append(dict(id=0, box=list(local['box']), score=float(local['score']), source='local'))
        else:
            self.local_shadow = None
            self.local_unique_streak = 0
        for branch in self.bank.branches:
            # A detection already at the verified local position supplies the
            # same spatial hypothesis. Distant candidates are always tested,
            # including in TRACKING; raw detection scores never veto a box.
            reuse = local_check['passed'] and iou(local['box'], branch.detection['box']) >= self.cfg.same_object_iou
            anchor = branch.shadow_box if branch.shadow_box is not None else branch.detection['box']
            if reuse:
                refined = local
                check = local_check
            else:
                refined = probe(image, anchor); probe_count += 1
                check = self.verifier.assess(image, refined)
            agreement = iou(refined['box'], branch.detection['box']) if usable(refined) else 0.
            if check['passed']:
                branch.shadow_box = list(refined['box']); branch.verified_streak += 1
                evidence.append(dict(id=branch.id, box=list(refined['box']), score=float(refined['score']), source='global_verification'))
            else:
                branch.clear_confirmation()
            logs.append(dict(id=branch.id, detection=branch.detection, birth_frame=branch.birth_frame,
                observations=branch.observations, probe_anchor=list(anchor), proposal=refined,
                agreement_iou=agreement, passed=bool(check['passed']), verified_streak=branch.verified_streak,
                proposal_reused_from_local=bool(reuse), reliability=check))
        groups = spatial_groups(evidence, self.cfg)
        unique = len(groups) == 1
        unique_ids = {item['id'] for item in groups[0]} if unique else set()
        self.local_unique_streak = self.local_unique_streak + 1 if 0 in unique_ids else 0
        for branch in self.bank.branches:
            branch.unique_streak = branch.unique_streak + 1 if branch.id in unique_ids else 0
        streaks = {0: self.local_unique_streak, **{b.id: b.unique_streak for b in self.bank.branches}}
        for log in logs: log['unique_streak'] = streaks[log['id']]
        commit = None; commit_source = None
        if previous == Mode.TRACKING and 0 in unique_ids:
            commit = list(local['box']); commit_source = 'local'; reason = 'LOCAL_REFERENCE_CONFIRMED'
        else:
            ready = [e for e in evidence if e['id'] in unique_ids and streaks[e['id']] >= self.cfg.confirmation_frames]
            if ready:
                chosen = min(ready, key=lambda e: e['id'])
                commit = list(chosen['box'])
                commit_source = 'local_reconfirmation' if chosen['id'] == 0 else 'global_verification'
                reason = 'RECOVERY_CONFIRMED'
            elif not evidence:
                reason = 'NO_VERIFIED_HYPOTHESIS'
            elif not unique:
                reason = 'AMBIGUOUS_VERIFIED_HYPOTHESES'
            else:
                reason = 'WAITING_FOR_CONFIRMATION'
        self.mode = Mode.TRACKING if commit is not None else (Mode.VERIFYING if evidence else Mode.SEARCHING)
        # Output has no feedback edge to the authoritative anchor. A usable
        # current local estimate remains visible during uncertainty/waiting.
        if commit is not None:
            output = list(commit); output_source = commit_source; output_status = 'confirmed'
        elif usable(local):
            output = list(local['box']); output_source = 'local_tentative'; output_status = 'tentative'
        elif unique:
            chosen = min(groups[0], key=lambda e: e['id'])
            output = list(chosen['box']); output_source = 'candidate_tentative'; output_status = 'tentative'
        else:
            output = None; output_source = None; output_status = 'unavailable'
        return dict(frame=frame, previous_state=previous.value, state=self.mode.value, reason=reason,
            bbox=output, output_source=output_source, proposed_commit=commit, commit_source=commit_source,
            localization_status=output_status, output_confirmed=commit is not None,
            physical_presence='unknown', physical_absence_inferred=False,
            local_health=dict(accepted=local_check['passed'], reason=local_check['reason'], reliability=local_check),
            active_candidates=candidates, rejected_invalid_candidates=rejected,
            retired_branch_ids=retired, verification_branches=logs, verified_spatial_groups=len(groups),
            local_confirmation_streak=self.local_unique_streak, probe_count=probe_count,
            reference_checks=self.verifier.calls, identity_guaranteed=False)
