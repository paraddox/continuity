"""Disclosure and audience contract invariants for Continuity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache


def _clean_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _optional_clean_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name)


def _dedupe(values: tuple[StrEnum, ...]) -> tuple[StrEnum, ...]:
    return tuple(dict.fromkeys(values))


class DisclosurePrincipal(StrEnum):
    ASSISTANT_INTERNAL = "assistant_internal"
    CURRENT_USER = "current_user"
    CURRENT_PEER = "current_peer"
    SHARED_SESSION = "shared_session"
    HOST_INTERNAL = "host_internal"


class ViewerKind(StrEnum):
    ASSISTANT = "assistant"
    USER = "user"
    PEER = "peer"
    HOST = "host"


class DisclosureChannel(StrEnum):
    PROMPT = "prompt"
    ANSWER = "answer"
    SEARCH = "search"
    PROFILE = "profile"
    EVIDENCE = "evidence"
    REPLAY = "replay"
    MIGRATION = "migration"
    INSPECTION = "inspection"


class DisclosurePurpose(StrEnum):
    PROMPT = "prompt"
    ANSWER = "answer"
    SEARCH = "search"
    PROFILE = "profile"
    EVIDENCE = "evidence"
    REPLAY = "replay"
    MIGRATION = "migration"
    INSPECTION = "inspection"


class DisclosureAction(StrEnum):
    ALLOW = "allow"
    SUMMARIZE = "summarize"
    REDACT = "redact"
    WITHHOLD = "withhold"
    NEEDS_CONSENT = "needs_consent"

    @property
    def severity(self) -> int:
        return {
            DisclosureAction.ALLOW: 1,
            DisclosureAction.SUMMARIZE: 2,
            DisclosureAction.REDACT: 3,
            DisclosureAction.NEEDS_CONSENT: 4,
            DisclosureAction.WITHHOLD: 5,
        }[self]


@dataclass(frozen=True, slots=True)
class DisclosureViewer:
    viewer_kind: ViewerKind
    viewer_subject_id: str | None = None
    active_user_id: str | None = None
    active_peer_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "viewer_subject_id",
            _optional_clean_text(self.viewer_subject_id, field_name="viewer_subject_id"),
        )
        object.__setattr__(
            self,
            "active_user_id",
            _optional_clean_text(self.active_user_id, field_name="active_user_id"),
        )
        object.__setattr__(
            self,
            "active_peer_id",
            _optional_clean_text(self.active_peer_id, field_name="active_peer_id"),
        )

        if self.viewer_kind in {ViewerKind.USER, ViewerKind.PEER} and self.viewer_subject_id is None:
            raise ValueError("user and peer viewers require viewer_subject_id")


@dataclass(frozen=True, slots=True)
class DisclosureContext:
    viewer: DisclosureViewer
    audience_principal: DisclosurePrincipal
    channel: DisclosureChannel
    purpose: DisclosurePurpose
    policy_stamp: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_stamp", _clean_text(self.policy_stamp, field_name="policy_stamp"))


@dataclass(frozen=True, slots=True)
class DisclosurePolicy:
    policy_name: str
    principal: DisclosurePrincipal
    allowed_viewers: frozenset[ViewerKind]
    allowed_channels: tuple[DisclosureChannel, ...]
    allowed_purposes: tuple[DisclosurePurpose, ...]
    default_action: DisclosureAction = DisclosureAction.ALLOW
    default_reason: str | None = None
    consent_required_purposes: tuple[DisclosurePurpose, ...] = ()
    cross_peer_sensitive: bool = False
    capture_for_replay: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_name", _clean_text(self.policy_name, field_name="policy_name"))
        object.__setattr__(
            self,
            "default_reason",
            _optional_clean_text(self.default_reason, field_name="default_reason"),
        )
        object.__setattr__(self, "allowed_channels", _dedupe(self.allowed_channels))
        object.__setattr__(self, "allowed_purposes", _dedupe(self.allowed_purposes))
        object.__setattr__(
            self,
            "consent_required_purposes",
            _dedupe(self.consent_required_purposes),
        )

        if not self.allowed_viewers:
            raise ValueError("allowed_viewers must be non-empty")
        if not self.allowed_channels:
            raise ValueError("allowed_channels must be non-empty")
        if not self.allowed_purposes:
            raise ValueError("allowed_purposes must be non-empty")

        if (
            self.default_action in {
                DisclosureAction.SUMMARIZE,
                DisclosureAction.REDACT,
                DisclosureAction.WITHHOLD,
            }
            and self.default_reason is None
        ):
            raise ValueError("transforming or withholding policies require a default_reason")


@dataclass(frozen=True, slots=True)
class DisclosureDecision:
    policy: DisclosurePolicy
    context: DisclosureContext
    action: DisclosureAction
    reason: str
    captured_for_replay: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _clean_text(self.reason, field_name="reason"))

    @property
    def utility_signal(self) -> str | None:
        return {
            DisclosureAction.ALLOW: None,
            DisclosureAction.SUMMARIZE: "summarized",
            DisclosureAction.REDACT: "redacted",
            DisclosureAction.WITHHOLD: "withheld",
            DisclosureAction.NEEDS_CONSENT: "withheld",
        }[self.action]

    @property
    def exposes_content(self) -> bool:
        return self.action in {
            DisclosureAction.ALLOW,
            DisclosureAction.SUMMARIZE,
            DisclosureAction.REDACT,
        }


def _compose_principal(
    left: DisclosurePrincipal,
    right: DisclosurePrincipal,
) -> DisclosurePrincipal:
    if left is right:
        return left

    shared_pair = {left, right}
    if shared_pair == {
        DisclosurePrincipal.SHARED_SESSION,
        DisclosurePrincipal.CURRENT_USER,
    }:
        return DisclosurePrincipal.CURRENT_USER
    if shared_pair == {
        DisclosurePrincipal.SHARED_SESSION,
        DisclosurePrincipal.CURRENT_PEER,
    }:
        return DisclosurePrincipal.CURRENT_PEER

    raise ValueError("incompatible disclosure principals cannot be composed")


def compose_disclosure_policies(
    *policies: DisclosurePolicy,
    policy_name: str,
) -> DisclosurePolicy:
    if not policies:
        raise ValueError("at least one disclosure policy is required")

    principal = policies[0].principal
    for policy in policies[1:]:
        principal = _compose_principal(principal, policy.principal)

    allowed_viewers = set(policies[0].allowed_viewers)
    for policy in policies[1:]:
        allowed_viewers &= set(policy.allowed_viewers)

    allowed_channels = tuple(
        channel
        for channel in policies[0].allowed_channels
        if all(channel in policy.allowed_channels for policy in policies[1:])
    )
    allowed_purposes = tuple(
        purpose
        for purpose in policies[0].allowed_purposes
        if all(purpose in policy.allowed_purposes for policy in policies[1:])
    )

    if not allowed_viewers:
        raise ValueError("disclosure composition removed every allowed viewer")
    if not allowed_channels:
        raise ValueError("disclosure composition removed every allowed channel")
    if not allowed_purposes:
        raise ValueError("disclosure composition removed every allowed purpose")

    strictest_policy = max(policies, key=lambda policy: policy.default_action.severity)
    consent_required_purposes = tuple(
        dict.fromkeys(
            purpose
            for policy in policies
            for purpose in policy.consent_required_purposes
            if purpose in allowed_purposes
        )
    )

    return DisclosurePolicy(
        policy_name=policy_name,
        principal=principal,
        allowed_viewers=frozenset(allowed_viewers),
        allowed_channels=allowed_channels,
        allowed_purposes=allowed_purposes,
        default_action=strictest_policy.default_action,
        default_reason=strictest_policy.default_reason,
        consent_required_purposes=consent_required_purposes,
        cross_peer_sensitive=any(policy.cross_peer_sensitive for policy in policies),
        capture_for_replay=any(policy.capture_for_replay for policy in policies),
    )


def _audience_matches(
    policy_principal: DisclosurePrincipal,
    requested_principal: DisclosurePrincipal,
) -> bool:
    if policy_principal is requested_principal:
        return True
    return (
        policy_principal is DisclosurePrincipal.SHARED_SESSION
        and requested_principal
        in {
            DisclosurePrincipal.CURRENT_USER,
            DisclosurePrincipal.CURRENT_PEER,
            DisclosurePrincipal.SHARED_SESSION,
        }
    )


def _decision(
    *,
    policy: DisclosurePolicy,
    context: DisclosureContext,
    action: DisclosureAction,
    reason: str,
) -> DisclosureDecision:
    return DisclosureDecision(
        policy=policy,
        context=context,
        action=action,
        reason=reason,
        captured_for_replay=policy.capture_for_replay,
    )


def evaluate_disclosure(
    policy: DisclosurePolicy,
    context: DisclosureContext,
) -> DisclosureDecision:
    if context.viewer.viewer_kind not in policy.allowed_viewers:
        return _decision(
            policy=policy,
            context=context,
            action=DisclosureAction.WITHHOLD,
            reason="viewer_kind_not_allowed",
        )

    if not _audience_matches(policy.principal, context.audience_principal):
        return _decision(
            policy=policy,
            context=context,
            action=DisclosureAction.WITHHOLD,
            reason="audience_principal_mismatch",
        )

    if context.channel not in policy.allowed_channels:
        return _decision(
            policy=policy,
            context=context,
            action=DisclosureAction.WITHHOLD,
            reason="channel_not_allowed",
        )

    if context.purpose not in policy.allowed_purposes:
        return _decision(
            policy=policy,
            context=context,
            action=DisclosureAction.WITHHOLD,
            reason="purpose_not_allowed",
        )

    if (
        policy.cross_peer_sensitive
        and context.viewer.viewer_kind is ViewerKind.PEER
        and (
            context.viewer.active_peer_id is None
            or context.viewer.viewer_subject_id != context.viewer.active_peer_id
        )
    ):
        return _decision(
            policy=policy,
            context=context,
            action=DisclosureAction.WITHHOLD,
            reason="cross_peer_block",
        )

    if context.purpose in policy.consent_required_purposes:
        return _decision(
            policy=policy,
            context=context,
            action=DisclosureAction.NEEDS_CONSENT,
            reason="withheld_requires_consent",
        )

    default_reason = policy.default_reason or f"policy:{policy.policy_name}"
    return _decision(
        policy=policy,
        context=context,
        action=policy.default_action,
        reason=default_reason,
    )


@lru_cache(maxsize=1)
def v1_disclosure_policies() -> dict[str, DisclosurePolicy]:
    standard_host_channels = (
        DisclosureChannel.PROMPT,
        DisclosureChannel.ANSWER,
        DisclosureChannel.SEARCH,
        DisclosureChannel.PROFILE,
        DisclosureChannel.EVIDENCE,
    )
    standard_host_purposes = (
        DisclosurePurpose.PROMPT,
        DisclosurePurpose.ANSWER,
        DisclosurePurpose.SEARCH,
        DisclosurePurpose.PROFILE,
        DisclosurePurpose.EVIDENCE,
    )
    return {
        "assistant_internal": DisclosurePolicy(
            policy_name="assistant_internal",
            principal=DisclosurePrincipal.ASSISTANT_INTERNAL,
            allowed_viewers=frozenset({ViewerKind.ASSISTANT}),
            allowed_channels=(
                DisclosureChannel.PROMPT,
                DisclosureChannel.INSPECTION,
                DisclosureChannel.EVIDENCE,
                DisclosureChannel.REPLAY,
            ),
            allowed_purposes=(
                DisclosurePurpose.PROMPT,
                DisclosurePurpose.INSPECTION,
                DisclosurePurpose.EVIDENCE,
                DisclosurePurpose.REPLAY,
            ),
        ),
        "current_user": DisclosurePolicy(
            policy_name="current_user",
            principal=DisclosurePrincipal.CURRENT_USER,
            allowed_viewers=frozenset({ViewerKind.ASSISTANT, ViewerKind.USER}),
            allowed_channels=standard_host_channels,
            allowed_purposes=standard_host_purposes,
        ),
        "current_peer": DisclosurePolicy(
            policy_name="current_peer",
            principal=DisclosurePrincipal.CURRENT_PEER,
            allowed_viewers=frozenset({ViewerKind.ASSISTANT, ViewerKind.PEER}),
            allowed_channels=standard_host_channels,
            allowed_purposes=standard_host_purposes,
            cross_peer_sensitive=True,
        ),
        "shared_session": DisclosurePolicy(
            policy_name="shared_session",
            principal=DisclosurePrincipal.SHARED_SESSION,
            allowed_viewers=frozenset(
                {
                    ViewerKind.ASSISTANT,
                    ViewerKind.USER,
                    ViewerKind.PEER,
                }
            ),
            allowed_channels=standard_host_channels,
            allowed_purposes=standard_host_purposes,
        ),
        "host_internal": DisclosurePolicy(
            policy_name="host_internal",
            principal=DisclosurePrincipal.HOST_INTERNAL,
            allowed_viewers=frozenset({ViewerKind.ASSISTANT, ViewerKind.HOST}),
            allowed_channels=(
                DisclosureChannel.INSPECTION,
                DisclosureChannel.MIGRATION,
                DisclosureChannel.REPLAY,
                DisclosureChannel.EVIDENCE,
            ),
            allowed_purposes=(
                DisclosurePurpose.INSPECTION,
                DisclosurePurpose.MIGRATION,
                DisclosurePurpose.REPLAY,
                DisclosurePurpose.EVIDENCE,
            ),
        ),
    }


def disclosure_policy_for(policy_name: str) -> DisclosurePolicy:
    cleaned = _clean_text(policy_name, field_name="policy_name")
    return v1_disclosure_policies()[cleaned]
