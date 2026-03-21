"""Reasoning contracts and adapters for Continuity."""

from .base import (
    AnswerQueryRequest,
    ClaimDerivationRequest,
    RawStructuredOutput,
    ReasoningAdapter,
    ReasoningMessage,
    SchemaValidationError,
    SessionSummaryRequest,
    StructuredGenerationRequest,
    StructuredOutputSchema,
    TextResponse,
    ValidatedStructuredOutput,
    publish_authoritative_mutation,
    validate_structured_output,
)
from .claim_derivation import (
    ClaimDerivationCandidate,
    ClaimDerivationEnvelope,
    ClaimDerivationEnvelopeSchema,
    ClaimDerivationPipeline,
    ClaimDerivationResult,
    fingerprint_candidate_content,
)
from .codex_adapter import (
    CodexAdapter,
    CodexAdapterConfig,
    CodexAdapterError,
    CodexPromptPolicy,
    CodexSDKUnavailableError,
    CodexStructuredSpec,
    prompt_policy_for,
)
from .hermes_chat_adapter import (
    HermesChatAdapter,
    HermesChatAdapterConfig,
)

__all__ = [
    "AnswerQueryRequest",
    "ClaimDerivationRequest",
    "ClaimDerivationCandidate",
    "ClaimDerivationEnvelope",
    "ClaimDerivationEnvelopeSchema",
    "ClaimDerivationPipeline",
    "ClaimDerivationResult",
    "CodexAdapter",
    "CodexAdapterConfig",
    "CodexAdapterError",
    "CodexPromptPolicy",
    "CodexSDKUnavailableError",
    "CodexStructuredSpec",
    "HermesChatAdapter",
    "HermesChatAdapterConfig",
    "RawStructuredOutput",
    "ReasoningAdapter",
    "ReasoningMessage",
    "SchemaValidationError",
    "SessionSummaryRequest",
    "StructuredGenerationRequest",
    "StructuredOutputSchema",
    "TextResponse",
    "ValidatedStructuredOutput",
    "fingerprint_candidate_content",
    "publish_authoritative_mutation",
    "prompt_policy_for",
    "validate_structured_output",
]
