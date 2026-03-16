from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Literal, Protocol, TypeVar, runtime_checkable


MessageRole = Literal["system", "user", "assistant", "tool"]

StructuredPayloadT = TypeVar("StructuredPayloadT")
StructuredPayloadT_co = TypeVar("StructuredPayloadT_co", covariant=True)
PublishResultT = TypeVar("PublishResultT")


@dataclass(frozen=True)
class ReasoningMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True)
class AnswerQueryRequest:
    query: str
    messages: tuple[ReasoningMessage, ...] = ()


@dataclass(frozen=True)
class StructuredGenerationRequest:
    instructions: str
    messages: tuple[ReasoningMessage, ...] = ()


@dataclass(frozen=True)
class SessionSummaryRequest:
    session_key: str
    messages: tuple[ReasoningMessage, ...] = ()


@dataclass(frozen=True)
class ClaimDerivationRequest:
    observations: tuple[ReasoningMessage, ...] = ()


@dataclass(frozen=True)
class TextResponse:
    text: str


@dataclass(frozen=True)
class RawStructuredOutput:
    payload: object


@dataclass(frozen=True)
class ValidatedStructuredOutput(Generic[StructuredPayloadT_co]):
    schema_name: str
    payload: StructuredPayloadT_co


class SchemaValidationError(ValueError):
    def __init__(self, *, schema_name: str, message: str) -> None:
        super().__init__(f"{schema_name}: {message}")
        self.schema_name = schema_name
        self.message = message


@runtime_checkable
class StructuredOutputSchema(Protocol[StructuredPayloadT_co]):
    name: str

    def validate(self, payload: object) -> StructuredPayloadT_co: ...


@runtime_checkable
class ReasoningAdapter(Protocol):
    def answer_query(self, request: AnswerQueryRequest) -> TextResponse: ...

    def generate_structured(self, request: StructuredGenerationRequest) -> RawStructuredOutput: ...

    def summarize_session(self, request: SessionSummaryRequest) -> TextResponse: ...

    def derive_claims(self, request: ClaimDerivationRequest) -> RawStructuredOutput: ...


def validate_structured_output(
    output: RawStructuredOutput,
    schema: StructuredOutputSchema[StructuredPayloadT],
) -> ValidatedStructuredOutput[StructuredPayloadT]:
    try:
        payload = schema.validate(output.payload)
    except SchemaValidationError:
        raise
    except Exception as exc:
        raise SchemaValidationError(schema_name=schema.name, message=str(exc)) from exc

    return ValidatedStructuredOutput(schema_name=schema.name, payload=payload)


def publish_authoritative_mutation(
    validated_output: ValidatedStructuredOutput[StructuredPayloadT],
    publish: Callable[[StructuredPayloadT], PublishResultT],
) -> PublishResultT:
    if not isinstance(validated_output, ValidatedStructuredOutput):
        raise TypeError("authoritative mutation requires schema-validated structured output")

    return publish(validated_output.payload)
