"""
Apex Harness — Context Management System.

Provides:
- Role-based context filtering (ceo/orchestrator/cluster_leader/worker)
- Context compression with configurable max_tokens
- Context enrichment from memory system
- Context validation with schema enforcement
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class Role(Enum):
    """System roles with different context access levels."""

    CEO = "ceo"
    ORCHESTRATOR = "orchestrator"
    CLUSTER_LEADER = "cluster_leader"
    WORKER = "worker"


class ContextField(Enum):
    """Standard context fields that can be filtered by role."""

    SYSTEM_PROMPT = "system_prompt"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_RESULTS = "tool_results"
    MEMORY = "memory"
    METADATA = "metadata"
    ERRORS = "errors"
    TELEMETRY = "telemetry"
    AUDIT_LOG = "audit_log"
    CROSS_SESSION = "cross_session"


@dataclass
class ContextBlock:
    """A single block of context information."""

    field: ContextField
    content: str
    tokens: int = 0
    priority: float = 0.5  # 0.0-1.0, higher = more important
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    required_roles: list[Role] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tokens == 0:
            self.tokens = self._estimate_tokens(self.content)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English text."""
        return max(1, len(text) // 4)


@dataclass
class ContextWindow:
    """A complete context window with multiple blocks."""

    blocks: list[ContextBlock] = field(default_factory=list)
    max_tokens: int = 2000
    role: Role = Role.WORKER
    user_id: str = ""
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all blocks."""
        return sum(b.tokens for b in self.blocks)

    @property
    def remaining_tokens(self) -> int:
        """Remaining token budget."""
        return max(0, self.max_tokens - self.total_tokens)

    def add_block(self, block: ContextBlock) -> bool:
        """Add a block if it fits within budget."""
        if self.total_tokens + block.tokens > self.max_tokens:
            return False
        self.blocks.append(block)
        return True

    def to_messages(self) -> list[dict[str, str]]:
        """Convert context blocks to message dicts."""
        messages: list[dict[str, str]] = []
        for block in self.blocks:
            role_map = {
                ContextField.SYSTEM_PROMPT: "system",
                ContextField.USER_MESSAGE: "user",
                ContextField.ASSISTANT_MESSAGE: "assistant",
                ContextField.TOOL_RESULTS: "tool",
            }
            msg_role = role_map.get(block.field, "user")
            messages.append({
                "role": msg_role,
                "content": block.content,
            })
        return messages


@dataclass
class ValidationResult:
    """Result of context validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    token_count: int = 0

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another validation result into this one."""
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            token_count=max(self.token_count, other.token_count),
        )


# ---------------------------------------------------------------------------
# Role-Based Filtering
# ---------------------------------------------------------------------------


class RoleContextFilter:
    """Filters context blocks based on role permissions."""

    # Role hierarchy: higher index = more access
    ROLE_HIERARCHY: dict[Role, int] = {
        Role.WORKER: 0,
        Role.CLUSTER_LEADER: 1,
        Role.ORCHESTRATOR: 2,
        Role.CEO: 3,
    }

    # Default field access by role tier
    FIELD_ACCESS: dict[int, set[ContextField]] = {
        0: {  # Worker
            ContextField.SYSTEM_PROMPT,
            ContextField.USER_MESSAGE,
            ContextField.ASSISTANT_MESSAGE,
            ContextField.TOOL_RESULTS,
        },
        1: {  # Cluster Leader
            ContextField.SYSTEM_PROMPT,
            ContextField.USER_MESSAGE,
            ContextField.ASSISTANT_MESSAGE,
            ContextField.TOOL_RESULTS,
            ContextField.MEMORY,
            ContextField.METADATA,
        },
        2: {  # Orchestrator
            ContextField.SYSTEM_PROMPT,
            ContextField.USER_MESSAGE,
            ContextField.ASSISTANT_MESSAGE,
            ContextField.TOOL_RESULTS,
            ContextField.MEMORY,
            ContextField.METADATA,
            ContextField.ERRORS,
            ContextField.TELEMETRY,
        },
        3: {  # CEO — full access
            ContextField.SYSTEM_PROMPT,
            ContextField.USER_MESSAGE,
            ContextField.ASSISTANT_MESSAGE,
            ContextField.TOOL_RESULTS,
            ContextField.MEMORY,
            ContextField.METADATA,
            ContextField.ERRORS,
            ContextField.TELEMETRY,
            ContextField.AUDIT_LOG,
            ContextField.CROSS_SESSION,
        },
    }

    def __init__(self, role: Role) -> None:
        self._role = role
        self._tier = self.ROLE_HIERARCHY[role]
        self._allowed = self.FIELD_ACCESS[self._tier]

    def filter(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Filter context blocks to only those accessible by the role."""
        filtered: list[ContextBlock] = []
        for block in blocks:
            # Check role-specific required_roles constraint
            if block.required_roles and self._role not in block.required_roles:
                continue
            # Check tier-based field access
            if block.field in self._allowed:
                filtered.append(block)
        return filtered

    def can_access(self, field: ContextField) -> bool:
        """Check if the role can access a specific field type."""
        return field in self._allowed

    @property
    def role(self) -> Role:
        return self._role

    @property
    def tier(self) -> int:
        return self._tier


# ---------------------------------------------------------------------------
# Context Compression
# ---------------------------------------------------------------------------


class CompressionStrategy(ABC):
    """Abstract base for context compression strategies."""

    @abstractmethod
    def compress(self, blocks: list[ContextBlock], budget: int) -> list[ContextBlock]:
        """Compress blocks to fit within token budget."""
        ...


class TruncationCompressor(CompressionStrategy):
    """Simple truncation compressor — drops lowest-priority blocks first."""

    def compress(self, blocks: list[ContextBlock], budget: int) -> list[ContextBlock]:
        """Keep highest-priority blocks that fit in budget."""
        sorted_blocks = sorted(blocks, key=lambda b: b.priority, reverse=True)
        result: list[ContextBlock] = []
        used = 0
        for block in sorted_blocks:
            if used + block.tokens <= budget:
                result.append(block)
                used += block.tokens
        # Restore original order
        order_map = {id(b): i for i, b in enumerate(blocks)}
        result.sort(key=lambda b: order_map.get(id(b), 0))
        return result


class SummarizationCompressor(CompressionStrategy):
    """Compress by summarizing older blocks while keeping recent ones intact."""

    def __init__(
        self,
        summarize_fn: Callable[[str], str] | None = None,
        recent_threshold: int = 3,
    ) -> None:
        self._summarize_fn = summarize_fn or self._default_summarize
        self._recent_threshold = recent_threshold

    @staticmethod
    def _default_summarize(text: str) -> str:
        """Default summarization: keep first sentence + ellipsis."""
        sentences = text.split(". ")
        if len(sentences) <= 1:
            return text[:200] + "..." if len(text) > 200 else text
        return sentences[0] + ". [...]"

    def compress(self, blocks: list[ContextBlock], budget: int) -> list[ContextBlock]:
        """Summarize older blocks, keep recent ones intact."""
        if not blocks:
            return []

        # Keep last N blocks intact
        recent = blocks[-self._recent_threshold:]
        older = blocks[:-self._recent_threshold]

        # Summarize older blocks
        summarized: list[ContextBlock] = []
        for block in older:
            summary = self._summarize_fn(block.content)
            new_tokens = ContextBlock._estimate_tokens(summary)
            summarized.append(ContextBlock(
                field=block.field,
                content=summary,
                tokens=new_tokens,
                priority=block.priority * 0.8,  # Slightly reduce priority
                source=f"{block.source}:summarized",
                metadata={**block.metadata, "summarized": True},
                required_roles=block.required_roles,
            ))

        # Combine and truncate if still over budget
        combined = summarized + recent
        total = sum(b.tokens for b in combined)
        if total <= budget:
            return combined

        # Fall back to truncation for the remainder
        truncator = TruncationCompressor()
        return truncator.compress(combined, budget)


class SlidingWindowCompressor(CompressionStrategy):
    """Keep only the most recent blocks that fit in the budget."""

    def compress(self, blocks: list[ContextBlock], budget: int) -> list[ContextBlock]:
        """Keep the most recent blocks within budget."""
        result: list[ContextBlock] = []
        used = 0
        for block in reversed(blocks):
            if used + block.tokens <= budget:
                result.append(block)
                used += block.tokens
            else:
                break
        result.reverse()
        return result


# ---------------------------------------------------------------------------
# Context Enrichment
# ---------------------------------------------------------------------------


class ContextEnricher:
    """Enriches context blocks with relevant memories and metadata."""

    def __init__(
        self,
        memory_manager: Any | None = None,
        max_memories: int = 5,
    ) -> None:
        self._memory = memory_manager
        self._max_memories = max_memories

    def enrich(
        self,
        blocks: list[ContextBlock],
        user_id: str,
        query: str = "",
    ) -> list[ContextBlock]:
        """Enrich context with relevant memories."""
        if not self._memory or not query:
            return blocks

        try:
            result = self._memory.retrieve(
                query=query,
                user_id=user_id,
                limit=self._max_memories,
            )
            if result.entries:
                memory_content = "\n".join(
                    f"- {e.content}" for e in result.entries
                )
                memory_block = ContextBlock(
                    field=ContextField.MEMORY,
                    content=f"Relevant memories:\n{memory_content}",
                    priority=0.7,
                    source="memory_enricher",
                    metadata={
                        "strategy": result.strategy_used.value,
                        "latency_ms": result.latency_ms,
                        "count": len(result.entries),
                    },
                )
                blocks = [memory_block] + blocks
        except Exception as exc:
            logger.warning("Context enrichment failed: %s", exc)

        return blocks

    def add_system_context(
        self,
        blocks: list[ContextBlock],
        role: Role,
        session_id: str = "",
    ) -> list[ContextBlock]:
        """Prepend system context block with role and session info."""
        sys_content = (
            f"Role: {role.value}\n"
            f"Session: {session_id}\n"
            f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        sys_block = ContextBlock(
            field=ContextField.SYSTEM_PROMPT,
            content=sys_content,
            priority=1.0,
            source="system",
        )
        return [sys_block] + blocks


# ---------------------------------------------------------------------------
# Context Validation
# ---------------------------------------------------------------------------


class ContextValidator:
    """Validates context windows for correctness and completeness."""

    def __init__(
        self,
        max_tokens: int = 2000,
        max_blocks: int = 50,
        required_fields: list[ContextField] | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._max_blocks = max_blocks
        self._required_fields = required_fields or [ContextField.SYSTEM_PROMPT]

    def validate(self, window: ContextWindow) -> ValidationResult:
        """Validate a context window."""
        errors: list[str] = []
        warnings: list[str] = []

        # Token budget check
        if window.total_tokens > self._max_tokens:
            errors.append(
                f"Token budget exceeded: {window.total_tokens}/{self._max_tokens}"
            )

        # Block count check
        if len(window.blocks) > self._max_blocks:
            warnings.append(
                f"Block count high: {len(window.blocks)} (max {self._max_blocks})"
            )

        # Required fields check
        present_fields = {b.field for b in window.blocks}
        for req in self._required_fields:
            if req not in present_fields:
                errors.append(f"Missing required field: {req.value}")

        # Content quality checks
        for block in window.blocks:
            if not block.content.strip():
                warnings.append(f"Empty content in block: {block.field.value}")
            if block.tokens > self._max_tokens * 0.5:
                warnings.append(
                    f"Large block ({block.tokens} tokens): {block.field.value}"
                )

        # Role consistency
        for block in window.blocks:
            if block.required_roles and window.role not in block.required_roles:
                errors.append(
                    f"Role {window.role.value} cannot access block "
                    f"requiring {', '.join(r.value for r in block.required_roles)}"
                )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            token_count=window.total_tokens,
        )


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------


class ContextManager:
    """Unified context manager orchestrating filtering, compression, enrichment, and validation."""

    def __init__(
        self,
        max_tokens: int = 2000,
        compression_strategy: CompressionStrategy | None = None,
        memory_manager: Any | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._compressor = compression_strategy or TruncationCompressor()
        self._enricher = ContextEnricher(memory_manager=memory_manager)
        self._validator = ContextValidator(max_tokens=max_tokens)
        self._windows: dict[str, ContextWindow] = {}  # session_id -> window

    def create_window(
        self,
        role: Role,
        user_id: str,
        session_id: str = "",
        initial_blocks: list[ContextBlock] | None = None,
    ) -> ContextWindow:
        """Create a new context window for a role."""
        window = ContextWindow(
            blocks=initial_blocks or [],
            max_tokens=self._max_tokens,
            role=role,
            user_id=user_id,
            session_id=session_id,
        )
        if session_id:
            self._windows[session_id] = window
        return window

    def add_block(
        self,
        window: ContextWindow,
        block: ContextBlock,
        auto_compress: bool = True,
    ) -> bool:
        """Add a block to the window, compressing if necessary."""
        # Role filter check
        role_filter = RoleContextFilter(window.role)
        if not role_filter.can_access(block.field):
            logger.debug(
                "Block field %s not accessible by role %s",
                block.field.value,
                window.role.value,
            )
            return False

        if window.add_block(block):
            return True

        if auto_compress:
            # Try to compress existing blocks to make room
            compressed = self._compressor.compress(
                window.blocks, window.max_tokens - block.tokens
            )
            window.blocks = compressed
            return window.add_block(block)

        return False

    def compress(self, window: ContextWindow) -> ContextWindow:
        """Compress the window to fit within token budget."""
        window.blocks = self._compressor.compress(window.blocks, window.max_tokens)
        return window

    def enrich(
        self,
        window: ContextWindow,
        query: str = "",
    ) -> ContextWindow:
        """Enrich window with memories and system context."""
        window.blocks = self._enricher.enrich(
            window.blocks, window.user_id, query
        )
        window.blocks = self._enricher.add_system_context(
            window.blocks, window.role, window.session_id
        )
        return window

    def validate(self, window: ContextWindow) -> ValidationResult:
        """Validate the context window."""
        return self._validator.validate(window)

    def filter_for_role(
        self,
        window: ContextWindow,
        target_role: Role,
    ) -> ContextWindow:
        """Create a new window filtered for a different role."""
        role_filter = RoleContextFilter(target_role)
        filtered_blocks = role_filter.filter(window.blocks)
        return ContextWindow(
            blocks=filtered_blocks,
            max_tokens=window.max_tokens,
            role=target_role,
            user_id=window.user_id,
            session_id=window.session_id,
            metadata=window.metadata,
        )

    def get_window(self, session_id: str) -> ContextWindow | None:
        """Retrieve a window by session ID."""
        return self._windows.get(session_id)

    def to_messages(self, window: ContextWindow) -> list[dict[str, str]]:
        """Convert window to message list for LLM consumption."""
        return window.to_messages()
