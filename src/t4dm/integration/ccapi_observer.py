"""
T4DM Observer for ccapi.

Implements the llm_agents Observer protocol to capture outcomes
and trigger learning updates in T4DM's neuro-symbolic system.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


# Type stubs for ccapi (avoid hard dependency)
class EventType:
    """Stub for llm_agents.observe.EventType."""
    AGENT_START = "agent.start"
    AGENT_END = "agent.end"
    AGENT_ERROR = "agent.error"
    TOOL_START = "tool.start"
    TOOL_END = "tool.end"
    TOOL_ERROR = "tool.error"
    LLM_RESPONSE = "llm.response"
    MEMORY_RETRIEVE = "memory.retrieve"


class Event:
    """Stub for llm_agents.observe.Event."""
    def __init__(
        self,
        type: str,
        name: str,
        data: dict[str, Any] | None = None,
        severity: str = "info",
        trace_id: str | None = None,
        span_id: str | None = None,
        duration_ms: float | None = None,
        tags: dict[str, str] | None = None,
    ):
        self.type = type
        self.name = name
        self.data = data or {}
        self.severity = severity
        self.trace_id = trace_id
        self.span_id = span_id
        self.duration_ms = duration_ms
        self.tags = tags or {}
        self.timestamp = datetime.now()


class Span:
    """Stub for llm_agents.observe.Span."""
    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
    ):
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.status = status
        self.attributes = attributes or {}
        self.start_time = datetime.now()
        self.end_time: datetime | None = None
        self.duration_ms: float | None = None


class WWObserver:
    """
    T4DM Observer for ccapi integration.

    Captures agent and tool outcomes, converts them to WW OutcomeEvents,
    and triggers credit assignment in the learning system.

    Key behaviors:
    - AGENT_END with status="ok" -> SUCCESS outcome
    - AGENT_END with status="error" -> FAILURE outcome
    - TOOL_END -> Records tool usage for procedural learning
    - MEMORY_RETRIEVE -> Tracks retrieval for credit assignment

    Usage:
        from t4dm.integration.ccapi_observer import WWObserver
        from llm_agents.observe import ObserverManager

        observer = WWObserver(session_id="agent-session-001")
        ObserverManager().add_observer(observer)

        # Agent runs, observer captures outcomes automatically
        # Learning system receives feedback signals
    """

    def __init__(
        self,
        session_id: str = "default",
        project: str = "",
        enable_learning: bool = True,
        buffer_size: int = 100,
    ):
        """
        Initialize WW Observer.

        Args:
            session_id: Session identifier for isolation
            project: Project context
            enable_learning: Enable learning feedback
            buffer_size: Maximum buffered events before flush
        """
        self.session_id = session_id
        self.project = project
        self.enable_learning = enable_learning
        self.buffer_size = buffer_size

        # Lazy-loaded WW components
        self._collector = None
        self._initialized = False

        # Event buffer (for batched processing)
        self._event_buffer: list[Event] = []

        # Track active spans for context
        self._active_spans: dict[str, Span] = {}

        # Track recent retrievals for credit assignment
        self._recent_retrievals: list[dict[str, Any]] = []
        self._max_retrievals = 50  # Keep last 50 retrievals

        # Pending async tasks for sync context
        self._pending_tasks: list = []

    def _schedule_async(self, coro) -> None:
        """Schedule an async coroutine, handling both sync and async contexts."""
        try:
            asyncio.get_running_loop()
            asyncio.create_task(coro)
        except RuntimeError:
            # No running loop - store for later or run in new loop
            self._pending_tasks.append(coro)
            # Try to run immediately if we have accumulated tasks
            if len(self._pending_tasks) >= 5:
                self._run_pending_tasks()

    def _run_pending_tasks(self) -> None:
        """Run all pending async tasks in a new event loop."""
        if not self._pending_tasks:
            return

        tasks = self._pending_tasks.copy()
        self._pending_tasks.clear()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            finally:
                loop.close()
        except Exception as e:
            logger.debug(f"Failed to run pending tasks: {e}")

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of WW learning components."""
        if self._initialized:
            return

        if not self.enable_learning:
            self._initialized = True
            return

        try:
            from t4dm.learning.collector import get_collector

            self._collector = get_collector()
            self._initialized = True
            logger.info(f"WWObserver initialized for session {self.session_id}")

        except Exception as e:
            logger.warning(f"WWObserver learning disabled: {e}")
            self.enable_learning = False
            self._initialized = True

    def on_event(self, event: Event) -> None:
        """
        Handle an observable event.

        Converts ccapi events to WW learning signals.

        Args:
            event: The event to handle
        """
        self._event_buffer.append(event)

        # Process specific event types
        event_type = getattr(event, "type", None)
        if hasattr(event_type, "value"):
            event_type = event_type.value

        if event_type == "agent.end":
            self._handle_agent_end(event)
        elif event_type == "tool.end":
            self._handle_tool_end(event)
        elif event_type == "tool.error":
            self._handle_tool_error(event)
        elif event_type == "agent.error":
            self._handle_agent_error(event)
        elif event_type == "memory.retrieve":
            self._handle_memory_retrieve(event)

        # Flush if buffer full
        if len(self._event_buffer) >= self.buffer_size:
            self.flush()

    def on_span_start(self, span: Span) -> None:
        """
        Handle span start.

        Tracks active spans for context correlation.

        Args:
            span: The started span
        """
        self._active_spans[span.span_id] = span

    def on_span_end(self, span: Span) -> None:
        """
        Handle span end.

        Triggers outcome processing for agent spans.

        Args:
            span: The ended span
        """
        if span.span_id in self._active_spans:
            del self._active_spans[span.span_id]

        # Process agent spans as outcomes
        if span.name.startswith("agent"):
            self._process_agent_outcome(span)

    def _handle_agent_end(self, event: Event) -> None:
        """Process agent end event as outcome."""
        try:
            # Determine outcome type from event data
            status = event.data.get("status", "ok")
            event.data.get("result", "")

            if status == "ok":
                outcome_type = "success"
                success_score = 1.0
            elif status == "partial":
                outcome_type = "partial"
                success_score = 0.5
            else:
                outcome_type = "failure"
                success_score = 0.0

            # Check for explicit user feedback
            user_feedback = event.data.get("user_feedback")
            if user_feedback == "positive":
                success_score = min(1.0, success_score + 0.2)
            elif user_feedback == "negative":
                success_score = max(0.0, success_score - 0.3)

            # Record outcome (handle both sync and async contexts)
            if self.enable_learning:
                self._schedule_async(self._record_outcome(
                    outcome_type=outcome_type,
                    success_score=success_score,
                    context=f"agent:{event.name}",
                    tool_results=event.data,
                ))

        except Exception as e:
            logger.warning(f"Failed to handle agent end: {e}")

    def _handle_tool_end(self, event: Event) -> None:
        """Process tool end event for procedural learning."""
        try:
            tool_name = event.data.get("tool", event.name)
            tool_result = event.data.get("result")
            duration = event.duration_ms or 0

            # Emit tool usage signal for procedural memory
            if self.enable_learning:
                self._schedule_async(self._record_tool_usage(
                    tool_name=tool_name,
                    success=True,
                    duration_ms=duration,
                    result_summary=str(tool_result)[:200] if tool_result else "",
                ))

        except Exception as e:
            logger.debug(f"Failed to handle tool end: {e}")

    def _handle_tool_error(self, event: Event) -> None:
        """Process tool error event."""
        try:
            tool_name = event.data.get("tool", event.name)
            error = event.data.get("error", "unknown error")

            if self.enable_learning:
                self._schedule_async(self._record_tool_usage(
                    tool_name=tool_name,
                    success=False,
                    error=str(error),
                ))

        except Exception as e:
            logger.debug(f"Failed to handle tool error: {e}")

    def _handle_agent_error(self, event: Event) -> None:
        """Process agent error event as failure outcome."""
        try:
            error = event.data.get("error", "unknown error")

            if self.enable_learning:
                self._schedule_async(self._record_outcome(
                    outcome_type="failure",
                    success_score=0.0,
                    context=f"agent_error:{event.name}",
                    tool_results={"error": str(error)},
                ))

        except Exception as e:
            logger.warning(f"Failed to handle agent error: {e}")

    def _handle_memory_retrieve(self, event: Event) -> None:
        """Track memory retrieval for credit assignment."""
        try:
            retrieval_info = {
                "query": event.data.get("query", ""),
                "results": event.data.get("results", []),
                "memory_type": event.data.get("memory_type", "episodic"),
                "timestamp": datetime.now(),
            }

            self._recent_retrievals.append(retrieval_info)

            # Trim old retrievals
            if len(self._recent_retrievals) > self._max_retrievals:
                self._recent_retrievals = self._recent_retrievals[-self._max_retrievals:]

        except Exception as e:
            logger.debug(f"Failed to handle memory retrieve: {e}")

    def _process_agent_outcome(self, span: Span) -> None:
        """Process completed agent span as outcome."""
        try:
            status = span.status
            attributes = span.attributes

            if status == "ok":
                outcome_type = "success"
                success_score = 1.0
            elif status == "error":
                outcome_type = "failure"
                success_score = 0.0
            else:
                outcome_type = "neutral"
                success_score = 0.5

            if self.enable_learning:
                self._schedule_async(self._record_outcome(
                    outcome_type=outcome_type,
                    success_score=success_score,
                    context=f"span:{span.name}",
                    tool_results=attributes,
                ))

        except Exception as e:
            logger.warning(f"Failed to process agent outcome: {e}")

    async def _record_outcome(
        self,
        outcome_type: str,
        success_score: float,
        context: str,
        tool_results: dict[str, Any],
    ) -> None:
        """Record outcome to WW learning system."""
        await self._ensure_initialized()

        if not self._collector:
            return

        try:
            from t4dm.learning.events import FeedbackSignal, OutcomeType

            # Map outcome type
            type_map = {
                "success": OutcomeType.SUCCESS,
                "partial": OutcomeType.PARTIAL,
                "failure": OutcomeType.FAILURE,
                "neutral": OutcomeType.NEUTRAL,
            }
            ww_outcome_type = type_map.get(outcome_type, OutcomeType.UNKNOWN)

            # Detect feedback signals
            feedback_signals = []
            if success_score >= 0.8:
                feedback_signals.append(FeedbackSignal.ACCEPT)
            elif success_score <= 0.2:
                feedback_signals.append(FeedbackSignal.REJECT)

            # Extract explicit citations from tool results
            citations = self._extract_citations(tool_results)

            # Record via collector
            self._collector.record_outcome(
                outcome_type=ww_outcome_type,
                success_score=success_score,
                context=context,
                explicit_citations=citations,
                feedback_signals=feedback_signals,
                session_id=self.session_id,
                tool_results=tool_results,
            )

            logger.debug(
                f"Recorded outcome: type={outcome_type}, "
                f"score={success_score:.2f}, context={context}"
            )

        except Exception as e:
            logger.warning(f"Failed to record outcome: {e}")

    async def _record_tool_usage(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float = 0,
        result_summary: str = "",
        error: str = "",
    ) -> None:
        """Record tool usage for procedural learning."""
        await self._ensure_initialized()

        if not self._collector:
            return

        try:
            # Tool usage becomes procedural memory signal
            # This feeds into skill acquisition/refinement

            from t4dm.learning.events import OutcomeType

            outcome = OutcomeType.SUCCESS if success else OutcomeType.FAILURE
            score = 1.0 if success else 0.0

            self._collector.record_outcome(
                outcome_type=outcome,
                success_score=score,
                context=f"tool:{tool_name}",
                session_id=self.session_id,
                tool_results={
                    "tool": tool_name,
                    "success": success,
                    "duration_ms": duration_ms,
                    "result_summary": result_summary,
                    "error": error,
                },
            )

            logger.debug(
                f"Recorded tool usage: {tool_name}, "
                f"success={success}, duration={duration_ms}ms"
            )

        except Exception as e:
            logger.debug(f"Failed to record tool usage: {e}")

    def _extract_citations(self, tool_results: dict[str, Any]) -> list[UUID]:
        """Extract memory citations from tool results."""
        citations = []

        try:
            # Look for explicit memory references
            refs = tool_results.get("memory_references", [])
            if isinstance(refs, list):
                for ref in refs:
                    if isinstance(ref, str):
                        try:
                            citations.append(UUID(ref))
                        except ValueError:
                            pass
                    elif isinstance(ref, UUID):
                        citations.append(ref)

            # Look for used_memories field
            used = tool_results.get("used_memories", [])
            if isinstance(used, list):
                for mem_id in used:
                    try:
                        citations.append(UUID(str(mem_id)))
                    except ValueError:
                        pass

        except Exception as e:
            logger.debug(f"Failed to extract citations: {e}")

        return citations

    def flush(self) -> None:
        """Flush buffered events and pending async tasks."""
        # Run any pending async tasks first
        self._run_pending_tasks()

        # Clear buffer
        self._event_buffer.clear()

        # Flush collector if available
        if self._collector and hasattr(self._collector, "flush"):
            try:
                self._collector.flush()
            except Exception as e:
                logger.debug(f"Failed to flush collector: {e}")

        logger.debug(f"WWObserver flushed for session {self.session_id}")

    def close(self) -> None:
        """Close the observer and release resources."""
        self.flush()

        self._active_spans.clear()
        self._recent_retrievals.clear()

        logger.info(f"WWObserver closed for session {self.session_id}")

    def get_recent_retrievals(self) -> list[dict[str, Any]]:
        """Get recent memory retrievals for context."""
        return self._recent_retrievals.copy()

    def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "project": self.project,
            "learning_enabled": self.enable_learning,
            "buffered_events": len(self._event_buffer),
            "active_spans": len(self._active_spans),
            "recent_retrievals": len(self._recent_retrievals),
            "initialized": self._initialized,
        }


# Factory function for convenience
def create_ww_observer(
    session_id: str = "default",
    **kwargs,
) -> WWObserver:
    """
    Create a WWObserver instance.

    Args:
        session_id: Session identifier
        **kwargs: Additional configuration

    Returns:
        WWObserver instance
    """
    return WWObserver(session_id=session_id, **kwargs)


__all__ = ["Event", "EventType", "Span", "WWObserver", "create_ww_observer"]
