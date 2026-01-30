"""
Voice Action Executor - Main entry point for Kymera Voice + World Weaver integration.

Provides a unified interface for processing voice commands through:
- Intent parsing
- Memory retrieval
- Action routing
- Response generation
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from ww.core.actions import ActionRegistry
from ww.integrations.kymera.action_router import VoiceActionRouter
from ww.integrations.kymera.bridge import MemoryContext, VoiceContext, VoiceMemoryBridge
from ww.integrations.kymera.context_injector import ContextInjector, ConversationContextManager
from ww.integrations.kymera.intent_parser import VoiceIntentParser

logger = logging.getLogger(__name__)


@dataclass
class VoiceExecutionConfig:
    """Configuration for voice execution."""
    enable_memory: bool = True
    enable_personal_data: bool = True
    enable_proactive_context: bool = True
    store_conversations: bool = True
    use_llm_parsing: bool = True
    max_context_chars: int = 2000


@dataclass
class VoiceResponse:
    """Complete response from voice execution."""
    spoken_text: str
    full_text: str | None = None
    action_name: str | None = None
    success: bool = True
    data: Any = None
    context_used: MemoryContext | None = None
    proactive_message: str | None = None
    requires_confirmation: bool = False
    conversation_turn: int = 0


class VoiceActionExecutor:
    """
    Main executor for voice commands with World Weaver integration.

    This is the primary integration point between Kymera Voice and World Weaver.

    Usage:
        executor = VoiceActionExecutor.create(ww_client, claude_client)

        # Start conversation
        context = executor.start_conversation(session_id="abc123")

        # Process voice input
        response = await executor.process("remind me to call mom tomorrow")
        logger.info(f"Response: {response.spoken_text}")  # "I'll remind you to call mom tomorrow."

        # End conversation
        await executor.end_conversation()
    """

    def __init__(
        self,
        parser: VoiceIntentParser,
        router: VoiceActionRouter,
        memory_bridge: VoiceMemoryBridge,
        context_injector: ContextInjector,
        config: VoiceExecutionConfig | None = None,
    ):
        """
        Initialize voice action executor.

        Use VoiceActionExecutor.create() factory method for easier setup.
        """
        self.parser = parser
        self.router = router
        self.memory = memory_bridge
        self.injector = context_injector
        self.config = config or VoiceExecutionConfig()

        # Conversation state
        self._current_context: VoiceContext | None = None
        self._conversation_manager: ConversationContextManager | None = None
        self._turn_count = 0
        self._conversation_history: list[tuple[str, str]] = []

        logger.info("VoiceActionExecutor initialized")

    @classmethod
    def create(
        cls,
        ww_client: Any,
        claude_client: Any,
        config: VoiceExecutionConfig | None = None,
    ) -> "VoiceActionExecutor":
        """
        Factory method to create fully configured executor.

        Args:
            ww_client: World Weaver MCP client
            claude_client: Claude API client
            config: Optional configuration

        Returns:
            Configured VoiceActionExecutor
        """
        # Create components
        memory_bridge = VoiceMemoryBridge(ww_client)
        registry = ActionRegistry()

        parser = VoiceIntentParser(llm_client=claude_client)
        router = VoiceActionRouter(
            registry=registry,
            memory_bridge=memory_bridge,
            mcp_client=ww_client,
            claude_client=claude_client,
        )
        context_injector = ContextInjector(memory_bridge)

        return cls(
            parser=parser,
            router=router,
            memory_bridge=memory_bridge,
            context_injector=context_injector,
            config=config,
        )

    def start_conversation(
        self,
        session_id: str | None = None,
        project: str | None = None,
        cwd: str | None = None,
    ) -> VoiceContext:
        """
        Start a new voice conversation.

        Args:
            session_id: Optional session identifier
            project: Optional project context
            cwd: Optional current working directory

        Returns:
            VoiceContext for this conversation
        """
        conversation_id = str(uuid4())

        self._current_context = VoiceContext(
            session_id=session_id or str(uuid4()),
            project=project,
            cwd=cwd,
            conversation_id=conversation_id,
            turn_number=0,
        )

        self._conversation_manager = ConversationContextManager(self.injector)
        self._turn_count = 0
        self._conversation_history = []

        logger.info(f"Started conversation {conversation_id}")
        return self._current_context

    async def process(
        self,
        text: str,
        context: VoiceContext | None = None,
    ) -> VoiceResponse:
        """
        Process voice input and return response.

        Args:
            text: Transcribed voice input
            context: Optional context override

        Returns:
            VoiceResponse with spoken text and metadata
        """
        ctx = context or self._current_context
        if not ctx:
            ctx = self.start_conversation()

        # Increment turn
        self._turn_count += 1
        ctx.turn_number = self._turn_count

        # Parse intent
        if self.config.use_llm_parsing:
            intent = await self.parser.parse_with_llm(text)
        else:
            intent = self.parser.parse(text)

        logger.debug(f"Parsed intent: {intent.action_name} (confidence: {intent.confidence})")

        # Convert to action request
        request = self.parser.to_action_request(intent, ctx.session_id)
        request.user_utterance = text

        # Store user speech in memory
        if self.config.store_conversations:
            await self.memory.on_user_speech(text, ctx)

        # Route and execute
        result = await self.router.route(request, ctx)

        # Store in conversation history
        self._conversation_history.append((text, result.spoken_response))

        # Get proactive context for next turn
        proactive = None
        if self.config.enable_proactive_context and self._turn_count == 1:
            proactive = await self.injector.get_proactive_context(ctx)

        # Build response
        response = VoiceResponse(
            spoken_text=result.spoken_response,
            full_text=result.spoken_response,  # Could differ for complex responses
            action_name=intent.action_name,
            success=result.success,
            data=result.data,
            requires_confirmation=result.requires_confirmation,
            proactive_message=proactive,
            conversation_turn=self._turn_count,
        )

        # Store assistant response if action was taken
        if self.config.store_conversations and result.success:
            await self.memory.on_assistant_response(
                result.spoken_response,
                result.spoken_response,
                ctx,
                was_action=(intent.action_name != "claude.chat"),
            )

        return response

    async def process_batch(
        self,
        texts: list[str],
        context: VoiceContext | None = None,
    ) -> list[VoiceResponse]:
        """
        Process multiple voice inputs in sequence.

        Args:
            texts: List of transcribed voice inputs
            context: Optional context

        Returns:
            List of VoiceResponses
        """
        responses = []
        for text in texts:
            response = await self.process(text, context)
            responses.append(response)
        return responses

    async def get_greeting(self, context: VoiceContext | None = None) -> str:
        """
        Get contextual greeting with proactive information.

        Args:
            context: Optional context

        Returns:
            Greeting string with relevant context
        """
        ctx = context or self._current_context
        if not ctx:
            ctx = self.start_conversation()

        # Get time-appropriate greeting
        hour = datetime.now().hour
        if hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        else:
            greeting = "Good evening"

        # Add proactive context
        proactive = await self.injector.get_proactive_context(ctx)

        if proactive:
            return f"{greeting}. {proactive}"

        return f"{greeting}. How can I help you?"

    async def end_conversation(
        self,
        generate_summary: bool = True,
    ) -> str | None:
        """
        End current conversation and store summary.

        Args:
            generate_summary: Whether to generate and store summary

        Returns:
            Conversation summary if generated
        """
        if not self._current_context:
            return None

        summary = None

        if generate_summary and self._conversation_history:
            # Generate summary from history
            summary = self._summarize_conversation()

            # Store summary
            await self.memory.on_conversation_end(
                self._current_context,
                summary=summary,
            )

        # Reset state
        if self._conversation_manager:
            self._conversation_manager.reset()

        self._current_context = None
        self._turn_count = 0
        self._conversation_history = []

        logger.info("Ended conversation")
        return summary

    def _summarize_conversation(self) -> str:
        """Generate simple summary of conversation."""
        if not self._conversation_history:
            return ""

        turns = len(self._conversation_history)
        topics = []

        # Extract key topics from user utterances
        for user_text, _ in self._conversation_history:
            # Simple keyword extraction
            lower = user_text.lower()
            if "email" in lower:
                topics.append("email")
            elif "calendar" in lower or "meeting" in lower or "schedule" in lower:
                topics.append("calendar")
            elif "remind" in lower:
                topics.append("reminders")
            elif "remember" in lower:
                topics.append("memory")
            elif "task" in lower or "todo" in lower:
                topics.append("tasks")

        unique_topics = list(dict.fromkeys(topics))

        if unique_topics:
            return f"Voice conversation ({turns} turns). Topics: {', '.join(unique_topics)}"

        return f"Voice conversation ({turns} turns)"

    @property
    def current_context(self) -> VoiceContext | None:
        """Get current conversation context."""
        return self._current_context

    @property
    def turn_count(self) -> int:
        """Get current turn count."""
        return self._turn_count


class JarvisIntegration:
    """
    Integration layer for Kymera Voice's Jarvis interface.

    Drop-in integration that can be called from Jarvis's conversation flow.
    """

    def __init__(self, executor: VoiceActionExecutor):
        self.executor = executor
        self._active_sessions: dict[str, VoiceContext] = {}

    async def on_wake_word(self, session_id: str) -> str:
        """
        Called when wake word detected.

        Returns greeting response.
        """
        context = self.executor.start_conversation(session_id=session_id)
        self._active_sessions[session_id] = context

        return await self.executor.get_greeting(context)

    async def on_speech(
        self,
        session_id: str,
        transcription: str,
    ) -> VoiceResponse:
        """
        Called when speech transcribed.

        Args:
            session_id: Session identifier
            transcription: Transcribed speech

        Returns:
            VoiceResponse to speak
        """
        context = self._active_sessions.get(session_id)
        if not context:
            context = self.executor.start_conversation(session_id=session_id)
            self._active_sessions[session_id] = context

        return await self.executor.process(transcription, context)

    async def on_silence(self, session_id: str, duration_seconds: float) -> None:
        """
        Called when silence detected after speech.

        Long silence might indicate end of conversation.
        """
        if duration_seconds > 30:
            await self.on_goodbye(session_id)

    async def on_goodbye(self, session_id: str) -> str | None:
        """
        Called when conversation ends.

        Returns optional farewell message.
        """
        context = self._active_sessions.pop(session_id, None)
        if context:
            self.executor._current_context = context
            await self.executor.end_conversation()

        return "Goodbye!"

    def get_context(self, session_id: str) -> VoiceContext | None:
        """Get context for session."""
        return self._active_sessions.get(session_id)


# Convenience function for quick setup
async def create_voice_executor(
    ww_client: Any,
    claude_client: Any,
) -> VoiceActionExecutor:
    """
    Quick setup for voice executor.

    Args:
        ww_client: World Weaver MCP client
        claude_client: Claude API client

    Returns:
        Ready-to-use VoiceActionExecutor
    """
    return VoiceActionExecutor.create(ww_client, claude_client)
