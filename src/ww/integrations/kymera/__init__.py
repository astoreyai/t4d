"""
Kymera Voice Integration for World Weaver.

Connects World Weaver's tripartite memory system to Kymera Voice
for a complete AI assistant with persistent memory.

Quick Start:
    # In kymera-voice jarvis.py:
    from ww.integrations.kymera import JarvisMemoryHook

    # Create hook
    hook = JarvisMemoryHook.create_with_client(ww_client)

    # Before Claude call:
    context = await hook.enhance_context(user_text)
    system_prompt = get_enhanced_system_prompt(base_prompt, context)

    # After Claude response:
    await hook.on_response(user_text, response_text, was_action=True)
"""

from ww.integrations.kymera.action_router import (
    VoiceActionResult,
    VoiceActionRouter,
)
from ww.integrations.kymera.advanced_features import (
    ContentType,
    ContextItem,
    MultiModalContext,
    Notification,
    NotificationManager,
    NotificationPriority,
    NotificationType,
    PreferenceLearner,
    UserPreference,
    VoiceTrigger,
    VoiceTriggerManager,
)
from ww.integrations.kymera.bridge import (
    MemoryContext,
    VoiceContext,
    VoiceMemoryBridge,
)
from ww.integrations.kymera.context_injector import (
    ContextInjector,
    ConversationContextManager,
    InjectionConfig,
)
from ww.integrations.kymera.intent_parser import (
    ParsedIntent,
    TimeParser,
    VoiceIntentParser,
)
from ww.integrations.kymera.jarvis_hook import (
    VOICE_SYSTEM_PROMPT_ADDITION,
    EnhancedContext,
    JarvisMemoryHook,
    get_enhanced_system_prompt,
)
from ww.integrations.kymera.memory_continuity import (
    Conversation,
    ConversationCapture,
    ConversationSummarizer,
    ConversationTurn,
    MemoryConsolidator,
    ProactiveContext,
)
from ww.integrations.kymera.personal_data import (
    ContactCache,
    PersonalDataConfig,
    PersonalDataManager,
)
from ww.integrations.kymera.voice_actions import (
    JarvisIntegration,
    VoiceActionExecutor,
    VoiceExecutionConfig,
    VoiceResponse,
    create_voice_executor,
)

__all__ = [
    # Bridge
    "VoiceMemoryBridge",
    "VoiceContext",
    "MemoryContext",
    # Context Injection
    "ContextInjector",
    "ConversationContextManager",
    "InjectionConfig",
    # Intent Parsing
    "VoiceIntentParser",
    "ParsedIntent",
    "TimeParser",
    # Action Routing
    "VoiceActionRouter",
    "VoiceActionResult",
    # Main Executor
    "VoiceActionExecutor",
    "VoiceResponse",
    "VoiceExecutionConfig",
    "JarvisIntegration",
    "create_voice_executor",
    # Jarvis Hook (primary integration point)
    "JarvisMemoryHook",
    "EnhancedContext",
    "get_enhanced_system_prompt",
    "VOICE_SYSTEM_PROMPT_ADDITION",
    # Personal Data
    "PersonalDataManager",
    "PersonalDataConfig",
    "ContactCache",
    # Memory Continuity
    "ConversationCapture",
    "Conversation",
    "ConversationTurn",
    "ProactiveContext",
    "ConversationSummarizer",
    "MemoryConsolidator",
    # Advanced Features
    "NotificationManager",
    "Notification",
    "NotificationPriority",
    "NotificationType",
    "PreferenceLearner",
    "UserPreference",
    "MultiModalContext",
    "ContextItem",
    "ContentType",
    "VoiceTriggerManager",
    "VoiceTrigger",
]
