"""
Privacy Filter - Redacts sensitive information before storage.

Ensures PII, credentials, and user-marked private content
never reaches long-term memory.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """Classification of content sensitivity."""
    PUBLIC = "public"           # Safe to store
    INTERNAL = "internal"       # Store but don't share externally
    CONFIDENTIAL = "confidential"  # Redact specific fields
    RESTRICTED = "restricted"   # Do not store at all


@dataclass
class RedactionResult:
    """Result of privacy filtering."""
    original_length: int
    redacted_length: int
    content: str
    redactions: list[str]  # What was redacted
    sensitivity: SensitivityLevel
    blocked: bool  # True if content should not be stored at all


@dataclass
class PrivacyRule:
    """A privacy rule with pattern and action."""
    name: str
    pattern: re.Pattern
    replacement: str
    sensitivity: SensitivityLevel
    enabled: bool = True


class PrivacyFilter:
    """
    Filters and redacts sensitive information from content.

    Categories:
    - PII: SSN, credit cards, phone numbers, addresses
    - Credentials: Passwords, API keys, tokens
    - Health: Medical information (HIPAA-like)
    - Financial: Bank accounts, financial data
    - User-controlled: "off the record", "private"
    """

    # PII Patterns
    PII_PATTERNS = [
        # SSN
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", "SSN"),
        (r"\b\d{9}\b(?=.*\b(ssn|social)\b)", "[SSN_REDACTED]", "SSN"),

        # Credit Cards (Luhn-valid patterns)
        (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
         "[CARD_REDACTED]", "Credit Card"),
        (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD_REDACTED]", "Credit Card"),

        # Phone Numbers
        (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]", "Phone"),

        # Email (optional - sometimes want to keep)
        # (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', 'Email'),

        # IP Addresses (internal networks)
        (r"\b(?:10|172\.(?:1[6-9]|2[0-9]|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b",
         "[INTERNAL_IP]", "Internal IP"),
    ]

    # Credential Patterns
    CREDENTIAL_PATTERNS = [
        # Passwords in common formats
        (r'(?i)password[:\s=]+["\']?([^\s"\']+)["\']?', "[PASSWORD_REDACTED]", "Password"),
        (r'(?i)passwd[:\s=]+["\']?([^\s"\']+)["\']?', "[PASSWORD_REDACTED]", "Password"),
        (r'(?i)pwd[:\s=]+["\']?([^\s"\']+)["\']?', "[PASSWORD_REDACTED]", "Password"),

        # API Keys
        (r'(?i)api[_-]?key[:\s=]+["\']?([A-Za-z0-9_\-]{20,})["\']?', "[API_KEY_REDACTED]", "API Key"),
        (r'(?i)secret[_-]?key[:\s=]+["\']?([A-Za-z0-9_\-]{20,})["\']?', "[SECRET_REDACTED]", "Secret Key"),
        (r'(?i)access[_-]?token[:\s=]+["\']?([A-Za-z0-9_\-\.]{20,})["\']?', "[TOKEN_REDACTED]", "Access Token"),

        # AWS
        (r"\b(AKIA[0-9A-Z]{16})\b", "[AWS_KEY_REDACTED]", "AWS Access Key"),
        (r'(?i)aws[_-]?secret[:\s=]+["\']?([A-Za-z0-9/+=]{40})["\']?', "[AWS_SECRET_REDACTED]", "AWS Secret"),

        # GitHub
        (r"\b(ghp_[A-Za-z0-9]{36})\b", "[GITHUB_TOKEN_REDACTED]", "GitHub Token"),
        (r"\b(github_pat_[A-Za-z0-9_]{22,})\b", "[GITHUB_PAT_REDACTED]", "GitHub PAT"),

        # Generic tokens
        (r"\b(Bearer\s+[A-Za-z0-9\-._~+/]+=*)\b", "[BEARER_TOKEN_REDACTED]", "Bearer Token"),
        (r"\b(sk-[A-Za-z0-9]{32,})\b", "[OPENAI_KEY_REDACTED]", "OpenAI Key"),
        (r"\b(sk-ant-[A-Za-z0-9\-]{80,})\b", "[ANTHROPIC_KEY_REDACTED]", "Anthropic Key"),
    ]

    # Voice-specific privacy commands
    PRIVACY_COMMANDS = [
        (r"(?i)\boff\s+the\s+record\b", "START_PRIVATE"),
        (r"(?i)\bback\s+on\s+the\s+record\b", "END_PRIVATE"),
        (r"(?i)\bprivate\s*:", "INLINE_PRIVATE"),
        (r"(?i)\bforget\s+(that|this|what\s+i\s+(just\s+)?said)\b", "FORGET"),
        (r"(?i)\bdon\'?t\s+(store|remember|save)\s+(that|this)\b", "FORGET"),
    ]

    def __init__(
        self,
        redact_emails: bool = False,
        redact_names: bool = False,
        custom_patterns: list[tuple[str, str, str]] | None = None,
        sensitivity_threshold: SensitivityLevel = SensitivityLevel.CONFIDENTIAL,
    ):
        """
        Initialize privacy filter.

        Args:
            redact_emails: Whether to redact email addresses
            redact_names: Whether to redact proper names
            custom_patterns: Additional (pattern, replacement, name) tuples
            sensitivity_threshold: Block content at or above this level
        """
        self.redact_emails = redact_emails
        self.redact_names = redact_names
        self.sensitivity_threshold = sensitivity_threshold

        # Build rules
        self._rules: list[PrivacyRule] = []
        self._build_rules(custom_patterns)

        # Privacy mode state
        self._private_mode = False

        logger.info(f"PrivacyFilter initialized with {len(self._rules)} rules")

    def _build_rules(self, custom_patterns: list[tuple[str, str, str]] | None) -> None:
        """Build compiled regex rules."""
        # PII rules
        for pattern, replacement, name in self.PII_PATTERNS:
            self._rules.append(PrivacyRule(
                name=name,
                pattern=re.compile(pattern, re.IGNORECASE),
                replacement=replacement,
                sensitivity=SensitivityLevel.CONFIDENTIAL,
            ))

        # Credential rules (higher sensitivity)
        for pattern, replacement, name in self.CREDENTIAL_PATTERNS:
            self._rules.append(PrivacyRule(
                name=name,
                pattern=re.compile(pattern),
                replacement=replacement,
                sensitivity=SensitivityLevel.RESTRICTED,
            ))

        # Email (optional)
        if self.redact_emails:
            self._rules.append(PrivacyRule(
                name="Email",
                pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
                replacement="[EMAIL_REDACTED]",
                sensitivity=SensitivityLevel.CONFIDENTIAL,
            ))

        # Custom patterns
        if custom_patterns:
            for pattern, replacement, name in custom_patterns:
                self._rules.append(PrivacyRule(
                    name=name,
                    pattern=re.compile(pattern),
                    replacement=replacement,
                    sensitivity=SensitivityLevel.CONFIDENTIAL,
                ))

    def filter(self, content: str) -> RedactionResult:
        """
        Filter content and redact sensitive information.

        Args:
            content: Raw content to filter

        Returns:
            RedactionResult with filtered content and metadata
        """
        original_length = len(content)
        redactions = []
        max_sensitivity = SensitivityLevel.PUBLIC
        blocked = False

        # Check for privacy commands
        command = self._check_privacy_commands(content)
        if command == "FORGET":
            return RedactionResult(
                original_length=original_length,
                redacted_length=0,
                content="",
                redactions=["User requested forget"],
                sensitivity=SensitivityLevel.RESTRICTED,
                blocked=True,
            )
        if command == "START_PRIVATE":
            self._private_mode = True
            return RedactionResult(
                original_length=original_length,
                redacted_length=0,
                content="[Private mode started]",
                redactions=["Private mode enabled"],
                sensitivity=SensitivityLevel.RESTRICTED,
                blocked=True,
            )
        if command == "END_PRIVATE":
            self._private_mode = False
            return RedactionResult(
                original_length=original_length,
                redacted_length=len(content),
                content=content,
                redactions=["Private mode disabled"],
                sensitivity=SensitivityLevel.PUBLIC,
                blocked=False,
            )

        # Block if in private mode
        if self._private_mode:
            return RedactionResult(
                original_length=original_length,
                redacted_length=0,
                content="",
                redactions=["Private mode active"],
                sensitivity=SensitivityLevel.RESTRICTED,
                blocked=True,
            )

        # Apply redaction rules
        filtered = content
        for rule in self._rules:
            if not rule.enabled:
                continue

            matches = rule.pattern.findall(filtered)
            if matches:
                filtered = rule.pattern.sub(rule.replacement, filtered)
                redactions.append(f"{rule.name}: {len(matches)} instance(s)")

                if rule.sensitivity.value > max_sensitivity.value:
                    max_sensitivity = rule.sensitivity

        # Check if should block entirely
        if max_sensitivity.value >= self.sensitivity_threshold.value:
            # For restricted content, check if too much was redacted
            redacted_ratio = 1 - (len(filtered) / original_length) if original_length > 0 else 0
            if redacted_ratio > 0.5:  # More than half redacted
                blocked = True

        return RedactionResult(
            original_length=original_length,
            redacted_length=len(filtered),
            content=filtered,
            redactions=redactions,
            sensitivity=max_sensitivity,
            blocked=blocked,
        )

    def _check_privacy_commands(self, content: str) -> str | None:
        """Check for voice privacy commands."""
        for pattern, command in self.PRIVACY_COMMANDS:
            if re.search(pattern, content):
                return command
        return None

    def is_private_mode(self) -> bool:
        """Check if private mode is active."""
        return self._private_mode

    def set_private_mode(self, enabled: bool) -> None:
        """Manually set private mode."""
        self._private_mode = enabled

    def add_rule(
        self,
        name: str,
        pattern: str,
        replacement: str,
        sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL,
    ) -> None:
        """Add a custom redaction rule."""
        self._rules.append(PrivacyRule(
            name=name,
            pattern=re.compile(pattern),
            replacement=replacement,
            sensitivity=sensitivity,
        ))

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def disable_rule(self, name: str) -> bool:
        """Disable a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = False
                return True
        return False

    def enable_rule(self, name: str) -> bool:
        """Enable a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = True
                return True
        return False


class ContentClassifier:
    """
    Classifies content into sensitivity categories.

    Used for determining storage policies beyond just redaction.
    """

    CATEGORY_PATTERNS = {
        "financial": [
            r"\b(bank|account|balance|transfer|payment|invoice)\b",
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
            r"\b(salary|income|tax|irs)\b",
        ],
        "health": [
            r"\b(doctor|hospital|medication|diagnosis|symptom)\b",
            r"\b(prescription|pharmacy|medical|health)\b",
        ],
        "legal": [
            r"\b(attorney|lawyer|lawsuit|court|legal)\b",
            r"\b(contract|agreement|liability|settlement)\b",
        ],
        "personal": [
            r"\b(birthday|anniversary|family|relationship)\b",
            r"\b(home|house|apartment|address)\b",
        ],
        "work": [
            r"\b(project|deadline|meeting|boss|colleague)\b",
            r"\b(salary|promotion|review|performance)\b",
        ],
    }

    def __init__(self) -> None:
        self._patterns = {
            cat: [re.compile(p, re.IGNORECASE) for p in patterns]
            for cat, patterns in self.CATEGORY_PATTERNS.items()
        }

    def classify(self, content: str) -> dict[str, float]:
        """
        Classify content into categories with confidence scores.

        Returns:
            Dict of category -> confidence (0-1)
        """
        scores = {}
        words = len(content.split())

        for category, patterns in self._patterns.items():
            matches = sum(
                len(p.findall(content))
                for p in patterns
            )
            # Normalize by content length
            scores[category] = min(1.0, matches / max(1, words / 10))

        return scores
