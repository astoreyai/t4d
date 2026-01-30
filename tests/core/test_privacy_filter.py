"""Tests for privacy filter module."""

import pytest

from ww.core.privacy_filter import (
    SensitivityLevel,
    RedactionResult,
    PrivacyRule,
    PrivacyFilter,
    ContentClassifier,
)


class TestSensitivityLevel:
    """Tests for SensitivityLevel enum."""

    def test_levels_exist(self):
        """Test all sensitivity levels exist."""
        assert SensitivityLevel.PUBLIC.value == "public"
        assert SensitivityLevel.INTERNAL.value == "internal"
        assert SensitivityLevel.CONFIDENTIAL.value == "confidential"
        assert SensitivityLevel.RESTRICTED.value == "restricted"

    def test_level_count(self):
        """Test number of levels."""
        assert len(list(SensitivityLevel)) == 4


class TestRedactionResult:
    """Tests for RedactionResult dataclass."""

    def test_basic_result(self):
        """Test basic result creation."""
        result = RedactionResult(
            original_length=100,
            redacted_length=80,
            content="Filtered content",
            redactions=["SSN: 1 instance(s)"],
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            blocked=False,
        )
        assert result.original_length == 100
        assert result.redacted_length == 80
        assert result.blocked is False

    def test_blocked_result(self):
        """Test blocked result."""
        result = RedactionResult(
            original_length=50,
            redacted_length=0,
            content="",
            redactions=["Content blocked"],
            sensitivity=SensitivityLevel.RESTRICTED,
            blocked=True,
        )
        assert result.blocked is True
        assert result.content == ""


class TestPrivacyRule:
    """Tests for PrivacyRule dataclass."""

    def test_basic_rule(self):
        """Test basic rule creation."""
        import re
        rule = PrivacyRule(
            name="Test",
            pattern=re.compile(r"\d+"),
            replacement="[NUMBER]",
            sensitivity=SensitivityLevel.CONFIDENTIAL,
        )
        assert rule.name == "Test"
        assert rule.enabled is True

    def test_disabled_rule(self):
        """Test disabled rule."""
        import re
        rule = PrivacyRule(
            name="Disabled",
            pattern=re.compile(r"test"),
            replacement="[REDACTED]",
            sensitivity=SensitivityLevel.PUBLIC,
            enabled=False,
        )
        assert rule.enabled is False


class TestPrivacyFilter:
    """Tests for PrivacyFilter class."""

    @pytest.fixture
    def filter(self):
        """Create privacy filter."""
        return PrivacyFilter()

    def test_initialization(self, filter):
        """Test filter initialization."""
        assert len(filter._rules) > 0
        assert filter._private_mode is False

    def test_initialization_with_email_redaction(self):
        """Test initialization with email redaction enabled."""
        filter = PrivacyFilter(redact_emails=True)
        # Should have email rule
        email_rules = [r for r in filter._rules if r.name == 'Email']
        assert len(email_rules) == 1

    def test_initialization_with_custom_patterns(self):
        """Test initialization with custom patterns."""
        filter = PrivacyFilter(custom_patterns=[
            (r'secret\d+', '[SECRET]', 'CustomSecret')
        ])
        custom_rules = [r for r in filter._rules if r.name == 'CustomSecret']
        assert len(custom_rules) == 1

    def test_redact_ssn(self, filter):
        """Test SSN redaction."""
        result = filter.filter("My SSN is 123-45-6789")
        assert "[SSN_REDACTED]" in result.content
        assert "123-45-6789" not in result.content

    def test_redact_credit_card(self, filter):
        """Test credit card redaction."""
        result = filter.filter("Card: 4111 1111 1111 1111")
        assert "[CARD_REDACTED]" in result.content
        assert "4111" not in result.content

    def test_redact_phone(self, filter):
        """Test phone number redaction."""
        result = filter.filter("Call me at (555) 123-4567")
        assert "[PHONE_REDACTED]" in result.content
        assert "555" not in result.content

    def test_redact_internal_ip(self, filter):
        """Test internal IP redaction."""
        result = filter.filter("Server at 192.168.1.100")
        assert "[INTERNAL_IP]" in result.content

    def test_redact_password(self, filter):
        """Test password redaction."""
        result = filter.filter("password: mysecretpassword123")
        assert "[PASSWORD_REDACTED]" in result.content
        assert "mysecretpassword123" not in result.content

    def test_redact_api_key(self, filter):
        """Test API key redaction."""
        result = filter.filter("api_key=sk_test_abcdefghijklmnopqrst12345")
        assert "[API_KEY_REDACTED]" in result.content

    def test_redact_aws_key(self, filter):
        """Test AWS key redaction."""
        result = filter.filter("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert "[AWS_KEY_REDACTED]" in result.content

    def test_redact_github_token(self, filter):
        """Test GitHub token redaction."""
        # ghp_ tokens are exactly 36 chars after the prefix
        result = filter.filter("Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert "[GITHUB_TOKEN_REDACTED]" in result.content

    def test_redact_openai_key(self, filter):
        """Test OpenAI key redaction."""
        # sk- pattern at word boundary (32+ chars)
        result = filter.filter("The key is sk-abcdefghijklmnopqrstuvwxyz123456")
        assert "[OPENAI_KEY_REDACTED]" in result.content

    def test_privacy_command_forget(self, filter):
        """Test forget privacy command."""
        result = filter.filter("forget what I just said")
        assert result.blocked is True
        assert result.content == ""

    def test_privacy_command_dont_store(self, filter):
        """Test don't store command."""
        result = filter.filter("don't store that")
        assert result.blocked is True

    def test_privacy_command_off_record(self, filter):
        """Test off the record command."""
        result = filter.filter("off the record for a moment")
        assert result.blocked is True
        assert filter.is_private_mode() is True

    def test_privacy_command_back_on_record(self, filter):
        """Test back on the record command."""
        filter.set_private_mode(True)
        result = filter.filter("back on the record now")
        assert result.blocked is False
        assert filter.is_private_mode() is False

    def test_private_mode_blocks_content(self, filter):
        """Test private mode blocks all content."""
        filter.set_private_mode(True)
        result = filter.filter("This should be blocked")
        assert result.blocked is True
        assert result.content == ""

    def test_no_redaction_clean_content(self, filter):
        """Test clean content has no redactions."""
        result = filter.filter("Just a normal message without sensitive info")
        assert result.content == "Just a normal message without sensitive info"
        assert len(result.redactions) == 0
        assert result.sensitivity == SensitivityLevel.PUBLIC

    def test_multiple_redactions(self, filter):
        """Test multiple different redactions."""
        result = filter.filter(
            "SSN: 123-45-6789, Phone: (555) 123-4567, Password: secret123"
        )
        assert len(result.redactions) >= 2
        assert "[SSN_REDACTED]" in result.content
        assert "[PHONE_REDACTED]" in result.content

    def test_add_custom_rule(self, filter):
        """Test adding a custom rule."""
        filter.add_rule(
            name="ProjectCode",
            pattern=r"PROJECT-\d{4}",
            replacement="[PROJECT_REDACTED]",
        )
        result = filter.filter("Working on PROJECT-1234")
        assert "[PROJECT_REDACTED]" in result.content

    def test_remove_rule(self, filter):
        """Test removing a rule."""
        filter.add_rule("ToRemove", r"remove_me", "[REMOVED]")
        assert filter.remove_rule("ToRemove") is True
        # Should not find removed rule
        assert filter.remove_rule("ToRemove") is False

    def test_remove_nonexistent_rule(self, filter):
        """Test removing nonexistent rule."""
        assert filter.remove_rule("NonexistentRule") is False

    def test_disable_rule(self, filter):
        """Test disabling a rule."""
        filter.add_rule("TestDisable", r"test", "[TEST]")
        assert filter.disable_rule("TestDisable") is True
        # Rule should be disabled
        result = filter.filter("test content")
        assert "[TEST]" not in result.content

    def test_disable_nonexistent_rule(self, filter):
        """Test disabling nonexistent rule."""
        assert filter.disable_rule("Nonexistent") is False

    def test_enable_rule(self, filter):
        """Test enabling a rule."""
        filter.add_rule("TestEnable", r"enable", "[ENABLED]")
        filter.disable_rule("TestEnable")
        assert filter.enable_rule("TestEnable") is True
        result = filter.filter("enable this")
        assert "[ENABLED]" in result.content

    def test_enable_nonexistent_rule(self, filter):
        """Test enabling nonexistent rule."""
        assert filter.enable_rule("Nonexistent") is False

    def test_sensitivity_threshold(self):
        """Test sensitivity threshold for blocking."""
        filter = PrivacyFilter(sensitivity_threshold=SensitivityLevel.CONFIDENTIAL)
        # Content with restricted sensitivity should be flagged
        result = filter.filter("password: supersecret123")
        # High redaction ratio with high sensitivity may block
        assert result.sensitivity == SensitivityLevel.RESTRICTED

    def test_check_privacy_commands(self, filter):
        """Test privacy command detection."""
        assert filter._check_privacy_commands("forget that") == "FORGET"
        assert filter._check_privacy_commands("off the record") == "START_PRIVATE"
        assert filter._check_privacy_commands("normal text") is None

    def test_is_private_mode(self, filter):
        """Test private mode getter."""
        assert filter.is_private_mode() is False
        filter.set_private_mode(True)
        assert filter.is_private_mode() is True

    def test_set_private_mode(self, filter):
        """Test private mode setter."""
        filter.set_private_mode(True)
        assert filter._private_mode is True
        filter.set_private_mode(False)
        assert filter._private_mode is False

    def test_result_metadata(self, filter):
        """Test result contains correct metadata."""
        content = "My SSN is 123-45-6789 and my password is secret"
        result = filter.filter(content)
        assert result.original_length == len(content)
        assert result.redacted_length > 0
        assert result.redacted_length != result.original_length

    def test_empty_content(self, filter):
        """Test filtering empty content."""
        result = filter.filter("")
        assert result.blocked is False
        assert result.content == ""
        assert result.original_length == 0


class TestContentClassifier:
    """Tests for ContentClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create classifier."""
        return ContentClassifier()

    def test_initialization(self, classifier):
        """Test classifier initialization."""
        assert len(classifier._patterns) > 0

    def test_classify_financial(self, classifier):
        """Test financial content classification."""
        scores = classifier.classify(
            "Need to check my bank account balance and make a payment"
        )
        assert "financial" in scores
        assert scores["financial"] > 0

    def test_classify_health(self, classifier):
        """Test health content classification."""
        scores = classifier.classify(
            "Going to the doctor for my medication prescription"
        )
        assert "health" in scores
        assert scores["health"] > 0

    def test_classify_legal(self, classifier):
        """Test legal content classification."""
        scores = classifier.classify(
            "Discussing the contract with my attorney"
        )
        assert "legal" in scores
        assert scores["legal"] > 0

    def test_classify_personal(self, classifier):
        """Test personal content classification."""
        scores = classifier.classify(
            "Planning the family birthday celebration at home"
        )
        assert "personal" in scores
        assert scores["personal"] > 0

    def test_classify_work(self, classifier):
        """Test work content classification."""
        scores = classifier.classify(
            "Meeting with boss about the project deadline"
        )
        assert "work" in scores
        assert scores["work"] > 0

    def test_classify_neutral(self, classifier):
        """Test neutral content has low scores."""
        scores = classifier.classify(
            "The quick brown fox jumps over the lazy dog"
        )
        # Should have relatively low scores
        assert all(score < 0.5 for score in scores.values())

    def test_classify_empty(self, classifier):
        """Test classifying empty content."""
        scores = classifier.classify("")
        assert all(score == 0 for score in scores.values())

    def test_classify_multiple_categories(self, classifier):
        """Test content matching multiple categories."""
        scores = classifier.classify(
            "The doctor reviewed my salary and tax situation with the attorney"
        )
        # Should match health, work, financial, and legal
        assert scores["health"] > 0
        assert scores["financial"] > 0
        assert scores["legal"] > 0

    def test_scores_normalized(self, classifier):
        """Test scores are normalized between 0 and 1."""
        scores = classifier.classify(
            "bank account balance transfer payment invoice salary income tax"
        )
        assert all(0 <= score <= 1 for score in scores.values())
