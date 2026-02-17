import pytest

from app.modules.intelligence.tools.impact_analysis.impact_analysis_config import (
    canonicalize_identifier,
    expand_identifier_variants,
    is_allowed_xml_path,
    normalize_repo_relative_path,
)


@pytest.mark.parametrize(
    "absolute_path",
    [
        "/Users/someone/repo/src/file.cs",
        "C:/repo/src/file.cs",
        "~/repo/src/file.cs",
    ],
)
def test_normalize_repo_relative_path_rejects_absolute_paths(absolute_path: str):
    with pytest.raises(ValueError, match="Absolute paths are not allowed"):
        normalize_repo_relative_path(absolute_path)


def test_normalize_repo_relative_path_normalizes_relative_input():
    assert normalize_repo_relative_path("./src\\module\\file.cs") == "src/module/file.cs"


@pytest.mark.parametrize(
    "identifier",
    [
        "automation_ID",
        "automationID",
        "AutomationId",
        "AutomationProperties.AutomationId",
    ],
)
def test_automation_identifier_variants_have_same_canonical(identifier: str):
    assert canonicalize_identifier(identifier) == "automationid"


@pytest.mark.parametrize(
    "identifier",
    [
        "Name",
        "AutomationProperties.Name",
    ],
)
def test_name_identifier_variants_have_same_canonical(identifier: str):
    assert canonicalize_identifier(identifier) == "name"


@pytest.mark.parametrize(
    "identifier, expected",
    [
        ("ControlName", "controlname"),
        ("Accessibility", "accessibility"),
        ("automationIdentifiers", "automationidentifiers"),
    ],
)
def test_primary_display_identifier_variants_have_expected_canonical(
    identifier: str, expected: str
):
    assert canonicalize_identifier(identifier) == expected


def test_identifier_variant_expansion_preserves_original_value():
    variants = expand_identifier_variants("AutomationProperties.AutomationId")
    assert variants[0] == "AutomationProperties.AutomationId"
    assert "automationID" in variants
    assert "AutomationId" in variants


def test_primary_display_identifier_variant_expansion():
    control_variants = expand_identifier_variants("ControlName")
    accessibility_variants = expand_identifier_variants("Accessibility")
    automation_identifiers_variants = expand_identifier_variants("automationIdentifiers")

    assert "ControlName" in control_variants
    assert "controlName" in control_variants
    assert "Accessibility" in accessibility_variants
    assert "automationIdentifier" in automation_identifiers_variants


def test_xml_scope_allowlist_enforced():
    assert is_allowed_xml_path(
        "TestCode/FlaUITaskLayer/PrimaryDisplayUI/PrimaryDisplayControls.xml"
    )
    assert is_allowed_xml_path(
        "TestCode/FlaUITaskLayer/PrimaryDisplayUI/Subdir/Any.xml"
    )
    assert not is_allowed_xml_path("_Release/UDD/Desktop/Source/PrimeDisp/View/MainView.xml")
