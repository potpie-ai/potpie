import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, List


@dataclass(frozen=True)
class ImpactAnalysisConfig:
    path_mode: str = "relative_only"
    xml_allowlist: tuple[str, ...] = (
        "TestCode/FlaUITaskLayer/PrimaryDisplayUI/PrimaryDisplayControls.xml",
        "TestCode/FlaUITaskLayer/PrimaryDisplayUI/syncFusionDataGrid_Metadata.xml",
        "TestCode/FlaUITaskLayer/PrimaryDisplayUI/**/*.xml",
    )
    reject_xml_outside_scope: bool = True
    case_insensitive_identifier_match: bool = True


IMPACT_ANALYSIS_CONFIG = ImpactAnalysisConfig()


AUTOMATION_ID_VARIANTS: tuple[str, ...] = (
    "automation_ID",
    "automationID",
    "AutomationId",
    "AutomationProperties.AutomationId",
)

NAME_VARIANTS: tuple[str, ...] = (
    "Name",
    "AutomationProperties.Name",
)

CONTROL_NAME_VARIANTS: tuple[str, ...] = (
    "ControlName",
    "controlName",
)

ACCESSIBILITY_VARIANTS: tuple[str, ...] = (
    "Accessibility",
    "accessibility",
)

AUTOMATION_IDENTIFIERS_VARIANTS: tuple[str, ...] = (
    "automationIdentifiers",
    "AutomationIdentifiers",
    "automationIdentifier",
)


_IDENTIFIER_CANONICAL_MAP: Dict[str, str] = {
    "automationid": "automationid",
    "automationpropertiesautomationid": "automationid",
    "name": "name",
    "automationpropertiesname": "name",
    "controlname": "controlname",
    "accessibility": "accessibility",
    "automationidentifiers": "automationidentifiers",
    "automationidentifier": "automationidentifiers",
}


def canonicalize_identifier(identifier: str) -> str:
    """Map common UI identifier spellings to canonical tokens."""
    if not identifier:
        return ""

    normalized = re.sub(r"[^a-zA-Z0-9]", "", identifier).lower()
    return _IDENTIFIER_CANONICAL_MAP.get(normalized, normalized)


def expand_identifier_variants(identifier: str) -> List[str]:
    """Return search variants while preserving the original token for evidence."""
    original = (identifier or "").strip()
    if not original:
        return []

    canonical = canonicalize_identifier(original)
    variants: list[str] = [original]

    if canonical == "automationid":
        variants.extend(AUTOMATION_ID_VARIANTS)
    elif canonical == "name":
        variants.extend(NAME_VARIANTS)
    elif canonical == "controlname":
        variants.extend(CONTROL_NAME_VARIANTS)
    elif canonical == "accessibility":
        variants.extend(ACCESSIBILITY_VARIANTS)
    elif canonical == "automationidentifiers":
        variants.extend(AUTOMATION_IDENTIFIERS_VARIANTS)

    # Preserve order while deduplicating.
    deduped = []
    seen = set()
    for variant in variants:
        if variant not in seen:
            deduped.append(variant)
            seen.add(variant)
    return deduped


def normalize_repo_relative_path(path: str) -> str:
    """Validate and normalize a repository-relative path."""
    raw = (path or "").strip()
    if not raw:
        raise ValueError("Path cannot be empty. Provide a repo-relative path.")

    normalized = raw.replace("\\", "/")

    windows_abs = re.match(r"^[A-Za-z]:/", normalized)
    if os.path.isabs(normalized) or windows_abs or normalized.startswith("~/"):
        raise ValueError(
            "Absolute paths are not allowed. Provide a path relative to the repository root."
        )

    while normalized.startswith("./"):
        normalized = normalized[2:]

    if normalized == ".." or normalized.startswith("../") or "/../" in normalized:
        raise ValueError("Path traversal segments ('..') are not allowed.")

    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts:
        raise ValueError("Path cannot be empty after normalization.")

    return "/".join(parts)


def to_repo_relative_output_path(path: str, project_id: str | None = None) -> str:
    """Best-effort conversion of an arbitrary path to repo-relative output form."""
    if not path:
        return ""

    normalized = path.strip().replace("\\", "/")

    if project_id:
        marker = f"/projects/{project_id}/"
        if marker in normalized:
            return normalized.split(marker, 1)[1].lstrip("/")

    if "/projects/" in normalized:
        tail = normalized.split("/projects/", 1)[1]
        if "/" in tail:
            return tail.split("/", 1)[1].lstrip("/")

    windows_prefix = re.match(r"^[A-Za-z]:/", normalized)
    if windows_prefix:
        normalized = normalized[3:]

    return normalized.lstrip("/")


def is_xml_file(path: str) -> bool:
    return (path or "").lower().endswith(".xml")


def is_allowed_xml_path(path: str) -> bool:
    if not is_xml_file(path):
        return False

    normalized = to_repo_relative_output_path(path)
    posix_path = PurePosixPath(normalized)
    return any(posix_path.match(pattern) for pattern in IMPACT_ANALYSIS_CONFIG.xml_allowlist)
