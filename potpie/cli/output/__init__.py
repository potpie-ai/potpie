"""Versioned machine-output contracts for the Potpie CLI."""

from potpie.cli.output.contracts import (
    CLI_SCHEMA_VERSION,
    error_envelope,
    success_envelope,
)

__all__ = ["CLI_SCHEMA_VERSION", "error_envelope", "success_envelope"]
