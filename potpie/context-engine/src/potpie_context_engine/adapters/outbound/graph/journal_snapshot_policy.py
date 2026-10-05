"""Legacy snapshot v2 cannot encode parallel stored claim incarnations."""

from potpie_context_core.graph_journal import JournalError


def require_snapshot_identity(claim_keys):
    keys = [key for key in claim_keys if key]
    if len(keys) != len(set(keys)):
        raise JournalError(
            "snapshot export is unavailable: format v2 cannot preserve parallel claim incarnations; retain the native journal"
        )
