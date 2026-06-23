from __future__ import annotations

from context_engine.adapters.inbound.cli.repo_location import normalize_repo_ref


def test_normalize_repo_ref_strips_url_credentials() -> None:
    assert (
        normalize_repo_ref("https://user:token@github.com/potpie-ai/potpie.git")
        == "github.com/potpie-ai/potpie"
    )


def test_normalize_repo_ref_keeps_port_without_credentials() -> None:
    assert (
        normalize_repo_ref("https://user:token@git.example.com:8443/acme/repo.git")
        == "git.example.com:8443/acme/repo"
    )
