from __future__ import annotations

import subprocess
from pathlib import Path

from potpie.cli.repo_location import (
    REPO_MATCH_CONTAINED,
    REPO_MATCH_SELF,
    classify_repo_source_match,
    current_repo_identity,
    normalize_repo_ref,
    repo_identity_key,
    repo_identity_key_for_location,
)


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


# --- routing identity: one key for the write and the read --------------------


def _git_tree(root: Path, remote: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    subprocess.run(  # noqa: S603, S607
        ["git", "-C", str(root), "remote", "add", "origin", remote], check=True
    )
    return root


def test_path_registered_repo_is_keyed_by_the_remote_routing_uses(tmp_path) -> None:
    """`--default` on a path wrote a key no repo-scoped command ever read.

    ``repo_identity_key`` keys a path by the path; routing keys the working tree
    by its git remote. Registering ``/path/to/shop`` therefore bound the pot
    under the path while every later command looked it up under
    ``github.com/acme/shop``.
    """
    tree = _git_tree(tmp_path / "shop", "git@github.com:acme/shop.git")

    assert repo_identity_key(str(tree)) == str(tree.resolve())
    assert repo_identity_key_for_location(str(tree)) == "github.com/acme/shop"
    assert repo_identity_key_for_location(str(tree)) == current_repo_identity(tree)


def test_location_without_a_remote_keeps_the_path_key(tmp_path) -> None:
    plain = tmp_path / "notes"
    plain.mkdir()

    assert repo_identity_key_for_location(str(plain)) == str(plain.resolve())


def test_owner_repo_ref_is_never_probed_as_a_path(tmp_path, monkeypatch) -> None:
    """A directory that happens to be named like a remote ref is not one."""
    _git_tree(tmp_path / "acme" / "shop", "git@github.com:other/elsewhere.git")
    monkeypatch.chdir(tmp_path)

    assert repo_identity_key_for_location("acme/shop") == "acme/shop"


# --- workspace root vs project ------------------------------------------------


def test_cwd_inside_the_registered_project_is_a_self_match(tmp_path) -> None:
    project = tmp_path / "alpha"
    (project / "src").mkdir(parents=True)

    assert (
        classify_repo_source_match(str(project), cwd=project, remote=None)
        == REPO_MATCH_SELF
    )
    assert (
        classify_repo_source_match(str(project), cwd=project / "src", remote=None)
        == REPO_MATCH_SELF
    )


def test_registered_project_under_the_cwd_is_only_contained(tmp_path) -> None:
    """Standing in a workspace root is not standing in its child project.

    Collapsed into one boolean, ``~/work`` with ``~/work/alpha`` registered
    scoped every command to alpha's pot without saying so.
    """
    workspace = tmp_path / "work"
    (workspace / "alpha").mkdir(parents=True)

    assert (
        classify_repo_source_match(str(workspace / "alpha"), cwd=workspace, remote=None)
        == REPO_MATCH_CONTAINED
    )


def test_remote_match_is_a_self_match(tmp_path) -> None:
    assert (
        classify_repo_source_match(
            "git@github.com:acme/shop.git",
            cwd=tmp_path,
            remote="github.com/acme/shop",
        )
        == REPO_MATCH_SELF
    )


def test_unrelated_ref_matches_nothing(tmp_path) -> None:
    assert (
        classify_repo_source_match(
            "github.com/other/thing", cwd=tmp_path, remote="github.com/acme/shop"
        )
        is None
    )
