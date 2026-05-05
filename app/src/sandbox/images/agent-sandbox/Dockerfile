# potpie/agent-sandbox — the runtime image used inside Daytona sandboxes
# (and the Docker runtime backend for parity).
#
# Goals:
#   * Carry every tool the sandbox client + agent tools dispatch to:
#       git, ripgrep, fd, jq, python3, node, gh, less, ca-certificates.
#   * Carry the in-sandbox parser (`potpie-parse`) so parsing runs inside
#     the workspace instead of on the host filesystem. The runtime needs
#     `parsing_rs` (a PyO3 wheel built from `app/src/parsing/`) and the
#     ``parser_runner`` shim that streams NDJSON to stdout.
#   * Sized for cold-start: trim apt cache, no Node dev deps, no compilers
#     in the final stage (rust toolchain stays in the builder stage).
#   * Non-root `agent` user with a persistent $HOME at /home/agent so worktrees
#     under /home/agent/work survive Daytona auto-stop / restart.
#
# Build context expectations (all staged by the build driver
# `scripts/build_agent_snapshot.py`; gitignored because they're
# regenerated on every build):
#   * `parsing_src/` — a copy of `app/src/parsing/` (the parsing_rs Rust crate).
#   * `sandbox_src/` — a copy of the `potpie-sandbox` package. The runner
#     lives at `sandbox/parser_runner/` inside it; we install the whole
#     package so the Dockerfile gets `potpie-parse` via the package's
#     `[project.scripts]` entry rather than a hand-rolled shim.
#
# Tag: potpie/agent-sandbox:<version>
# Built either via `docker build -t potpie/agent-sandbox:<v> .` (then push to
# a registry the runtime can reach) or via the Daytona SDK using
# `Image.from_dockerfile(...)` — see scripts/build_agent_snapshot.py.

# ---------------------------------------------------------------------------
# Stage 1 — build the parsing_rs wheel.
# Kept in its own stage so the rust toolchain (~hundreds of MB) doesn't
# bloat the runtime image. Output: `/wheels/parsing-*.whl`.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS parser-builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain via rustup (more reliable than apt for matching the
# pyo3-required toolchain).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && rustc --version

RUN pip install --no-cache-dir maturin==1.7.4

WORKDIR /build
COPY parsing_src/ /build/parsing/
WORKDIR /build/parsing

# `--release` because the runner is invoked once per parse and we want
# the rust speed; `--strip` to keep the wheel small. Output goes to
# /wheels which the runtime stage copies in.
RUN maturin build --release --strip --out /wheels \
    && ls -la /wheels

# ---------------------------------------------------------------------------
# Stage 2 — the runtime image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_MAJOR=20

# Tools the sandbox client and agent tools dispatch to. ripgrep is what
# `client.search` calls; fd is for cheap file walks; gh for PR creation;
# git-lfs because some of the repos potpie indexes use it; tini so node
# / long-running execs don't orphan zombies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        less \
        git \
        git-lfs \
        ripgrep \
        fd-find \
        jq \
        tini \
        tree \
        procps \
        openssh-client \
        unzip \
        xz-utils \
        gnupg \
    && git lfs install --skip-repo \
    && rm -rf /var/lib/apt/lists/*

# Node LTS — the official NodeSource setup script is the path of least
# resistance and gives us the same distribution as upstream.
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm config set update-notifier false \
    && npm cache clean --force

# GitHub CLI — useful for `sandbox_open_pr` and similar agent flows.
RUN install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | gpg --dearmor -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# fd ships as `fdfind` on Debian to avoid a name clash; symlink so agents
# can call `fd` like everywhere else.
RUN ln -s /usr/bin/fdfind /usr/local/bin/fd

# Python tooling — uv for fast venv installs in test runs, plus pip stays.
RUN pip install --no-cache-dir uv

# In-sandbox parser: install the parsing_rs wheel built in stage 1 and
# pip-install the potpie-sandbox package. The package's
# `[project.scripts] potpie-parse` entry installs the runner on PATH;
# the host calls `potpie-parse <repo_dir>` via `client.exec` and
# consumes the NDJSON stream emitted on stdout.
COPY --from=parser-builder /wheels/ /tmp/wheels/
COPY sandbox_src/ /tmp/sandbox_src/
RUN pip install --no-cache-dir /tmp/wheels/parsing-*.whl \
    && pip install --no-cache-dir /tmp/sandbox_src/ \
    && rm -rf /tmp/wheels /tmp/sandbox_src \
    && which potpie-parse \
    && python -c "import parsing_rs; print('parsing_rs version:', getattr(parsing_rs, '__version__', 'unknown'))"

# Non-root user. Most agent operations don't need root; running as `agent`
# also means accidental `rm -rf /` style mistakes can't trash the image.
RUN groupadd --system --gid 1001 agent \
    && useradd --system --uid 1001 --gid 1001 --create-home --shell /bin/bash agent \
    && mkdir -p /home/agent/work \
    && chown -R agent:agent /home/agent

# Default git config so commits work without per-call --git-config flags.
# Agent tools override author when needed via GIT_AUTHOR_NAME / EMAIL env.
RUN git config --system user.name  "Potpie Agent" \
    && git config --system user.email "agent@potpie.local" \
    && git config --system init.defaultBranch main \
    && git config --system pull.rebase false \
    && git config --system safe.directory '*'

USER agent
WORKDIR /home/agent/work

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-l"]
