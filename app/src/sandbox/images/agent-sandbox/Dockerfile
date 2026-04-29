# potpie/agent-sandbox — the runtime image used inside Daytona sandboxes
# (and the Docker runtime backend for parity).
#
# Goals:
#   * Carry every tool the sandbox client + agent tools dispatch to:
#       git, ripgrep, fd, jq, python3, node, gh, less, ca-certificates.
#   * Sized for cold-start: trim apt cache, no Node dev deps, no compilers.
#   * Non-root `agent` user with a persistent $HOME at /home/agent so worktrees
#     under /home/agent/work survive Daytona auto-stop / restart.
#
# Tag: potpie/agent-sandbox:<version>
# Built either via `docker build -t potpie/agent-sandbox:<v> .` (then push to
# a registry the runtime can reach) or via the Daytona SDK using
# `Image.from_dockerfile(...)` — see scripts/build_agent_snapshot.py.

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
