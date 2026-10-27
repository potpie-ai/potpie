"""Declared package boundaries and ownership for source contract tests."""

LEGACY_DAEMON_NAMESPACES = {
    "potpie_context_engine.adapters.inbound.daemon_http",
    "potpie_context_engine.adapters.inbound.http.ui",
    "potpie_context_engine.adapters.outbound.daemon_process",
    "potpie_context_engine.adapters.outbound.managed_services",
    "potpie_context_engine.application.services.managed_service_manager",
    "potpie_context_engine.domain.ports.daemon",
    "potpie_context_engine.host.daemon",
    "potpie_context_engine.host.daemon_client",
    "potpie_context_engine.host.daemon_main",
    "potpie_context_engine.host.daemon_rpc",
    "potpie_context_engine.host.daemon_runtime",
}

FORBIDDEN_IDENTIFIERS = frozenset(
    {
        "DaemonRpcClient",
        "HostShell",
        "LegacyEngineClientAdapter",
        "LegacyRequestInvoker",
        "RemoteHostShell",
        "RemoteSurface",
        "build_host_shell",
        "build_legacy_engine_client",
        "get_host",
        "set_host",
    }
)

CANONICAL_RUNTIME_DEFINITIONS = {
    "CanonicalDaemonRuntime": "potpie/runtime/server.py",
    "DaemonController": "potpie/runtime/controller.py",
    "DaemonEngineClient": "potpie/runtime/clients.py",
    "write_daemon_discovery": "potpie/daemon/discovery.py",
}

EXPECTED_CAPABILITY_SYMBOLS = {
    "potpie/agent_context.py": {"AgentContextService"},
    "potpie/auth/ports/identity.py": {"AuthIdentity", "AuthService"},
    "potpie/auth/adapters/local_identity.py": {"LocalAuthService"},
    "potpie/config/contracts.py": {"ConfigService"},
    "potpie/config/local.py": {"LocalConfigService"},
    "potpie/config/local_paths.py": {"default_home"},
    "potpie/pots/contracts.py": {
        "PotAggregateStatus",
        "PotInfo",
        "PotManagementService",
        "PotRepoSource",
        "SourceInfo",
    },
    "potpie/pots/local_service.py": {"LocalPotManagementService"},
    "potpie/pots/local_store.py": {"LocalPotStore"},
    "potpie/pots/resolution.py": {"match_pot_ref", "repo_source_index"},
    "potpie/setup/contracts.py": {
        "NoOpSetupObserver",
        "SetupObserver",
        "SetupOrchestrator",
    },
    "potpie/setup/installation.py": {"Installer"},
    "potpie/setup/local_installer.py": {"LocalInstaller"},
    "potpie/setup/state.py": {"MigrationPort", "StateStorePort"},
    "potpie/setup/flat_file_state.py": {
        "FlatFileMigrator",
        "FlatFileStateStore",
    },
    "potpie/setup/orchestrator.py": {"DefaultSetupOrchestrator"},
    "potpie/skills/contracts.py": {
        "AgentTargetPort",
        "SkillInfo",
        "SkillManager",
        "SkillOperationResult",
        "SkillStatus",
    },
    "potpie/skills/catalog.py": {"catalog_by_id", "load_bundle_skills"},
    "potpie/skills/installer.py": {"InstallResult", "install_agent_bundle"},
    "potpie/skills/snippets.py": {"validate_packaged_skill_command_snippets"},
    "potpie/skills/manager.py": {"DefaultSkillManager"},
    "potpie/skills/targets.py": {"AgentTarget"},
}

DOCUMENTATION_ONLY_INITIALIZERS = (
    "potpie/config/__init__.py",
    "potpie/pots/__init__.py",
    "potpie/setup/__init__.py",
    "potpie/skills/__init__.py",
)

FORBIDDEN_SOURCE_UMBRELLAS = (
    "app",
    "application",
    "capabilities",
    "control_plane",
    "product",
)

CONCRETE_ASSEMBLY_TYPES = frozenset(
    {
        "AgentContextService",
        "DefaultSetupOrchestrator",
        "DefaultSkillManager",
        "FlatFileMigrator",
        "FlatFileStateStore",
        "LocalAuthService",
        "LocalConfigService",
        "LocalInstaller",
        "LocalPotManagementService",
        "LocalPotStore",
    }
)

EXPECTED_SDIST_CAPABILITIES = {
    "/potpie/agent_context.py",
    "/potpie/auth",
    "/potpie/cli",
    "/potpie/config",
    "/potpie/daemon",
    "/potpie/pots",
    "/potpie/runtime",
    "/potpie/setup",
    "/potpie/skills",
}
