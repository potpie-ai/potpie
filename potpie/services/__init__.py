"""Potpie host-app services and composition wiring.

The product-side application services (setup orchestration, pot management,
skills install, managed local services, config, auth) plus the composition
root (``host_wiring.build_host_shell``) that assembles the HostShell from
context-engine adapters. Delivery surfaces (``potpie.cli``, ``potpie.daemon``,
``potpie.mcp``) drive these services; the context engine below never imports
this package.
"""
