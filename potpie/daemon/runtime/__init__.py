"""``potpie.daemon.runtime`` - the in-daemon-process runtime.

When the host runs detached (``host_mode = "daemon"``), this is what the background
process executes: the async ``DaemonRuntime`` that binds transports, starts managed
services, and serves the registered components' operations. Distinct from
``root PotpieRuntime`` (the in-process service facade) - the runtime *hosts* a
PotpieRuntime and exposes its surfaces over a transport.
"""
