"""``potpie.services.managed_services`` — ``ServiceBackend`` adapters the daemon's
``ServiceManager`` drives to run supporting services: a local ``subprocess``, a docker
``container``, or an already-running ``external`` endpoint (probe-only).
"""
