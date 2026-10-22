# Potpie's FalkorDBLite Linux ARM64 build

This distribution repairs FalkorDBLite 0.10.0 packaging for Linux ARM64. It
provides the same `redislite` import and data format. Do not install it alongside
the upstream `falkordblite` distribution: both own that import namespace.

For an existing environment, stop the Potpie daemon and download/build the
corrected wheel first. Then remove the upstream distribution **before**
installing the repair:

```sh
python -m pip uninstall falkordblite
python -m pip install --force-reinstall --no-deps /path/to/potpie_falkordblite-0.10.0.post1-<platform>.whl
```

Use the wheel's actual filename. Package removal does not remove Potpie's data
directory. Installing the repair first and then uninstalling the old package
is unsafe: pip does not track shared file ownership and can remove the repair's
`redislite` files. If both distributions were already installed, uninstall the
original and force-reinstall the repair in that order before restarting the
daemon. Fresh isolated environments do not need this migration.

The upstream source archive includes its builder's x86-64 graph module. On
Linux systems too old for the upstream ARM64 wheel, a source build can therefore
compile an ARM64 Redis server but reuse the x86-64 module. This recipe discards
that module and builds from SHA256-pinned upstream inputs. It validates the
architecture of both native binaries and runs `auditwheel repair` to validate
the Linux compatibility tag and bundle permitted shared libraries. It retains
Potpie's Redis client `<8` constraint.

The build runs on native Linux ARM64 with a C compiler and `make`, targeting
glibc 2.36 or newer (for example Debian Bookworm). Building on a newer distro may
produce binaries with newer symbol requirements; auditwheel then fails rather
than claiming false compatibility. The native module is FalkorDB v4.18.3 and
the Redis sources come from the pinned FalkorDBLite 0.10.0 source archive.

```sh
python -m build --wheel potpie/falkordblite
```

To build with network access disabled, first populate a source cache:

```sh
python scripts/build_falkordblite_wheel.py --download-only --cache-dir /tmp/native-inputs
POTPIE_NATIVE_SOURCE_CACHE=/tmp/native-inputs POTPIE_NATIVE_OFFLINE=1 \
    python -m build --wheel potpie/falkordblite
```

Build requirements must also be preinstalled when building offline; pass
`--no-isolation` to `build` in that case. Runtime never downloads native code or
changes installed packages. Wheel provenance is stored in
`redislite/potpie_build.json`. Source distributions contain only this recipe,
so building them on another machine cannot reuse the release builder's binary.

The repaired wheel preserves upstream license files. Upstream source and
licensing information: <https://github.com/FalkorDB/falkordblite/tree/v0.10.0>,
<https://github.com/FalkorDB/FalkorDB/tree/v4.18.3>.
