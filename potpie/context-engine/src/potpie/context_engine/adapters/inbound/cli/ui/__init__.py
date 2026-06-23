"""Terminal presentation for the host CLI.

Pure rendering — shared visual language (:mod:`format`, :mod:`brand`),
formatting/logging (:mod:`output`), the animated setup wizard
(:mod:`setup_ux`, :mod:`setup_wizard_ui`), and the logo display
(:mod:`potpie_logo_anim`, :mod:`static_logo_loader`). No domain/service logic;
command bodies under ``commands/`` and ``auth/`` drive these.
"""
