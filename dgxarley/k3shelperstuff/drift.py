#!/usr/bin/env python3
"""One CLI for both drift checks: ``pin`` (:mod:`dgxarley.k3shelperstuff.pin_drift`) and ``keel`` (:mod:`dgxarley.k3shelperstuff.keel_drift`).

Installed as the ``k3s-drift`` entry point (extra ``dgxarley[k3s]``), also
runnable as ``python -m dgxarley.k3shelperstuff.drift``.

Examples:
    k3s-drift --help
    k3s-drift pin [pin-drift options]
    k3s-drift keel [keel-drift options]
"""

import os
from typing import Final

import typer

from dgxarley.k3shelperstuff import keel_drift, pin_drift

CLI_HELP: Final[str] = """Drift checks for a K3s cluster and the repo that deploys it.

[bold]pin[/]  compares fixed version pins (v1.2.3) against upstream releases.
[bold]keel[/] compares the digests of running pods against what their rolling tags point at now.

Both exit with 1 as soon as there is drift, so they work as a gate in a pipeline.
"""

app: Final[typer.Typer] = typer.Typer(
    add_completion=False, no_args_is_help=True, rich_markup_mode="rich", help=CLI_HELP
)
app.command("pin", help=pin_drift.CLI_HELP, short_help="Version pins in pin_drift.yml vs. upstream releases.")(
    pin_drift.main
)
app.command(
    "keel", help=keel_drift.CLI_HELP, short_help="Running pod digests vs. registry digests of their rolling tags."
)(keel_drift.main)


if __name__ == "__main__":
    app(prog_name=os.environ.get("DRIFT_PROG_NAME"))
