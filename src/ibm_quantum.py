"""IBM Quantum status (configuration detection only).

QueSat v1.0 does NOT execute anything on IBM hardware. This module only reports whether
credentials appear to be configured, without reading or displaying their values.
Credentials must be supplied through environment variables or a saved Qiskit account —
never in source code. See docs/ibm_quantum.md.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

TOKEN_ENV = "QISKIT_IBM_TOKEN"
CONFIG_REQUIRED = "IBM Quantum Hardware — Configuration Required"


def ibm_status() -> dict:
    runtime_installed = importlib.util.find_spec("qiskit_ibm_runtime") is not None
    env_token = bool(os.environ.get(TOKEN_ENV))
    saved_account = (Path.home() / ".qiskit" / "qiskit-ibm.json").exists()
    credentials = env_token or saved_account

    if not credentials:
        label, detail = CONFIG_REQUIRED, "No IBM Quantum credentials were found in the environment."
    elif not runtime_installed:
        label = "Credentials found — qiskit-ibm-runtime not installed"
        detail = "Install qiskit-ibm-runtime to enable the hardware extension."
    else:
        label = "Credentials found — hardware execution not performed"
        detail = ("Credentials were detected but not verified, and this prototype does not submit "
                  "hardware jobs. See docs/ibm_quantum.md to extend it.")
    return {"label": label, "detail": detail, "credentials_found": credentials,
            "runtime_installed": runtime_installed, "hardware_executed": False,
            "source": "environment variable" if env_token else ("saved account" if saved_account else None)}
