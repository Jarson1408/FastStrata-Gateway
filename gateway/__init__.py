"""FastStrata Gateway — an OpenAI-compatible proxy for Qwen XML tool calls
that supervises the LODESTAR memory protocol.

The INSTALLED version, read from package metadata. It used to be a literal
here and said 2.4.0 while app.py said 6.1.0-Lodestar and pyproject said 6.1.0
-- three spellings, and nothing read the stale one, which is exactly why it
stayed wrong.

**This used to claim it "cannot drift from pyproject.toml". It can.** Metadata
is a snapshot from the last `pip install`, so between a version bump and a
reinstall the source tree and this disagree -- seen cutting v7.0.0 on
2026-09-13, when pyproject said 7.0.0 and this said 6.1.0. That is the right
answer for an installed package and the wrong one for a source checkout, and
nothing in this package reads it either way.

The number a reader actually sees is `GATEWAY_VERSION` in app.py, which is a
literal. `gateway/tests/test_version_is_one_number.py` compares the two files
a human edits, because a claim that two things cannot disagree is worth less
than a test that fails when they do.
"""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("faststrata-gateway")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+source"
