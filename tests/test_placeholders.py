from coduar import __version__ as coduar_version
from assembly_copilot import __version__ as assembly_copilot_version


def test_placeholder_versions() -> None:
    assert coduar_version == "0.0.0"
    assert assembly_copilot_version == "0.0.0"
