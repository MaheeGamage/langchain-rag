# pipeline/stages/walker.py
"""
Stage 1 — Walk the data root and return all file paths.

Swap this stage to change how files are discovered (e.g. S3 bucket,
filtered glob, explicit file list).
"""

from pathlib import Path


# Extensions that carry no embeddable knowledge content.
# Add to this set rather than scattering skip logic elsewhere.
SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".rst",     # Sphinx directive stubs — too terse without docstring resolution
    ".json",    # TOC / navigation metadata
    ".csv",     # Raw tabular data
    ".pack",    # Git object pack
    ".idx",     # Git index
    ".rev",     # Git revision
    ".sample",  # Git hook samples
})


class WalkerStage:
    """
    Recursively collect all files under a root directory.

    Parameters
    ----------
    skip_extensions:
        File suffixes to ignore.  Defaults to SKIP_EXTENSIONS.
    skip_hidden:
        If True (default), skip files/dirs whose name starts with '.'.
    """

    def __init__(
        self,
        skip_extensions: frozenset[str] = SKIP_EXTENSIONS,
        skip_hidden: bool = True,
    ) -> None:
        self.skip_extensions = skip_extensions
        self.skip_hidden = skip_hidden

    def run(self, data_root: Path) -> list[Path]:
        """
        Return a sorted list of candidate file paths under data_root.

        Sorted for deterministic ordering across runs.
        """
        paths: list[Path] = []
        for p in sorted(data_root.rglob("*")):
            if not p.is_file():
                continue
            if self.skip_hidden and any(part.startswith(".") for part in p.parts):
                continue
            if p.suffix.lower() in self.skip_extensions:
                continue
            paths.append(p)
        return paths
