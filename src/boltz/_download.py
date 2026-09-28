"""Publish cache entries only after a complete download or extraction."""

import tarfile
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory


def download_file(urls: list[str], destination: Path) -> None:
    """Try mirrors without leaving a failed transfer at the cache filename."""
    if not urls:
        raise ValueError("At least one download URL is required")
    for index, url in enumerate(urls):
        # Same filesystem as the destination, so replace is atomic. The context
        # also removes partial bytes on exceptions and keyboard interruption.
        with TemporaryDirectory(
            prefix=f".{destination.name}-", dir=destination.parent
        ) as tmp:
            staging = Path(tmp) / destination.name
            try:
                urllib.request.urlretrieve(url, staging)  # noqa: S310
                staging.replace(destination)
                return
            except Exception as error:
                if index == len(urls) - 1:
                    raise RuntimeError(
                        f"Failed to download {destination.name} from all URLs"
                    ) from error


def extract_mols(archive: Path, cache: Path) -> None:
    """Expose the molecule cache only when every archive member was extracted."""
    with TemporaryDirectory(prefix=".mols-", dir=cache) as tmp:
        staging = Path(tmp)
        with tarfile.open(archive, "r") as source:
            # The molecule archive contains data files and directories. Reject
            # links/devices as well as paths escaping the temporary directory.
            for member in source.getmembers():
                if not (member.isfile() or member.isdir()):
                    raise ValueError(
                        f"Unsupported molecule archive member: {member.name}"
                    )
            source.extractall(staging, filter="data")
        mols = staging / "mols"
        if not mols.is_dir():
            raise ValueError("Molecule archive does not contain a mols directory")
        mols.replace(cache / "mols")
