"""Interrupted downloads/extractions cannot create completed cache entries."""

import io
import tarfile
from pathlib import Path

import pytest

from boltz._download import download_file, extract_mols


def test_download_retry_does_not_reuse_partial_file(tmp_path, monkeypatch):
    destination = tmp_path / "weights.ckpt"

    def interrupted(url, path):
        Path(path).write_bytes(b"partial")
        raise OSError("connection dropped")

    monkeypatch.setattr("urllib.request.urlretrieve", interrupted)
    with pytest.raises(RuntimeError, match="all URLs"):
        download_file(["first", "second"], destination)
    assert list(tmp_path.iterdir()) == []

    def complete(url, path):
        Path(path).write_bytes(b"complete")

    monkeypatch.setattr("urllib.request.urlretrieve", complete)
    download_file(["first"], destination)
    assert destination.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [destination]


def test_fallback_starts_empty_and_keyboard_interrupt_is_clean(tmp_path, monkeypatch):
    destination = tmp_path / "weights.ckpt"
    calls = []

    def transfer(url, path):
        assert not Path(path).exists()
        calls.append(url)
        Path(path).write_bytes(url.encode())
        if url == "first":
            raise OSError("connection dropped")

    monkeypatch.setattr("urllib.request.urlretrieve", transfer)
    download_file(["first", "second"], destination)
    assert calls == ["first", "second"]
    assert destination.read_bytes() == b"second"

    def interrupted(url, path):
        Path(path).write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr("urllib.request.urlretrieve", interrupted)
    with pytest.raises(KeyboardInterrupt):
        download_file(["first"], destination)
    assert destination.read_bytes() == b"second"
    assert list(tmp_path.iterdir()) == [destination]


def make_archive(path, names):
    with tarfile.open(path, "w") as archive:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 5
            archive.addfile(member, io.BytesIO(b"hello"))


def test_interrupted_extraction_can_retry(tmp_path, monkeypatch):
    archive = tmp_path / "mols.tar"
    make_archive(archive, ["mols/ALA.pkl", "mols/GLY.pkl"])
    original = tarfile.TarFile.extractall

    def interrupted(self, path, **kwargs):
        partial = Path(path) / "mols"
        partial.mkdir()
        (partial / "ALA.pkl").write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(tarfile.TarFile, "extractall", interrupted)
    with pytest.raises(OSError, match="disk full"):
        extract_mols(archive, tmp_path)
    assert list(tmp_path.iterdir()) == [archive]
    monkeypatch.setattr(tarfile.TarFile, "extractall", original)
    extract_mols(archive, tmp_path)
    assert (tmp_path / "mols" / "ALA.pkl").read_bytes() == b"hello"
    assert (tmp_path / "mols" / "GLY.pkl").read_bytes() == b"hello"


def test_archive_escape_is_rejected_without_publishing_cache(tmp_path):
    archive = tmp_path / "mols.tar"
    make_archive(archive, ["mols/ALA.pkl", "../escaped.pkl"])
    with pytest.raises(tarfile.FilterError):
        extract_mols(archive, tmp_path)
    assert list(tmp_path.iterdir()) == [archive]
    assert not (tmp_path.parent / "escaped.pkl").exists()
