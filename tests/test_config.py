from __future__ import annotations

import typing as t

from landnet.config import save_vars_as_json

if t.TYPE_CHECKING:
    from pathlib import Path


def test_save_vars_as_json(tmp_path: Path):
    out_file = tmp_path / 'test_save_vars.json'
    save_vars_as_json(tmp_path / 'test_save_vars.json')
    assert out_file.exists()
