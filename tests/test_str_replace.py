# -*- coding: utf-8 -*-
"""str_replace 精确编辑测试（阶段 1：edit_file 的唯一性校验语义）。

deepagents 的 FilesystemBackend.edit 就是 edit_file 工具的底层实现，
唯一性校验、replace_all、文件缺失报错都在这一层，直接对它做测试。
"""

import pytest

from deepagents.backends.filesystem import FilesystemBackend


@pytest.fixture()
def backend(tmp_path):
    return FilesystemBackend(root_dir=str(tmp_path))


def _edit(backend, content, old, new, replace_all=False):
    backend.write("/sample.py", content)
    return backend.edit("/sample.py", old, new, replace_all=replace_all)


def test_unique_replacement_succeeds(backend):
    content = "def add(a, b):\n    return a + b\n"
    result = _edit(backend, content, "return a + b", "return a + b  # sum")
    assert result.error is None
    assert result.occurrences == 1
    assert backend.read("/sample.py").file_data["content"] == (
        "def add(a, b):\n    return a + b  # sum\n"
    )


def test_non_unique_old_string_fails(backend):
    content = "x = 1\ny = 1\n"
    result = _edit(backend, content, "1", "2")
    assert result.error is not None          # 不唯一 → 明确报错，拒绝替换
    # 文件保持原样，没有部分修改
    assert backend.read("/sample.py").file_data["content"] == content


def test_old_string_not_found_fails(backend):
    content = "value = 42\n"
    result = _edit(backend, content, "not_in_file", "whatever")
    assert result.error is not None


def test_replace_all_replaces_every_occurrence(backend):
    content = "tmp = 1\ntmp = tmp + 1\n"
    result = _edit(backend, content, "tmp", "total", replace_all=True)
    assert result.error is None
    assert result.occurrences == 3
    assert backend.read("/sample.py").file_data["content"] == (
        "total = 1\ntotal = total + 1\n"
    )


def test_edit_missing_file_reports_error(backend):
    result = backend.edit("/no_such_file.py", "a", "b")
    assert result.error is not None
    assert "not found" in result.error
