# -*- coding: utf-8 -*-
"""用户级记忆文件加载（~/.mycodingagent/MEMORY.md）测试。

真实文件位于用户主目录、不纳入 git，因此这些测试全部用 tmp_path 构造临时
文件（通过给 _user_memory_block 显式传路径），不读取/不依赖真实记忆文件。
"""

from pathlib import Path

from mycodingagent.agent import _read_optional_text, _user_memory_block


# ============================================================
# _read_optional_text
# ============================================================
def test_read_optional_text_returns_content(tmp_path):
    f = tmp_path / "MEMORY.md"
    f.write_text("# 标题\n内容", encoding="utf-8")
    assert _read_optional_text(f) == "# 标题\n内容"


def test_read_optional_text_missing_returns_empty(tmp_path):
    assert _read_optional_text(tmp_path / "nope.md") == ""


def test_read_optional_text_directory_returns_empty(tmp_path):
    # 对目录调 read_text 抛 IsADirectoryError（OSError 子类），应安全降级
    assert _read_optional_text(tmp_path) == ""


# ============================================================
# _user_memory_block
# ============================================================
def test_user_memory_block_with_content(tmp_path):
    f = tmp_path / "MEMORY.md"
    f.write_text("- 用中文沟通", encoding="utf-8")
    block = _user_memory_block(f)
    assert "用户级记忆" in block
    assert "~/.mycodingagent/MEMORY.md" in block
    assert block.endswith("- 用中文沟通")


def test_user_memory_block_empty_file_returns_empty(tmp_path):
    f = tmp_path / "MEMORY.md"
    f.write_text("   \n", encoding="utf-8")  # 仅空白：视为无内容
    assert _user_memory_block(f) == ""


def test_user_memory_block_missing_returns_empty(tmp_path):
    assert _user_memory_block(tmp_path / "nope.md") == ""


def test_user_memory_block_default_path_under_home(tmp_path, monkeypatch):
    """不传路径时默认锚定 Path.home()/.mycodingagent/MEMORY.md。

    Windows 下 Path.home() 读 USERPROFILE，临时改写以隔离真实主目录。
    """
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    # 尚无文件：安全降级为空
    assert _user_memory_block() == ""
    # 在默认位置创建文件：应被读取并注入
    mem_dir = tmp_path / ".mycodingagent"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text("- 默认路径内容", encoding="utf-8")
    block = _user_memory_block()
    assert "用户级记忆" in block and "默认路径内容" in block
