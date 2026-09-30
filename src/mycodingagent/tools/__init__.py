# -*- coding: utf-8 -*-
"""工具包。

自研工具模块：
- calculator：安全计算器（AST 白名单求值）；
- shell：SafeShellBackend（含危险命令拦截）；
- git_tools：git status/diff/log/commit；
- ask_user / plan：执行中提问 / Plan 模式提交计划；
- user_memory：个人级长期记忆（命名空间 ("users",)，跨项目共享）；
- project_memory：项目级长期记忆（按项目命名空间隔离）；
- memory_common：两套记忆共用的记录结构与检索辅助。

文件读写 / 编辑 / 搜索（ls、read_file、write_file、edit_file、glob、grep、
execute）由 deepagents 依据 backend 能力自动注册，不在本包重复实现。
"""
