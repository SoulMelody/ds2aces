# ds2aces

将 DiffSinger 文件转换为 ACE Sequence 文件，并在终端内渲染（无需打开 ACE Studio）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 简介

`ds2aces` 是一个 Python 工具，可以将 DiffSinger 项目文件转换为 ACE Studio 可用的 ACE Sequence 文件格式。它还提供了直接在终端中渲染这些序列文件的功能。

## 使用方法

### 基本命令

```bash
# 转换 DiffSinger 文件到 ACE Sequence 格式，并渲染
python -m ds2aces ds2aces --render <input.ds>
```

### 登录 ACE Studio 账号

对于Linux用户，可以在终端内登录 ACE Studio 账号：

```bash
python -m ds2aces login
```

## 注意事项

- 本工具不能和 ACE Studio 软件同时运行。
- 渲染功能需要先登录 ACE Studio 账号，否则无法使用。
- 目前仅支持中文音素的 DiffSinger 文件转换。
- 目前仅支持音高参数的转换。

## 参考

- [LibreSVIP 的 DiffSinger 插件](https://github.com/SoulMelody/LibreSVIP/tree/main/libresvip/plugins/ds)
- [ACE Sequence File](https://github.com/timedomain-tech/ACE_sequence_file/)