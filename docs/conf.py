# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'msModelSlim'
copyright = '2026, Huawei'
author = 'Huawei'

# -- General configuration ---------------------------------------------------

# 1. 配置文档主题（
html_theme = 'sphinx_book_theme'

html_theme_options = {
    # 开启左侧导航折叠
    'collapse_navbar': True,
    # 右侧边栏显示当前文档目录的深度（设置 2 或 3）
    'show_toc_level': 2,
    # 限制左侧全局导航的初始显示深度为 1
    'navigation_depth': 1,
}

myst_heading_anchors = 3

# 2. 支持中文（避免乱码）
language = 'zh_CN'

# 3. 添加必要扩展（支持 Markdown、代码高亮、目录生成等）
extensions = [
    'sphinx.ext.autodoc',    # 自动生成代码文档
    'sphinx.ext.napoleon',   # 支持 Google/Numpy 风格的注释
    'myst_parser',           # 支持 Markdown 文件
    'sphinx.ext.todo',       # 支持 TODO 标记
    'sphinx.ext.intersphinx', # 支持跨文档引用
    'sphinx.ext.imgconverter', # 支持图片格式转换
    'sphinx.ext.mathjax',    # 支持数学公式
    'sphinx.ext.viewcode',   # 查看代码源文件
]

# 4. 若使用 Markdown，需指定源文件后缀
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# 添加源文件编码配置
source_encoding = 'utf-8'

# 5. 配置文档版本（与 GitHub 仓库分支对应，如 v1.0）
version = '1.0'
release = '1.0'

# 6. 添加 myst_parser 配置
myst_enable_extensions = [
    'linkify',
    'html_image',
    'smartquotes',
    'dollarmath',            # 支持 $ 分隔的数学公式
    'html_admonition',       # 支持 HTML 警告框
    'replacements',          # 支持文本替换
]

# （可选）配置 Mermaid 输出格式
myst_mermaid_output_format = 'svg'  # 或 'png'

# 添加以下配置来解决Pygments无法识别mermaid的问题
# 忽略Pygments无法识别mermaid的警告
suppress_warnings = [
    'myst.xref_missing',     # 忽略交叉引用丢失的警告
    'myst.header',           # 忽略标题格式警告
    'misc.highlighting_failure', # 忽略语法高亮失败的警告
]

# 添加交叉引用支持
default_role = 'any'
myst_all_links_external = False
myst_highlight_code_blocks = True
myst_heading_anchors = 3  # 为标题生成锚点的级别
myst_footnote_transition = True
myst_dmath_double_inline = True

# 8. 添加静态文件路径配置
html_static_path = ['_static']
