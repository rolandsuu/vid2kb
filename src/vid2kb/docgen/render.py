from __future__ import annotations

import html
import re
from pathlib import Path

from vid2kb.schemas import KnowledgeDocument

_PDF_CSS = (
    "@page { size: A4; margin: 1.5cm } "
    "body { font-family: 'PingFang SC', 'Noto Sans CJK SC', sans-serif; "
    "font-size: 11pt; line-height: 1.6 } "
    "h1 { font-size: 20pt } "
    "h2 { font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt } "
    "table { border-collapse: collapse } "
    "td, th { border: 1px solid #999; padding: 3pt 6pt }"
)


def _fmt_timestamp(seconds: float) -> str:
    return f'{int(seconds // 60):02d}:{int(seconds % 60):02d}'


def render_markdown(doc: KnowledgeDocument) -> str:
    lines: list[str] = [f'# {doc.title}', '']

    lines.append(f'- 文档类型：{doc.doc_type}')
    lines.append(f'- 目标读者：{doc.audience}')
    lines.append(f'- 摘要：{doc.summary}')
    lines.append('')

    for section in doc.sections:
        lines.append(f'## {section.heading}')
        lines.append('')
        if section.source_timestamps:
            marks = ' '.join(f'[{_fmt_timestamp(ts)}]' for ts in section.source_timestamps)
            lines.append(f'{section.body_md} {marks}')
        else:
            lines.append(section.body_md)
        lines.append('')

    lines.append('## 关键要点')
    lines.append('')
    for point in doc.key_points:
        lines.append(f'- {point}')
    lines.append('')

    lines.append('## 术语表')
    lines.append('')
    lines.append('| 术语 | 含义 |')
    lines.append('| --- | --- |')
    for term, meaning in doc.glossary.items():
        lines.append(f'| {term} | {meaning} |')
    lines.append('')

    lines.append('## 说明')
    lines.append('')
    for warning in doc.warnings:
        lines.append(f'- {warning}')

    return '\n'.join(lines).rstrip() + '\n'


def _render_table(rows: list[list[str]]) -> str:
    if not rows:
        return ''
    parts: list[str] = ['<table>']
    header = rows[0]
    data_rows = rows[1:]
    if data_rows and all(_is_separator(cell) for cell in data_rows[0]):
        data_rows = data_rows[1:]
        parts.append('<tr>' + ''.join(f'<th>{cell}</th>' for cell in header) + '</tr>')
    else:
        data_rows = rows
    for row in data_rows:
        parts.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>')
    parts.append('</table>')
    return ''.join(parts)


def _is_separator(cell: str) -> bool:
    return bool(re.fullmatch(r':?-+:?', cell))


def md_to_html(md_text: str) -> str:
    escaped = html.escape(md_text)
    lines = escaped.split('\n')
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith('### '):
            out.append(f'<h3>{stripped[4:]}</h3>')
        elif stripped.startswith('## '):
            out.append(f'<h2>{stripped[3:]}</h2>')
        elif stripped.startswith('# '):
            out.append(f'<h1>{stripped[2:]}</h1>')
        elif stripped.startswith('|'):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                rows.append(cells)
                i += 1
            out.append(_render_table(rows))
            continue
        elif stripped.startswith('- '):
            out.append(f'<li>{stripped[2:]}</li>')
        else:
            out.append(f'<p>{stripped}</p>')
        i += 1
    return '\n'.join(out)


def render_pdf(md_text: str, out_path: Path) -> Path:
    import weasyprint  # noqa: F401
    from weasyprint import HTML

    body = md_to_html(md_text)
    full_html = (
        '<html><head><meta charset="utf-8"><style>'
        f'{_PDF_CSS}'
        f'</style></head><body>{body}</body></html>'
    )
    HTML(string=full_html).write_pdf(str(out_path))
    return out_path
