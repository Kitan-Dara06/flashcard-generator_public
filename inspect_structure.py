"""How much structure does the source document actually have?

Hierarchy (document -> sections -> claims) needs sections. This reports what
the extractor currently sees and what a heading-aware extractor could see.
"""
import sys

import docx
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_blocks(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag.endswith('}p'):
            yield Paragraph(child, document)
        elif child.tag.endswith('}tbl'):
            yield Table(child, document)


def main():
    path = sys.argv[1]
    document = docx.Document(path)

    total_paragraphs = 0
    total_chars = 0
    headings = []
    heading_chars = 0

    for block in iter_blocks(document):
        if not isinstance(block, Paragraph):
            continue
        text = block.text.strip()
        if not text:
            continue
        total_paragraphs += 1
        total_chars += len(text)
        style = (block.style.name or '') if block.style is not None else ''
        if style.lower().startswith('heading') or style.lower() in ('title', 'subtitle'):
            headings.append((style, text))
            heading_chars += len(text)

    print(f"file            : {path.split('/')[-1]}")
    print(f"paragraphs      : {total_paragraphs}")
    print(f"body chars      : {total_chars:,}")
    print(f"headings found  : {len(headings)}")
    print(f"heading chars   : {heading_chars:,}")
    print()

    if headings:
        print("Outline (what hierarchy would use):")
        for style, text in headings[:40]:
            indent = '  ' * max(0, int(''.join(c for c in style if c.isdigit()) or 1) - 1)
            print(f"  {indent}[{style}] {text[:88]}")
        if len(headings) > 40:
            print(f"  ... and {len(headings) - 40} more")
    else:
        print("NO heading styles: hierarchy would have to infer sections heuristically")
        print("(short lines, ALL CAPS, '1.2.3' numbering, or blank-line gaps)")

    # tables
    tables = document.tables
    print(f"\ntables          : {len(tables)}")
    if tables:
        cells = sum(len(r.cells) for t in tables for r in t.rows)
        print(f"table cells     : {cells}")

    # rough per-section sizes if we use headings as boundaries
    if headings:
        print("\nEstimated section sizes (chars):")
        section_chars = []
        current = 0
        for block in iter_blocks(document):
            if not isinstance(block, Paragraph):
                continue
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name or '') if block.style is not None else ''
            if style.lower().startswith('heading') or style.lower() in ('title', 'subtitle'):
                if current:
                    section_chars.append(current)
                current = 0
            current += len(text)
        if current:
            section_chars.append(current)
        sizes = sorted(section_chars, reverse=True)
        print(f"  sections: {len(sizes)} | largest {sizes[0]:,} | median "
              f"{sizes[len(sizes)//2]:,} | smallest {sizes[-1]:,}")
        if sizes:
            print(f"  a 24k-char chunk would hold ~{24000 // max(1, sizes[len(sizes)//2])} "
                  f"median sections")


if __name__ == '__main__':
    main()
