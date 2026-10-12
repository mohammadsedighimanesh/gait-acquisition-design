"""Assemble manuscript.md from the template + generated tables, then write manuscript.docx (raw OOXML, no extra packages) and manuscript.html.
Usage: python build_manuscript_docx.py"""
import base64
import html
import re
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / 'Paper/Manuscript_Acquisition_Design_2026_10_12'
TABLES = D / 'tables'

CAPTIONS = {
    'table1_datasets': ('Table 1.', 'Datasets, tasks and inputs. Trials are separately recorded walks (FPI, GaitRec) or footsteps (StepUP).', None),
    'table2_finite_differences': ('Table 2.', 'Finite differences between neighbouring cells of the learning surface. A step adds people at fixed trials per person, or adds trials at fixed people. A 95% interval above or below zero is counted per step.', 'finite_difference_summary'),
    'table3_equal_budget': ('Table 3.', 'Designs with equal numbers of recordings. Difference = design A (more people, fewer trials) minus design B, in balanced-accuracy points; log-loss difference is negative when design A is better. Intervals: 95% participant bootstrap conditional on the fitted models.', 'equal_budget_contrasts'),
    'table4_variability': ('Table 4.', 'Variability of the primary contrasts: standard deviation over acquisition draws within a repeat (mean of three repeats) and standard deviation of the three repeat means.', 'variability_primary_contrasts'),
    'table5_sensitivity': ('Table 5.', 'Primary contrasts with C fixed at 0.1 and with C chosen inside every acquired set. Differences are balanced accuracy (fraction) and log loss.', 'sensitivity_contrasts'),
    'table6_policy_ba': ('Table 6.', 'Balanced accuracy of the policies. Stage A4: FPI and GaitRec; A5: StepUP; A5.2: StepUP small-budget follow-up. Budgets marked saturated are budgets at which the recruitable pool is exhausted by the best designs.', 'policy_ba'),
    'table7_policy_contrasts': ('Table 7.', 'Sequential rule (MARGIN) compared with the five-trial convention (CONV5), the random rule and the best fixed design chosen after the fact. Differences in balanced-accuracy points with 95% participant bootstrap intervals. Composition is mean number of people x mean trials per person.', 'policy_contrasts'),
}
TABLE1 = """| Dataset | People | Trials used per person | Classes (people) | Inputs per trial | Folds (train / test people) |
|---|---|---|---|---|---|
| Foot posture (FPI) | 70 men | 5 walking trials | 3: neutral 30, highly supinated 20, highly pronated 20 | 28 trial descriptors (median over trials) and 19 anthropometric inputs | 3 x 5; 56 / 14 |
| GaitRec | 1,301 patients | 5 walking trials | 5: healthy 146, hip 234, knee 368, ankle 350, calcaneus 203 | 202 vertical-force values, reduced to 12 components | 3 x 5; about 1,040 / 260 |
| StepUP sex | 150 adults | 20 right footsteps | 2: female 76, male 74 | 200 pressure-block values, reduced to 12 components | 3 x 5; 120 / 30 |
| StepUP BMI | 150 adults | 20 right footsteps | 2: BMI below 25 76, 25 or above 74 | as StepUP sex | 3 x 5; 120 / 30 |
"""


def build_markdown():
    t = (D / 'manuscript_template.md').read_text(encoding='utf-8')
    (TABLES / 'table1_datasets.md').write_text(TABLE1, encoding='utf-8')

    def sub(m):
        name = m.group(1)
        if name in CAPTIONS:
            lab, cap, src = CAPTIONS[name]
            body = TABLE1 if name == 'table1_datasets' else (TABLES / f'{src}.md').read_text(encoding='utf-8')
            return f'**{lab}** {cap}\n\n{body}'
        return (TABLES / f'{name}.md').read_text(encoding='utf-8')
    t = re.sub(r'\{\{TABLE:([A-Za-z0-9_]+)\}\}', sub, t)
    t = t.replace('{{SHARE_PEOPLE}}', '56% to 67%')
    t = t.replace('{{UNDEFINED_TEXT}}', '2% to 4% of decisions in FPI, none in GaitRec, 7% at 300 recordings in StepUP, and 15% and 25% of decisions at 150 and 100 recordings in the StepUP small-budget stage')
    assert '{{' not in t, re.findall(r'\{\{[^}]+\}\}', t)
    (D / 'manuscript.md').write_text(t, encoding='utf-8')
    return t


# ----------------------------------------------------------------------------------------------- inline parsing
TOKEN = re.compile(r'(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)')


def runs(text):
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            out.append((part[2:-2], 'b'))
        elif part.startswith('*') and part.endswith('*') and len(part) > 2:
            out.append((part[1:-1], 'i'))
        else:
            out.append((part, ''))
    return out


def esc(x):
    return x.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def run_xml(txt, fmt, size=None, bold_all=False):
    rpr = ''
    if 'b' in fmt or bold_all:
        rpr += '<w:b/>'
    if 'i' in fmt:
        rpr += '<w:i/>'
    if size:
        rpr += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    return f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{esc(txt)}</w:t></w:r>'


def para(text, style=None, size=None, bold_all=False, ind=None, spacing_after=None, keep_next=False, jc=None):
    ppr = ''
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if keep_next:
        ppr += '<w:keepNext/>'
    if spacing_after is not None:
        ppr += f'<w:spacing w:after="{spacing_after}"/>'
    if ind:
        ppr += f'<w:ind w:left="{ind[0]}" w:hanging="{ind[1]}"/>'
    if jc:
        ppr += f'<w:jc w:val="{jc}"/>'
    return f'<w:p>{"<w:pPr>" + ppr + "</w:pPr>" if ppr else ""}{"".join(run_xml(t, f, size, bold_all) for t, f in runs(text))}</w:p>'


def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(24)
    return struct.unpack('>II', head[16:24])


def image_xml(rid, idx, path, max_in=6.3):
    w, h = png_size(path); cx = int(max_in * 914400); cy = int(cx * h / w)
    return (f'<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{idx}" name="Figure {idx}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{idx}" name="fig{idx}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def table_xml(rows):
    ncol = len(rows[0]); weights = []
    for c in range(ncol):
        L = max(len(re.sub(r'\*', '', r[c])) for r in rows if c < len(r)); weights.append(min(max(L, 16 if len(rows[0]) >= 8 else 11), 34))
    total = 9360; tw = sum(weights); widths = [int(total * w / tw) for w in weights]
    size = 15 if ncol >= 8 else (16 if ncol >= 7 else 17)
    b = '<w:tblBorders>' + ''.join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="8c8b86"/>' for s in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']) + '</w:tblBorders>'
    x = f'<w:tbl><w:tblPr><w:tblW w:w="{total}" w:type="dxa"/>{b}<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths) + '</w:tblGrid>'
    for ri, r in enumerate(rows):
        r = r + [''] * (ncol - len(r))
        x += '<w:tr>' + ('<w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' if ri == 0 else '<w:trPr><w:cantSplit/></w:trPr>')
        for ci, cell in enumerate(r):
            shd = '<w:shd w:val="clear" w:color="auto" w:fill="ecebe7"/>' if ri == 0 else ''
            x += f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shd}</w:tcPr>' + para(cell, size=size, bold_all=(ri == 0), spacing_after=0) + '</w:tc>'
        x += '</w:tr>'
    return x + '</w:tbl>' + '<w:p><w:pPr><w:spacing w:after="120"/></w:pPr></w:p>'


SETTINGS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat></w:settings>')

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri" w:eastAsia="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="320" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="220" w:after="80"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListPara"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="60"/></w:pPr></w:style>
</w:styles>"""


def to_docx(md, out_name='manuscript.docx'):
    body = []; media = []; fig_n = 0; lines = md.split('\n'); i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln:
            i += 1; continue
        if ln.startswith('# '):
            body.append(para(ln[2:], 'Title')); i += 1; continue
        if ln.startswith('## '):
            body.append(para(ln[3:], 'Heading1')); i += 1; continue
        if ln.startswith('### '):
            body.append(para(ln[4:], 'Heading2')); i += 1; continue
        if ln.strip() == '\\pagebreak':
            body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>'); i += 1; continue
        m = re.match(r'!\[[^\]]*\]\(([^)]+)\)', ln)
        if m:
            fig_n += 1; path = D / m.group(1); media.append((f'rIdImg{fig_n}', f'image{fig_n}.png', path)); body.append(image_xml(f'rIdImg{fig_n}', fig_n, path)); i += 1; continue
        if ln.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{3,}:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            body.append(table_xml(rows)); continue
        if ln.startswith('- '):
            while i < len(lines) and lines[i].startswith('- '):
                body.append(para('•  ' + lines[i][2:].strip(), 'ListPara', ind=(360, 270))); i += 1
            continue
        if re.match(r'\d+\. ', ln):
            while i < len(lines) and re.match(r'\d+\. ', lines[i]):
                body.append(para(lines[i].strip(), 'ListPara', ind=(450, 450), spacing_after=60)); i += 1
            continue
        buf = [ln]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'(#|\||- |!\[|\d+\. )', lines[i]):
            buf.append(lines[i].rstrip()); i += 1
        text = ' '.join(buf)
        keep = text.startswith('**Table') or text.startswith('**Figure')
        body.append(para(text, keep_next=text.startswith('**Table')))
    sect = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1304" w:right="1247" w:bottom="1304" w:left="1247" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>' + ''.join(body) + sect + '</w:body></w:document>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
          '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    drels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
             '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
             '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>'
             + ''.join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>' for rid, name, _ in media) + '</Relationships>')
    out = D / out_name
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ct); z.writestr('_rels/.rels', rels); z.writestr('word/document.xml', doc); z.writestr('word/styles.xml', STYLES); z.writestr('word/settings.xml', SETTINGS); z.writestr('word/_rels/document.xml.rels', drels)
        for rid, name, path in media:
            z.write(path, f'word/media/{name}')
    return out


def to_html(md, out_name='manuscript.html', title='Participants or repeated trials?'):
    css = ('body{font-family:Calibri,Arial,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#0b0b0b}h1{font-size:1.6rem}h2{font-size:1.25rem;margin-top:2rem}h3{font-size:1.05rem}'
           'table{border-collapse:collapse;font-size:.8rem;margin:.5rem 0 1.2rem;width:100%}th,td{border:1px solid #8c8b86;padding:3px 6px;vertical-align:top}th{background:#ecebe7}img{max-width:100%}')
    out = []; lines = md.split('\n'); i = 0

    def inl(t):
        t = html.escape(t); t = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t); t = re.sub(r'(?<![\w*])\*([^*\s][^*]*)\*', r'<i>\1</i>', t); return t
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln:
            i += 1; continue
        if ln.startswith('# '):
            out.append(f'<h1>{inl(ln[2:])}</h1>')
        elif ln.startswith('## '):
            out.append(f'<h2>{inl(ln[3:])}</h2>')
        elif ln.startswith('### '):
            out.append(f'<h3>{inl(ln[4:])}</h3>')
        elif ln.strip() == '\\pagebreak':
            out.append('<hr>')
        elif re.match(r'!\[[^\]]*\]\(([^)]+)\)', ln):
            path = D / re.match(r'!\[[^\]]*\]\(([^)]+)\)', ln).group(1); out.append(f'<p><img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"></p>')
        elif ln.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{3,}:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append('<table>' + '<tr>' + ''.join(f'<th>{inl(c)}</th>' for c in rows[0]) + '</tr>' + ''.join('<tr>' + ''.join(f'<td>{inl(c)}</td>' for c in r) + '</tr>' for r in rows[1:]) + '</table>'); continue
        elif ln.startswith('- '):
            items = []
            while i < len(lines) and lines[i].startswith('- '):
                items.append(f'<li>{inl(lines[i][2:])}</li>'); i += 1
            out.append('<ul>' + ''.join(items) + '</ul>'); continue
        elif re.match(r'\d+\. ', ln):
            items = []
            while i < len(lines) and re.match(r'\d+\. ', lines[i]):
                item_txt = re.sub(r'^\d+\. ', '', lines[i]); items.append('<li>' + inl(item_txt) + '</li>'); i += 1
            out.append('<ol>' + ''.join(items) + '</ol>'); continue
        else:
            buf = [ln]; i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r'(#|\||- |!\[|\d+\. )', lines[i]):
                buf.append(lines[i].rstrip()); i += 1
            out.append(f'<p>{inl(" ".join(buf))}</p>'); continue
        i += 1
    (D / out_name).write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head><body>' + '\n'.join(out) + '</body></html>', encoding='utf-8')


def split(md):
    marker = '\\pagebreak'
    main, supp = md.split(marker, 1)
    title = main.splitlines()[0][2:].strip()
    supp = supp.replace('## Supplementary material', '', 1).strip()
    supp = '# Supplementary Information\n\n' + f'for: {title}\n\n[Authors: to be completed by the authors]\n\n' + supp + '\n'
    (D / 'manuscript.md').write_text(main.rstrip() + '\n', encoding='utf-8'); (D / 'supplementary_information.md').write_text(supp, encoding='utf-8')
    return main, supp


if __name__ == '__main__':
    md = build_markdown(); main, supp = split(md)
    p = to_docx(main, 'Manuscript_final.docx'); to_html(main, 'Manuscript_final.html')
    p2 = to_docx(supp, 'Supplementary_Information_final.docx'); to_html(supp, 'Supplementary_Information_final.html', 'Supplementary Information')
    # well-formedness check
    import xml.dom.minidom as mdm
    for doc in (p, p2):
        with zipfile.ZipFile(doc) as z:
            for n in ['word/document.xml', 'word/styles.xml', 'word/settings.xml', '[Content_Types].xml', 'word/_rels/document.xml.rels']:
                mdm.parseString(z.read(n))
            print(doc.name, 'parts ok:', len(z.namelist()))
    print('main-text words incl. tables/refs:', len(re.findall(r'\w+', main)))
