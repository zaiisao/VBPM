"""Check the restored reference's computation against the PDF appendices.

PDF line numbers, page breaks, indentation loss, wrapped lines and typographic
apostrophes are formatting artifacts. Comments/docstrings are excluded from the
comparison; every remaining Python token must match in order.
"""
import ast
import hashlib
import io
import json
import re
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
REFERENCE = HERE/'reference'
PROVENANCE = REFERENCE/'provenance.json'


def numbered_code(text, heading, end_heading=None):
    start = re.search(r'^'+heading, text, re.MULTILINE).start()
    block = text[start:]
    if end_heading:
        block = block[:re.search(r'^'+end_heading, block, re.MULTILINE).start()]
    rows = {}
    previous = None
    for raw in block.splitlines():
        match = re.match(r'^\s*(\d+)(?: {3}(.*))?$', raw)
        if match and int(match.group(1)) == len(rows)+1:
            previous = int(match.group(1))
            rows[previous] = match.group(2) or ''
        elif previous is not None and raw.strip() and not raw.strip().isdigit():
            rows[previous] += ' '+raw.strip()
    return ('\n'.join(rows[number] for number in range(1, len(rows)+1))+'\n').replace('’', "'").replace('‘', "'")


def computation_tokens(source):
    # PDF indentation shifts at page breaks. Preserve it in the executable
    # transcription, but ignore indentation tokens during this comparison.
    flattened = '\n'.join(line.lstrip() for line in source.splitlines())
    omitted = {tokenize.ENDMARKER, tokenize.ENCODING, tokenize.INDENT,
               tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT}
    result = []
    for token in tokenize.generate_tokens(io.StringIO(flattened).readline):
        if token.type in omitted:
            continue
        if token.type == tokenize.STRING and token.string.startswith(('"""', "'''")):
            continue
        result.append((token.type, token.string))
    return result


def audit(output, pdf=None):
    provenance = json.loads(PROVENANCE.read_text())
    if pdf is not None:
        pdf = Path(pdf).resolve()
        text = subprocess.run(['pdftotext', '-layout', str(pdf), '-'],
                              check=True, capture_output=True, text=True).stdout
    else:
        text = None
    output.mkdir(parents=True, exist_ok=True)
    sections = [('vae_dbn.py', r'A\s+PyTorch implementation', r'B\s+Debug logger'),
                ('train_logger.py', r'B\s+Debug logger', r'C\s+Curve plotting'),
                ('plot_logs.py', r'C\s+Curve plotting', None)]
    result = dict(pdf=str(pdf) if pdf else None,
                  pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest() if pdf else provenance['pdf_sha256'],
                  audit_mode='supplied PDF' if pdf else 'archived PDF code extractions',
                  ignored='line numbers, layout whitespace, comments, docstrings; curly apostrophes normalized', files={})
    for filename, heading, end_heading in sections:
        extracted = (numbered_code(text, heading, end_heading) if text is not None
                     else (REFERENCE/'pdf_extracted'/(filename+'.txt')).read_text())
        source = (REFERENCE/filename).read_text()
        ast.parse(source)
        expected, actual = computation_tokens(extracted), computation_tokens(source)
        matched = expected == actual
        result['files'][filename] = dict(computation_tokens_match_pdf=matched,
            token_count=len(actual), sha256=hashlib.sha256(source.encode()).hexdigest(),
            recorded_source_sha256=provenance['files'][filename]['sha256'],
            source_matches_recorded_hash=hashlib.sha256(source.encode()).hexdigest() == provenance['files'][filename]['sha256'],
            extracted_numbered_lines=len(extracted.splitlines()))
        (output/(filename+'.pdf.txt')).write_text(extracted)
        difference_path = output/(filename+'.diff')
        if matched and difference_path.exists():
            difference_path.unlink()
        if not matched:
            import difflib
            diff = '\n'.join(difflib.unified_diff([repr(token) for token in expected],
                [repr(token) for token in actual], fromfile='PDF', tofile=filename))
            difference_path.write_text(diff+'\n')
            (output/'fidelity.json').write_text(json.dumps(result, indent=2)+'\n')
            raise RuntimeError('Reference differs from PDF: '+filename)
    result['passed'] = True
    (output/'fidelity.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(audit(ROOT/'outputs/30s/reference_fidelity'), indent=2))
