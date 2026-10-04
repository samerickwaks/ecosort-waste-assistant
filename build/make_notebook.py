"""Assemble the percent-format sources into the deliverable notebook."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from py2nb import convert                                      # noqa: E402

SOURCES = ['s0_setup.py', 's1_data.py', 's2_cnn.py', 's3_text.py', 's4_rag.py', 's5_assistant.py']
HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / 'waste_management_summative.ipynb'

def build():
    merged = HERE / '_merged.py'
    merged.write_text('\n'.join((HERE / s).read_text(encoding='utf-8') for s in SOURCES),
                      encoding='utf-8')
    convert(str(merged), str(OUT))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    build()
