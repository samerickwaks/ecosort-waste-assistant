"""Copy markdown-only edits from build/s*.py into the executed notebook, keeping outputs.

Refuses to run if any code cell differs, because that would mean the stored outputs no
longer match the code and the notebook needs a real re-execution instead.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from py2nb import convert                                                   # noqa: E402

# Deliberately NOT imported from make_notebook: that module builds the notebook at import
# time, which would overwrite the executed copy this script is meant to preserve.
SOURCES = ['s0_setup.py', 's1_data.py', 's2_cnn.py', 's3_text.py', 's4_rag.py', 's5_assistant.py']

NB = pathlib.Path('waste_management_summative.ipynb')
TMP = HERE / '_rebuilt.ipynb'

merged = HERE / '_merged.py'
merged.write_text('\n'.join((HERE / s).read_text(encoding='utf-8') for s in SOURCES),
                  encoding='utf-8')
convert(str(merged), str(TMP))

live = json.loads(NB.read_text(encoding='utf-8'))
fresh = json.loads(TMP.read_text(encoding='utf-8'))

if len(live['cells']) != len(fresh['cells']):
    sys.exit(f'cell count changed ({len(live["cells"])} -> {len(fresh["cells"])}); re-execute')

changed = 0
for i, (a, b) in enumerate(zip(live['cells'], fresh['cells'])):
    if a['cell_type'] != b['cell_type']:
        sys.exit(f'cell {i} changed type; re-execute')
    if a['cell_type'] == 'code':
        if ''.join(a['source']) != ''.join(b['source']):
            sys.exit(f'code cell {i} changed; re-execute instead of syncing')
    elif ''.join(a['source']) != ''.join(b['source']):
        a['source'] = b['source']
        changed += 1

NB.write_text(json.dumps(live, indent=1, ensure_ascii=False), encoding='utf-8')
TMP.unlink()
print(f'{changed} markdown cells updated; code cells and outputs untouched')
