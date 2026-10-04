"""Give every cell an nbformat 4.5 `id`, in place, without touching outputs."""
import json
import pathlib
import sys

NB = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'waste_management_summative.ipynb')
nb = json.loads(NB.read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    cell.setdefault('id', f'cell-{i:03d}')
nb['nbformat_minor'] = max(nb.get('nbformat_minor', 5), 5)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'{NB.name}: ids added to {len(nb["cells"])} cells')
