"""Execute the deliverable notebook in place, with a generous per-cell timeout."""
import sys
import time
import pathlib

import nbformat
from nbclient import NotebookClient

NB = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'waste_management_summative.ipynb')
TIMEOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 5400          # seconds per cell

nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=TIMEOUT, kernel_name='python3',
                        allow_errors=False, resources={'metadata': {'path': str(NB.parent)}})

t0 = time.time()
n_code = sum(c.cell_type == 'code' for c in nb.cells)
print(f'executing {NB.name}: {n_code} code cells, {TIMEOUT}s cell timeout', flush=True)

with client.setup_kernel():
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        label = ''.join(cell.source).strip().split('\n')[0][:72]
        t1 = time.time()
        try:
            client.execute_cell(cell, index)
        except Exception as exc:
            print(f'FAILED cell {index}: {label}\n{type(exc).__name__}: {exc}', flush=True)
            nbformat.write(nb, NB)
            raise
        print(f'  [{index:3d}] {time.time() - t1:7.1f}s  {label}', flush=True)

nbformat.write(nb, NB)
print(f'done in {(time.time() - t0) / 60:.1f} minutes -> {NB}')
