"""Convert a percent-format .py script into a .ipynb notebook.

Cell markers:
    # %% [markdown]   -> markdown cell (body lines are comments, leading '# ' stripped)
    # %%              -> code cell
"""
import json, sys, re

def convert(src_path, out_path):
    lines = open(src_path, encoding='utf-8').read().split('\n')
    cells, cur, kind = [], [], None

    def flush():
        if kind is None:
            return
        body = '\n'.join(cur).strip('\n')
        if kind == 'markdown':
            body = '\n'.join(re.sub(r'^# ?', '', ln) for ln in body.split('\n'))
        if not body.strip():
            return
        cell_id = f'cell-{len(cells):03d}'
        if kind == 'markdown':
            cells.append({'cell_type': 'markdown', 'id': cell_id, 'metadata': {},
                          'source': body.split('\n')})
        else:
            cells.append({'cell_type': 'code', 'id': cell_id, 'metadata': {},
                          'execution_count': None, 'outputs': [], 'source': body.split('\n')})

    for ln in lines:
        if ln.startswith('# %%'):
            flush()
            kind = 'markdown' if 'markdown' in ln else 'code'
            cur = []
        else:
            cur.append(ln)
    flush()

    # join source lines back with newlines (nbformat wants trailing \n on all but last)
    for c in cells:
        s = c['source']
        c['source'] = [l + '\n' for l in s[:-1]] + [s[-1]]

    nb = {'cells': cells,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.11.7'}},
          'nbformat': 4, 'nbformat_minor': 5}
    json.dump(nb, open(out_path, 'w', encoding='utf-8'), indent=1)
    print(f'{out_path}: {len(cells)} cells '
          f'({sum(c["cell_type"]=="code" for c in cells)} code, '
          f'{sum(c["cell_type"]=="markdown" for c in cells)} markdown)')

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
