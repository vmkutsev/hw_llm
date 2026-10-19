"""Execute the completed notebook on a CUDA machine and preserve partial output on failure."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/sft_function_calling_hw_executed.ipynb'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    notebook = nbformat.read(root/'sft_function_calling_hw.ipynb', as_version=4)
    # A partial failed rerun must not retain results from the published run.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata.pop('widgets', None)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    def progress(cell, cell_index, **kwargs):
        nbformat.write(notebook, output)
        print('Completed cell', cell_index, flush=True)
    client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                            resources={'metadata':{'path':str(root)}}, on_cell_executed=progress)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    print('Saved:', output)


if __name__ == '__main__':
    main()
