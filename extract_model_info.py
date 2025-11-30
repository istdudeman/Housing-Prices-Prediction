import json
import sys

with open('Housing_Price_Analysis.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f"Total cells: {len(notebook.get('cells', []))}", file=sys.stderr)

model_cells = []
for i, cell in enumerate(notebook.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if any(keyword in source for keyword in ['RandomForest', 'LinearRegression', 'GradientBoosting', 'XGBoost', '.fit(', 'model =', 'Model()']):
            model_cells.append((i, source))

print(f"\nFound {len(model_cells)} model-related cells", file=sys.stderr)

for i, source in model_cells[:10]:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"CELL {i}:", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    try:
        print(source, file=sys.stderr)
    except:
        print("[Content with encoding issues]", file=sys.stderr)
