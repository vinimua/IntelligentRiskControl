import sys
sys.path.insert(0, 'c:/Users/Vinim/Desktop/RiskItem')

# Directly test what build_html produces for a simple case
from apps.modelops_api.routers.dashboard import build_html
import json

test_data = json.dumps({"models": [], "window_metrics": {}, "drift_top": {}, "quality_top": {}, "pipeline_steps": [], "window_timeline": [], "total_metrics": 0, "latest_run_id": None})
html = build_html(test_data)

# Find toggleDiagRC
idx = html.find('toggleDiagRC')
if idx > 0:
    snippet = html[idx-10:idx+80]
    print('In served HTML:')
    print(repr(snippet))
    print()
    print('Hex:', snippet.encode().hex(' '))
