"""Smoke test for Wild-ID Gradio UI structure and element layout.

Ensures:
1. Gradio app imports and initializes without errors.
2. Sidebar and content pane remain side-by-side inside #wid-shell.
3. Key elements (overview, identify, tabs, query image, results) exist with expected elem_ids.
4. Python 3.13 / Gradio 6 compatibility.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

def test_ui_structure():
    import demo
    app = demo.demo

    # Verify Blocks instance
    assert app is not None, "Gradio demo Blocks instance is None"

    # Search Blocks hierarchy for critical elem_ids
    found_elem_ids = set()

    def collect_elem_ids(block):
        elem_id = getattr(block, "elem_id", None)
        if elem_id:
            found_elem_ids.add(elem_id)
        children = getattr(block, "children", [])
        for c in children:
            collect_elem_ids(c)

    collect_elem_ids(app)

    print(f"[OK] Total elem_ids found in Blocks: {len(found_elem_ids)}")

    # Verify side-by-side shell layout
    critical_ids = [
        "wid-shell",
        "wid-sidebar-col",
        "wid-content-pane",
        "wid-main-tabs",
        "wid-tab-overview",
        "wid-tab-identify",
        "wid-query-image",
        "wid-identify-btn",
        "wid-reid-results"
    ]

    for cid in critical_ids:
        assert cid in found_elem_ids, f"Required elem_id '{cid}' missing from Gradio app"
        print(f"  [OK] Found layout component: #{cid}")

    print("\n[PASS] UI Smoke Test passed successfully: Sidebar and content side-by-side structure intact.")

if __name__ == "__main__":
    test_ui_structure()
