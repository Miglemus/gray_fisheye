import sys

sys.meta_path = [
    finder
    for finder in sys.meta_path
    if not finder.__class__.__module__.startswith("_editable_skbc_gray")
]
