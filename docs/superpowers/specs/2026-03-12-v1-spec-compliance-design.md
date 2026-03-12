# Enterprise Vision AI — V1.0 Spec Compliance & CI Fix Design

> **Date:** 2026-03-12
> **Status:** Approved
> **Scope:** V1.0 — make CI green, clean file structure, align with SPEC §3.1

---

## 1. Problem Summary

The project is in a broken state due to three compounding issues:

1. **`.gitignore` blocks Python package directories:** `.gitignore` contains bare `models/` and `datasets/` patterns (meant for ML weights and raw datasets) which also match `src/enterprise_vision_ai/models/` and `src/enterprise_vision_ai/datasets/` — so these Python packages are never committed to git.
2. **GitHub `__init__.py` imports missing modules:** The committed `src/enterprise_vision_ai/__init__.py` imports from `models.model_manager` and `datasets.data_loader`, but those files are gitignored and not on GitHub. Every `import enterprise_vision_ai` fails in CI.
3. **Three CI workflows have individual bugs:** Unit Tests fail due to #2 above; Release has a Python operator bug and wrong step ID reference; HF Deploy has a dead import.

---

## 2. Root Cause Chain

```
.gitignore has `models/` and `datasets/`
  → src/enterprise_vision_ai/models/ and datasets/ are gitignored
  → model_manager.py and data_loader.py never pushed to GitHub
  → GitHub __init__.py imports from these missing modules
  → `import enterprise_vision_ai` fails in CI
  → pytest can't collect tests → "No module named 'enterprise_vision_ai.datasets'"
```

---

## 3. All CI Failures

| Workflow | Error | Root Cause |
|----------|-------|-----------|
| CI — Unit Tests | `No module named 'enterprise_vision_ai.datasets'` | `.gitignore` blocks `src/enterprise_vision_ai/models/` and `datasets/`; GitHub `__init__.py` imports from them |
| Release | `TypeError: unsupported operand type(s) for >>: 'str' and 'str'` | Python `>>` bitshift used instead of writing to `$GITHUB_OUTPUT` |
| Release | `release_notes` output always empty | Job-level output references `steps.version.outputs.release_notes` but release_notes is written in step `id: notes` |
| HF Deploy | `ImportError: cannot import name 'create_space'` | Dead import — `create_space` is imported but never called; `huggingface_hub` removed it |

---

## 4. File Changes Required

### 4.1 Fix .gitignore

Change bare directory patterns to root-anchored patterns so Python packages are not excluded:

```gitignore
# Before (matches anywhere):
models/
datasets/

# After (matches only at repo root):
/models/
/datasets/
```

This allows `src/enterprise_vision_ai/models/` and `src/enterprise_vision_ai/datasets/` to be tracked.

### 4.2 Stage and commit previously-gitignored Python packages

After fixing `.gitignore`, stage these directories (currently gitignored):
```
src/enterprise_vision_ai/models/     (model_manager.py, base.py, yolo_adapter.py,
                                      defect_detector.py, ore_classifier.py, __init__.py)
src/enterprise_vision_ai/datasets/   (data_loader.py, __init__.py)
```

### 4.3 Stage and commit currently-untracked src/ modules

```
src/enterprise_vision_ai/api/
src/enterprise_vision_ai/clients/
src/enterprise_vision_ai/core/
src/enterprise_vision_ai/utils/
```

### 4.4 Commit staged deletions (unstaged, must be staged first with `git rm`)

```
api/          (top-level, 9 files — git rm these)
clients/      (top-level, 3 files — git rm these)
```

Note: Use `git rm api/... clients/...` to stage these deletions before committing.

### 4.5 Fix src/enterprise_vision_ai/__init__.py

The local modified version removed the imports. Restore them properly:

```python
from enterprise_vision_ai.datasets.data_loader import (
    DefectDetectionDataset,
    OreClassificationDataset,
    load_dataset,
)
from enterprise_vision_ai.models.model_manager import ModelManager, load_model

__all__ = [
    "__version__",
    "__author__",
    "ModelManager",
    "load_model",
    "DefectDetectionDataset",
    "OreClassificationDataset",
    "load_dataset",
]
```

Do NOT add try/except guards — if these imports fail, CI should fail loudly.

### 4.6 Fix .github/workflows/release.yml — Two bugs

**Bug 1 — Python `>>` operator (line ~86):**
```python
# Wrong:
print(f"new_version={new}" >> os.environ['GITHUB_OUTPUT'])

# Fix — write to GITHUB_OUTPUT correctly:
with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"new_version={new}\n")
```

**Bug 2 — Wrong step ID for release_notes (line ~34):**
```yaml
# Wrong:
release_notes: ${{ steps.version.outputs.release_notes }}

# Fix — only this line changes; new_version stays on steps.version:
release_notes: ${{ steps.notes.outputs.release_notes }}
```

The `new_version` reference on line 33 (`steps.version.outputs.new_version`) is correct and must NOT be changed.

### 4.7 Fix .github/workflows/deploy-hf.yml

Remove the dead `create_space` import:

```python
# Wrong:
from huggingface_hub import HfApi, create_space

# Fix:
from huggingface_hub import HfApi
```

`upload_folder` is already called correctly via `api.upload_folder(...)` — no functional change needed.

---

## 5. Test Coverage

| File | Issue | Resolution |
|------|-------|-----------|
| `test_datasets.py` | `No module named 'enterprise_vision_ai.datasets'` | Fix .gitignore + commit datasets/ |
| `test_models.py` | Same (package fails to import) | Fix .gitignore + commit models/ |
| `test_backward_compat.py` | `load_model`/`load_dataset` not importable | Resolved by __init__.py fix (§4.5) |
| `test_utils.py` | `services.utils` imports | Already tracked and working locally — no change needed |

---

## 6. Out of Scope (V2.0)

- MLflow, PostgreSQL, Redis, Kubernetes, Auth/RBAC, Training pipeline

---

## 7. Success Criteria

- [ ] `pytest` passes locally with no import errors
- [ ] `.gitignore` uses `/models/` and `/datasets/` (root-anchored)
- [ ] `src/enterprise_vision_ai/models/` and `datasets/` are visible to `git status`
- [ ] CI `Unit Tests` green on Python 3.10, 3.11, 3.12
- [ ] CI `Release` workflow runs without TypeError and release_notes is populated
- [ ] CI `deploy-hf.yml` runs without ImportError
- [ ] Old `api/` and `clients/` top-level directories gone from GitHub
- [ ] `from enterprise_vision_ai import load_model, load_dataset` works
