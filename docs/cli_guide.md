# MANGLAR — CLI Operations Guide

Everything you need to operate this repository from the command line,
locally and remotely. No GUI required at any step.

---

## 1. First-time setup on a new local machine

```bash
# Clone the repo (replace with your actual GitHub URL)
git clone https://github.com/YOUR_USERNAME/manglar.git
cd manglar

# Run the setup script
bash setup.sh

# Activate the virtual environment
source .venv/bin/activate

# Verify config loads
python src/utils/config_loader.py
```

---

## 2. Create the GitHub repository (run once, on day one)

```bash
# Make sure GitHub CLI is installed: https://cli.github.com/
# Authenticate if not already done
gh auth login

# Create repo (private for now — make public at submission)
gh repo create manglar \
  --private \
  --description "Mangrove canopy degradation and fishing pressure — Reentrâncias Maranhenses" \
  --clone

# Or if you already have the local folder and want to push:
cd manglar
git init
git add .
git commit -m "chore: initial project structure"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/manglar.git
git push -u origin main
```

---

## 3. Daily Git workflow (feature branch model)

```bash
# Start a new task — always branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/zone3-gee-composites

# Work, then stage and commit
git add src/gee/zone3_s2_composites.js
git commit -m "feat(gee): add Zone 3 Sentinel-2 monthly composite script"

# Push branch to remote
git push -u origin feature/zone3-gee-composites

# Open pull request via CLI
gh pr create \
  --base develop \
  --title "feat(gee): Zone 3 S2 composites" \
  --body "Adds GEE script for Zone 3 calibration composites. Closes #ISSUE_NUMBER"

# After PR review, merge and clean up
git checkout develop
git pull origin develop
git branch -d feature/zone3-gee-composites
```

---

## 4. Branch naming convention

| Prefix | Use |
|--------|-----|
| `feature/` | New pipeline step, model, or script |
| `fix/` | Bug fix in existing code |
| `data/` | Data acquisition or processing update |
| `paper/` | Manuscript edits |
| `docs/` | Documentation only |

---

## 5. Commit message convention (Conventional Commits)

```
type(scope): short description

Body (optional): what changed and why.
Footer: Closes #ISSUE_NUMBER
```

Types: `feat`, `fix`, `data`, `docs`, `test`, `chore`, `refactor`

Scopes: `gee`, `pipeline1`, `pipeline2`, `pipeline3`, `stl`, `nbeats`, `presto`, `regression`, `paper`

Examples:
```bash
git commit -m "feat(gee): add Sentinel-1 VV/VH monthly composites for Zone 1"
git commit -m "fix(nbeats): correct seasonal stack degree parameter"
git commit -m "data(gfw): download and validate 2017-2024 fishing effort rasters"
git commit -m "paper(methods): draft STL vs N-BEATS comparison paragraph"
```

---

## 6. Syncing a Colab session with the repository

```python
# In a Colab cell — mount Drive and pull latest code
from google.colab import drive
drive.mount('/content/drive')

import subprocess
# Clone or pull the repo into Drive
repo_path = '/content/drive/MyDrive/manglar'
import os
if not os.path.exists(repo_path):
    subprocess.run(['git', 'clone',
                    'https://github.com/YOUR_USERNAME/manglar.git',
                    repo_path], check=True)
else:
    subprocess.run(['git', '-C', repo_path, 'pull'], check=True)

# Install dependencies
subprocess.run(['bash', f'{repo_path}/setup.sh', '--colab'], check=True)
```

---

## 7. Tagging milestones (use for paper-trackable snapshots)

```bash
# Tag at key reproducibility checkpoints
git tag -a v0.1-zone3-calibration -m "Zone 3 calibration baseline complete"
git tag -a v0.2-pipeline1-complete -m "Zone 1 full pipeline outputs produced"
git tag -a v0.3-models-ablation -m "STL/N-BEATS/Presto ablation table complete"
git tag -a v0.4-regression-final -m "Spatial regression results finalised"
git tag -a v1.0-submission -m "Manuscript submitted to JSTARS"

# Push tags
git push origin --tags

# Create a GitHub Release at submission
gh release create v1.0-submission \
  --title "JSTARS submission — August 2026" \
  --notes "Code and outputs at time of first submission."
```

---

## 8. Running tests locally

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## 9. Useful inspection commands

```bash
# Check repo status
git status

# See full tree of tracked files
git ls-files | head -60

# Show last 10 commits, one line each
git log --oneline -10

# Show what changed in last commit
git show --stat

# Check remote branches
git branch -r

# List all tags
git tag -l
```

---

## 10. Protected branch rules (set on GitHub once repo is created)

On GitHub → Settings → Branches → Add branch protection rule for `main`:

- Require pull request before merging
- Require status checks to pass (CI workflow must be green)
- Do not allow force pushes

`develop` is the integration branch. `main` is only updated at milestone tags.
