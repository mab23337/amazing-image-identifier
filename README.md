# Amazing Image Identifier (Team 5)

ITP 258 Raspberry Pi Team Project

## Overview
This project uses an AI vision model (DETR) to analyze an uploaded image and return
a natural-language description of the most prominent object(s).

## Team Workflow
- main: stable branch
- develop: integration branch
- feature/*: individual task branches

## Running Locally (Flask)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_production.txt
python production_app.py

```bash
mkdir -p docs

cat > docs/git-workflow.md <<'EOF'
# Git Workflow Rules

All team members should:

1. Pull latest code from develop
2. Create a feature branch:
   feature/<task-name>
3. Commit changes regularly
4. Open a Pull Request into develop

Do not commit directly to main.
