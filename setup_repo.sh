#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Sets up this project as a git repo with a clean, phase-by-phase commit
# history, then (optionally) pushes it to a GitHub repo you've already
# created. Run this from inside the fake-job-posting-detection/ folder.
#
# IMPORTANT: commits only count on your GitHub contribution heatmap if the
# git author email below matches an email verified on your GitHub account.
# Edit the two lines below before running.
# ---------------------------------------------------------------------------
set -e

GIT_NAME="Anshul"
GIT_EMAIL="vermaanshul4321@gmail.com"

if [ ! -d .git ]; then
  git init
  git branch -M main
fi

git config user.name  "$GIT_NAME"
git config user.email "$GIT_EMAIL"

echo "Phase 1/8: project scaffolding"
git add .gitignore README.md requirements.txt data/README.md
git commit -m "Initial commit: project scaffolding, README, requirements"

echo "Phase 2/8: data loading & preprocessing"
git add src/__init__.py src/preprocessing.py
git commit -m "Add data loading and text preprocessing module"

echo "Phase 3/8: feature engineering"
git add src/features.py
git commit -m "Add TF-IDF / CountVectorizer feature engineering module"

echo "Phase 4/8: modeling & evaluation"
git add src/modeling.py
git commit -m "Add model definitions, training, and evaluation module"

echo "Phase 5/8: prediction system"
git add src/predict.py
git commit -m "Add final prediction system for classifying new postings"

echo "Phase 6/8: end-to-end pipeline script"
git add src/train_pipeline.py
git commit -m "Add end-to-end CLI training pipeline"

echo "Phase 7/8: notebook + figures"
git add notebooks/ figures/
git commit -m "Add full analysis notebook (EDA, modeling, evaluation, interpretation) with figures"

echo "Phase 8/8: demo app + reports"
git add app/ reports/
git commit -m "Add Streamlit demo app, project report, and slide deck"

echo ""
echo "Done. Local commit history:"
git log --oneline

echo ""
echo "Next steps:"
echo "  1. Create an empty repo on GitHub (no README/license, so there's no conflict)."
echo "  2. git remote add origin https://github.com/<your-username>/<repo-name>.git"
echo "  3. git push -u origin main"
