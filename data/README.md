# Dataset

This project uses the **EMSCAD** (Employment Scam Aegean Dataset), distributed
publicly on Kaggle as *"Real / Fake Job Posting Prediction"*.

The raw CSV isn't committed to this repo (dataset redistribution + repo size),
so download it yourself:

1. Go to: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
2. Download `fake_job_postings.csv` (free Kaggle account required).
3. Place it here as `data/fake_job_postings.csv`.

Then run:

```bash
python -m src.train_pipeline --data data/fake_job_postings.csv
```
