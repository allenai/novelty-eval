# Novelty signal annotation

Checks the quality of the novelty signals extracted from reviews in `data/human-only`.

1. **Sample** 25 positive + 25 negative signals, one per idea:
   ```
   python -m novelty_eval.signal_annotation.sample_signals
   ```
   This writes `output/signal_annotation/form.html` (for the annotator) and `key.csv` (keep private).
2. **Annotate** by opening `form.html` in a browser and clicking *Download answers* at the end.
3. **Report**: put `answers.csv` in `output/signal_annotation/` and run `analyze_annotations.ipynb`.

Metrics: extraction precision, human–model Cohen's κ on polarity, and % of signals whose judged contribution is missing from the abstract.
