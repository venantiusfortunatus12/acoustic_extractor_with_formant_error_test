# Midpoint Acoustic Extraction + Formant Audit

Streamlit app that audits 28 Burg LPC parameter combinations and then exports
exactly one acoustic measurement row at the 50% midpoint of each TextGrid
interval.

The internal audit uses dense frames only for quality diagnostics. The final
CSV contains one row per interval with F0, intensity, F1, F2, and F3.

```bash
pip install -r requirements.txt
streamlit run app.py
```

For Streamlit Community Cloud, place these files at the GitHub repository root
and select `app.py` as the main file.
