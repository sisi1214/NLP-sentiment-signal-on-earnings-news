# First model result

Date: 2026-09-29

This note captures the first fitted logistic-regression model result for the earnings-transcript signal, as it was used in the validation workflow before deciding whether to continue with deeper feature engineering or move on.

## Model setup

- Target: 3-day post-earnings return direction (binary classification)
- Model: LogisticRegression
- Preprocessing: median imputation + StandardScaler
- Regularization / class handling: class_weight='balanced', random_state=42, max_iter=5000
- Time-aware validation was used to avoid leakage, with the model fit on training folds only.

## Current coefficients

Intercept: 0.03148225152602623

- LM_Freq_Constraining: -0.423304
- Prepared_TTR: 0.329920
- Overall_Fog_Index: -0.309501
- Prepared_Word_Count: -0.247880
- Prepared_Fog_Index: 0.225562
- LM_Freq_Litigious: 0.217079
- QA_LM_Freq_Litigious: 0.183912
- QA_LM_Freq_Constraining: -0.154227
- LM_Freq_Negative: 0.153372
- FinBERT_Negative_Prepared: 0.153372
- QA_LM_Freq_Negative: -0.150156
- FinBERT_Negative_QA: -0.150156
- QA_Word_Count: 0.102855
- QA_Uncertainty_Ratio: -0.072725
- QA_LM_Freq_Uncertainty: -0.072725
- QA_Fog_Index: -0.059671
- Sentiment_Delta_Positive: 0.059270
- LM_Freq_Positive: -0.056220
- FinBERT_Positive_Prepared: -0.056220
- LM_Freq_Uncertainty: -0.056172

## Interpretation

- Positive coefficient: pushes the model toward the positive-return class.
- Negative coefficient: pushes the model toward the negative-return class.
- The largest magnitudes are concentrated in language-constraint, complexity, and tone-related features, suggesting some relationship between transcript style and return direction, but not enough to claim a robust production signal.

## Conclusion

This was a good research-stage attempt: it produced a signal with some explanatory structure, but the overall predictive edge was weak relative to the baseline and not strong enough for production deployment. This should be treated as a valid first pass rather than a deployable model.
