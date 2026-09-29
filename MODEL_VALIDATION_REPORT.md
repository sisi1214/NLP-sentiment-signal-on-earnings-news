# Earnings-call ML model validation report

## Executive summary

This validation audit used the engineered transcript feature set in the dataset and applied a strict time-ordered benchmark using `TimeSeriesSplit(n_splits=5)`. The goal was to verify temporal integrity, assess the predictive signal from returns following the earnings event, and estimate whether the candidate models can support production deployment.

### Data and label integrity

- Event records: 173 valid observations after excluding rows without a valid `return_classification` label.
- Date coverage: 2016-04-19 to 2020-07-28.
- Target definition: `return_classification` is constructed from the 3-day post-earnings close return (`return_3d`) using the same event alignment used in `feature_engineering.py` where the close on the event date is compared to the close 3 trading sessions later.
- Leakage controls: all model fitting was done inside chronological folds; feature scaling (`StandardScaler`) and imputation were fit on each training fold only, never on the validation fold.
- Result: no feature-level leakage from future returns was used to construct the model inputs.

## Benchmark table (out-of-sample, time-series validation)

```text
Model              | ROC-AUC | Accuracy | Precision | Recall | Log Loss
-------------------|---------|----------|-----------|--------|---------
LogisticRegression | 0.517   | 0.586    | 0.626     | 0.654  | 0.861   
XGBClassifier      | 0.482   | 0.500    | 0.567     | 0.615  | 1.065   
LGBMClassifier     | 0.551   | 0.507    | 0.468     | 0.536  | 0.851   
Majority baseline  | 0.500   | 0.571    | 0.571     | 1.000  | 0.683   
```

### Interpretation

- LightGBM produced the best ROC-AUC at 0.551, but the margin over the majority baseline is small and the model is not consistently predictive across folds.
- The best-performing model remains close to a coin-flip signal, which indicates that the current transcript feature set has limited incremental predictive power for the 3-day return direction.
- Log loss and recall values remain weak enough that the models should not be treated as production-ready without additional feature engineering or a stronger market alignment target.

## SHAP feature attribution

The top OOS importance signals for the best-performing model are:

- Overall_Fog_Index: 0.7598
- Sentiment_Delta_Positive: 0.5766
- Overall_TTR: 0.4906
- Total_Word_Count: 0.4845
- FinBERT_Neutral_QA: 0.4636
- FinBERT_Negative_QA: 0.3401
- FinBERT_Positive_QA: 0.3309
- QA_Word_Count: 0.2702

The strongest influences are concentrated in the uncertainty and sentiment blocks, which is consistent with market intuition: higher uncertainty and negative QA sentiment are expected to depress future returns when the earnings narrative is weak or ambiguous.

## Financial strategy diagnostics

Using an OOS long/short quantile signal based on predicted probabilities:

- Sharpe ratio: 0.940
- Maximum drawdown: 0.426
- Win rate: 0.212
- Annualized return proxy: 0.049

The quantile strategy does not show enough persistence to justify live deployment, and the drawdown profile indicates the signal is weakly monetizable in its current form.

## Deployment recommendation

The models should be treated as research-stage benchmarks rather than production models. The key issues are:

1. weak out-of-sample discrimination relative to the majority class;
2. materially unstable feature importance across time;
3. limited strategy-level Sharpe and high drawdown risk.

Before production deployment, the team should update the target definition, test T+1 and T+5 return windows, enrich the feature set with earnings guidance, replay data, and re-run the validation using a more realistic event-time train/test split and a stricter embargo between transcript release and return measurement.
