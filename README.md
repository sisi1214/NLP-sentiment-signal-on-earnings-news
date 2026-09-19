# NLP-sentiment-signal-on-earnings-news
Score sentiment from earnings call transcripts or news headlines and test if it predicts short-term returns.

idea:


Score sentiment from earnings call transcripts  and test if it predicts short-term returns.

Use a free dataset of earnings call transcripts 

Apply a sentiment model (FinBERT or similar) 
    TF-IDF and Count Vectorization

Correlate sentiment shifts with subsequent stock returns
    Classification:closing price 1 day, 7 days and 28 days later. => buy hold sell ratin
    buy:reutrn>2% 
    hold: -2%<return < 2%
    sell: return < -2%
    Logistric regression, NLP
    Sentiment analysis: VADER Sentiment Intensity Analyzer (SIA) Polarity Scoring Hu Liu (HL) and Loughran McDonald  (LM) word lists

    

Deploy using FastAPI , so that future user can deploy it

limitations (small samples, overfitting risk, why a "signal" might be noise)