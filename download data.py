import yfinance as yf

tickers = ["AAPL", "AMD", "AMZN", "ASML", "CSCO", "GOOGL", "INTC", "MSFT", "MU", "NVDA"]

data = yf.download(tickers, start="2016-04-01", end="2020-08-01", group_by="ticker")

# Save each ticker to its own CSV
for t in tickers:
    data[t].to_csv(f"{t}_2016_2020.csv")

print("Done. Downloaded:", tickers)