import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from structure import StockMarketModel
from parameter import NUM_AGENTS, INITIAL_PRICE, MOMENTUM_THRESHOLD, VOLUME_THRESHOLD, MAX_STEPS

model = StockMarketModel(NUM_AGENTS, INITIAL_PRICE, MOMENTUM_THRESHOLD, VOLUME_THRESHOLD, MAX_STEPS)

for _ in range(MAX_STEPS):
    model.step()

data = model.datacollector.get_model_vars_dataframe()
plt.figure(figsize=(8, 4))
plt.plot(data["Stock Price"], label="Stock Price", color='blue', linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.title("Stock Price Over Time")
plt.legend()
plt.show()

agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
agent_data['id_strat'] = agent_data['AgentID'].astype(str) + '_' + agent_data['Strategy'].str[0:2]
holdings_data = agent_data.pivot(index="id_strat", columns="Step", values="Wealth")
plt.figure(figsize=(10, 6))
sns.heatmap(holdings_data, cmap="coolwarm", linewidths=0.5)
plt.xlabel("Time Step")
plt.ylabel("Agent ID")
plt.title("Wealth Heatmap Over Time")
plt.show()



plt.figure(figsize=(8, 4))
trading_volumes = [random.randint(50, 200) for _ in range(MAX_STEPS)]
plt.plot(trading_volumes, label="Trading Volume", color='orange', linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Trading Volume")
plt.title("Trading Volume Over Time")
plt.legend()
plt.show()


plt.figure(figsize=(8, 4))
market_sentiments = [random.uniform(-1, 1) for _ in range(MAX_STEPS)]
plt.plot(market_sentiments, label="Market Sentiment", color='red', linewidth=2)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Time Step")
plt.ylabel("Market Sentiment")
plt.title("Market Sentiment Over Time")
plt.legend()
plt.show()



plt.figure(figsize=(8, 4))
plt.scatter(market_sentiments, data["Stock Price"], color='blue', alpha=0.6)
plt.xlabel("Market Sentiment")
plt.ylabel("Stock Price")
plt.title("Market Sentiment vs. Stock Price")
plt.axhline(y=data["Stock Price"].mean(), color='gray', linestyle="--", label="Avg Stock Price")
plt.axvline(x=0, color='black', linestyle="--", label="Neutral Sentiment")
plt.legend()
plt.show()
