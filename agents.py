from mesa import Agent
import random

class TraderAgent(Agent):
    def __init__(self, unique_id, model):
        super(TraderAgent, self).__init__(model)
        self.wealth = 1000  # Starting cash
        self.holding = 0  # Stocks held
        self.strategy = random.choice(["momentum", "contrarian", "mean_reversion"])
        self.max_holdings = 5  # Limit stock holdings
        self.min_wealth_threshold = 100  # Prevents trading when very low on cash

    def step(self):
        """Decide whether to buy, sell, or hold based on strategy and additional factors."""
        price_change = 0
        price_momentum = self.model.current_price - self.model.initial_price
        trading_volume = self.model.trading_volume
        market_sentiment = self.model.market_sentiment
        decision = "hold"

        if self.strategy == "momentum":
            if price_momentum > self.model.momentum_threshold and self.holding < self.max_holdings and trading_volume > self.model.volume_threshold and market_sentiment > 0:
                self.buy()
                price_change += 1
                decision = "buy"
            elif price_momentum < -self.model.momentum_threshold and self.holding > 0:
                self.sell()
                price_change -= 1
                decision = "sell"

        elif self.strategy == "contrarian":
            if price_momentum > self.model.momentum_threshold and self.holding > 0:
                self.sell()
                price_change -= 1
                decision = "sell"
            elif price_momentum < -self.model.momentum_threshold and self.holding < self.max_holdings:
                self.buy()
                price_change += 1
                decision = "buy"

        elif self.strategy == "mean_reversion":
            if abs(price_momentum) > self.model.momentum_threshold and self.holding > 0:
                self.sell()
                price_change -= 1
                decision = "sell"
            elif self.holding < self.max_holdings:
                self.buy()
                price_change += 1
                decision = "buy"

        # Exit Strategy Fix
        if self.holding > self.max_holdings:
            self.sell()
            price_change -= 1
            decision = "exit-sell"
        elif self.wealth < self.min_wealth_threshold and self.holding > 0:
            self.sell()
            price_change -= 1
            decision = "exit-sell"

        self.model.update_stock_price(price_change)

    def buy(self):
        """Buy stocks if the agent has enough cash and has not exceeded max holdings."""
        if self.wealth >= self.model.current_price and self.holding < self.max_holdings:
            self.holding += 1
            self.model.buy_volume += 1
            self.model.trading_volume += 1
            self.wealth -= self.model.current_price

    def sell(self):
        """Sell stocks if the agent is holding any."""
        if self.holding > 0:
            self.holding -= 1
            self.model.sell_volume += 1
            self.model.trading_volume += 1  
            self.wealth += self.model.current_price
