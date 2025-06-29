import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Define the agent classes

class StockMarket(mesa.Model):
    """A model representing a stock market with different types of traders."""
    
    def __init__(self, N_momentum=50, N_contrarian=20, N_mean_reversion=30, 
                 initial_price=100, volatility=0.01, volume_base=1000,
                 sentiment_base=0.5, sentiment_volatility=0.1, momentum_threshold=0.05):
        super().__init__()
        self.num_agents = N_momentum + N_contrarian + N_mean_reversion
        # Using RandomActivation instead of time
        self.agents.shuffle_do("step")
        
        # Market parameters
        self.current_price = initial_price
        self.previous_prices = [initial_price]  # History of prices
        self.price_momentum = 0  # Current momentum
        self.volatility = volatility
        self.volume = volume_base
        self.sentiment = sentiment_base  # Market sentiment (0-1)
        self.sentiment_volatility = sentiment_volatility
        self.momentum_threshold = momentum_threshold
        self.is_momentum_rally = False
        
        # Create agents
        for i in range(N_momentum):
            a = MomentumTrader(i, self)
            self.schedule.add(a)
            
        for i in range(N_momentum, N_momentum + N_contrarian):
            a = ContrarianTrader(i, self)
            self.schedule.add(a)
            
        for i in range(N_momentum + N_contrarian, self.num_agents):
            a = MeanReversionTrader(i, self)
            self.schedule.add(a)
        
        # Collect data
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Price": lambda m: m.current_price,
                "Volume": lambda m: m.volume,
                "Sentiment": lambda m: m.sentiment,
                "Momentum": lambda m: m.price_momentum,
                "Is_Momentum_Rally": lambda m: 1 if m.is_momentum_rally else 0
            },
            agent_reporters={
                "Position": lambda a: a.position
            }
        )
    
    def update_market_conditions(self):
        # Update price momentum (rate of change)
        if len(self.previous_prices) >= 5:
            self.price_momentum = (self.current_price - self.previous_prices[-5]) / self.previous_prices[-5]
        
        # Determine if we're in a momentum rally
        self.is_momentum_rally = self.price_momentum > self.momentum_threshold
        
        # Update market sentiment (could be affected by external news)
        sentiment_change = np.random.normal(0, self.sentiment_volatility)
        self.sentiment = max(0, min(1, self.sentiment + sentiment_change))
        
        # Volume can fluctuate based on market conditions
        if self.is_momentum_rally:
            volume_multiplier = 1.2  # Higher volume during rallies
        else:
            volume_multiplier = 1.0
        
        self.volume = max(100, self.volume * volume_multiplier + np.random.normal(0, self.volume/10))
    
    def step(self):
        # Update market conditions
        self.update_market_conditions()
        
        # Run agent actions
        self.schedule.step()
        
        # Store the current price in history
        self.previous_prices.append(self.current_price)
        
        # Collect data
        self.datacollector.collect(self)


class Trader(mesa.Agent):
    """Base class for all traders"""
    
    def __init__(self, unique_id, model, risk_tolerance=0.5):
        super().__init__(model)
        self.position = 0  # Net position: positive for long, negative for short
        self.risk_tolerance = risk_tolerance
        self.cash = 10000  # Starting cash
        self.portfolio_value = self.cash
    
    def calculate_order_size(self):
        # Base order size depends on risk tolerance and portfolio size
        max_order = self.portfolio_value * self.risk_tolerance / self.model.current_price
        return max(1, int(max_order))
    
    def update_portfolio(self):
        # Update portfolio value
        self.portfolio_value = self.cash + (self.position * self.model.current_price)


class MomentumTrader(Trader):
    """Traders who follow momentum - buy when price is rising, sell when falling"""
    
    def __init__(self, unique_id, model, momentum_sensitivity=0.5):
        super().__init__(unique_id, model)
        self.momentum_sensitivity = momentum_sensitivity  # How sensitive to momentum (0-1)
    
    def step(self):
        # Buy when momentum is positive, sell when negative
        momentum = self.model.price_momentum
        sentiment_factor = self.model.sentiment  # Higher sentiment increases buying
        
        # Decision logic
        decision_factor = momentum * self.momentum_sensitivity + (sentiment_factor - 0.5) * 0.2
        order_size = self.calculate_order_size()
        
        if decision_factor > 0.02:  # Strong positive signal
            # Buy
            if self.cash >= order_size * self.model.current_price:
                self.position += order_size
                self.cash -= order_size * self.model.current_price
                self.model.current_price *= (1 + 0.001 * order_size / self.model.volume)  # Price impact
        
        elif decision_factor < -0.02:  # Strong negative signal
            # Sell
            if self.position > 0:
                sell_size = min(self.position, order_size)
                self.position -= sell_size
                self.cash += sell_size * self.model.current_price
                self.model.current_price *= (1 - 0.001 * sell_size / self.model.volume)  # Price impact
        
        # If momentum rally bursts, exit positions more aggressively
        if self.model.is_momentum_rally and self.model.price_momentum < 0 and self.position > 0:
            sell_size = self.position  # Sell everything
            self.position = 0
            self.cash += sell_size * self.model.current_price
            self.model.current_price *= (1 - 0.002 * sell_size / self.model.volume)  # Bigger price impact
        
        self.update_portfolio()


class ContrarianTrader(Trader):
    """Traders who go against the trend - sell when price is rising too fast, buy on dips"""
    
    def step(self):
        # Contrarians are more active during extreme momentum
        momentum = self.model.price_momentum
        sentiment_factor = self.model.sentiment
        
        # They think market is too enthusiastic
        overvalued = momentum > 0.1 or sentiment_factor > 0.8
        undervalued = momentum < -0.1 or sentiment_factor < 0.2
        
        order_size = self.calculate_order_size()
        
        if overvalued and self.position >= 0:  # Market seems too high
            # Sell or short
            sell_size = min(self.position, order_size) if self.position > 0 else order_size
            self.position -= sell_size
            self.cash += sell_size * self.model.current_price
            self.model.current_price *= (1 - 0.001 * sell_size / self.model.volume)
        
        elif undervalued and self.cash > 0:  # Market seems too low
            # Buy
            buy_size = min(order_size, int(self.cash / self.model.current_price))
            if buy_size > 0:
                self.position += buy_size
                self.cash -= buy_size * self.model.current_price
                self.model.current_price *= (1 + 0.001 * buy_size / self.model.volume)
        
        self.update_portfolio()


class MeanReversionTrader(Trader):
    """Traders who believe prices revert to a mean value"""
    
    def __init__(self, unique_id, model, window_size=20):
        super().__init__(unique_id, model)
        self.window_size = window_size  # Window to calculate the mean
    
    def step(self):
        # Calculate mean price if we have enough history
        if len(self.model.previous_prices) >= self.window_size:
            mean_price = sum(self.model.previous_prices[-self.window_size:]) / self.window_size
            
            # Calculate how far current price is from mean
            deviation = (self.model.current_price - mean_price) / mean_price
            
            order_size = self.calculate_order_size()
            
            # If price is significantly above mean, sell
            if deviation > 0.05 and self.position > 0:  
                sell_size = min(self.position, order_size)
                self.position -= sell_size
                self.cash += sell_size * self.model.current_price
                self.model.current_price *= (1 - 0.001 * sell_size / self.model.volume)
            
            # If price is significantly below mean, buy
            elif deviation < -0.05 and self.cash > 0:
                buy_size = min(order_size, int(self.cash / self.model.current_price))
                if buy_size > 0:
                    self.position += buy_size
                    self.cash -= buy_size * self.model.current_price
                    self.model.current_price *= (1 + 0.001 * buy_size / self.model.volume)
        
        self.update_portfolio()


# Run the model and visualize results
def run_model(steps=200):
    # Create model with parameters
    model = StockMarket(
        N_momentum=50,  # Momentum traders
        N_contrarian=20,  # Contrarian traders
        N_mean_reversion=30,  # Mean reversion traders
        initial_price=100,
        volatility=0.01,
        volume_base=1000,
        sentiment_base=0.5,  # Neutral sentiment
        sentiment_volatility=0.05,
        momentum_threshold=0.05  # Threshold to identify momentum rally
    )
    
    # Run for specified number of steps
    for i in range(steps):
        model.step()
    
    # Get the data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    return model, model_data, agent_data


def plot_results(model_data):
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot price
    axes[0].plot(model_data.index, model_data["Price"], label="Stock Price")
    axes[0].set_ylabel("Price")
    axes[0].set_title("Stock Price Over Time")
    axes[0].legend()
    
    # Plot momentum
    axes[1].plot(model_data.index, model_data["Momentum"], label="Price Momentum")
    axes[1].set_ylabel("Momentum")
    axes[1].set_title("Price Momentum Over Time")
    axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[1].legend()
    
    # Plot volume
    axes[2].plot(model_data.index, model_data["Volume"], label="Trading Volume")
    axes[2].set_ylabel("Volume")
    axes[2].set_title("Trading Volume Over Time")
    axes[2].legend()
    
    # Plot sentiment and momentum rally indicator
    axes[3].plot(model_data.index, model_data["Sentiment"], label="Market Sentiment")
    axes[3].set_ylabel("Sentiment")
    axes[3].set_title("Market Sentiment and Momentum Rally Periods")
    
    # Highlight momentum rally periods
    rally_periods = model_data["Is_Momentum_Rally"]
    axes[3].fill_between(model_data.index, 0, 1, where=rally_periods==1, 
                        color='green', alpha=0.3, transform=axes[3].get_xaxis_transform(),
                        label="Momentum Rally")
    axes[3].legend()
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.xlabel("Time Steps")
    plt.show()


def analyze_rally_burst(model_data):
    """Analyze when momentum rallies burst"""
    # Identify rally periods
    rally_starts = []
    rally_ends = []
    in_rally = False
    
    for i, is_rally in enumerate(model_data["Is_Momentum_Rally"]):
        if not in_rally and is_rally == 1:
            rally_starts.append(i)
            in_rally = True
        elif in_rally and is_rally == 0:
            rally_ends.append(i)
            in_rally = False
    
    # If we end in a rally, close it
    if in_rally:
        rally_ends.append(len(model_data) - 1)
    
    # Analyze each rally
    rally_analysis = []
    for start, end in zip(rally_starts, rally_ends):
        rally_duration = end - start
        max_price = model_data["Price"][start:end+1].max()
        max_price_idx = model_data["Price"][start:end+1].idxmax()
        price_increase = max_price / model_data["Price"][start] - 1
        
        # Check if there was a significant drop after the peak
        if end > max_price_idx:
            post_peak_drop = 1 - model_data["Price"][end] / max_price
        else:
            post_peak_drop = 0
        
        # Add to analysis
        rally_analysis.append({
            "start": start,
            "end": end,
            "duration": rally_duration,
            "max_price": max_price,
            "price_increase_pct": price_increase * 100,
            "post_peak_drop_pct": post_peak_drop * 100,
            "burst": post_peak_drop > 0.05  # Consider it burst if dropped more than 5%
        })
    
    return pd.DataFrame(rally_analysis)


# Main execution function
def main():
    # Run the model
    model, model_data, agent_data = run_model(steps=200)
    
    # Plot the results
    plot_results(model_data)
    
    # Analyze when rallies burst
    rally_analysis = analyze_rally_burst(model_data)
    
    print("Rally Analysis:")
    print(rally_analysis)
    
    # Additional analysis
    if not rally_analysis.empty:
        burst_rallies = rally_analysis[rally_analysis["burst"] == True]
        print(f"\nNumber of rallies: {len(rally_analysis)}")
        print(f"Number of burst rallies: {len(burst_rallies)}")
        
        if len(burst_rallies) > 0:
            print("\nCharacteristics of burst rallies:")
            print(f"Average duration: {burst_rallies['duration'].mean():.2f} steps")
            print(f"Average price increase: {burst_rallies['price_increase_pct'].mean():.2f}%")
            print(f"Average post-peak drop: {burst_rallies['post_peak_drop_pct'].mean():.2f}%")


if __name__ == "__main__":
    main()