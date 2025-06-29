## structure.py
from mesa import Model
from mesa.datacollection import DataCollector
import random
from agents import TraderAgent

class CustomScheduler:
    def __init__(self, model):
        self.model = model
        self.agents = []
    
    def add(self, agent):
        self.agents.append(agent)
    
    def step(self):
        for agent in self.agents:
            agent.step()

class StockMarketModel(Model):
    def __init__(self, num_agents, initial_price, momentum_threshold, volume_threshold, max_steps=20):
        super().__init__()
        self.num_agents = num_agents
        self.initial_price = initial_price
        self.current_price = initial_price
        self.momentum_threshold = momentum_threshold
        self.volume_threshold = volume_threshold
        self.trading_volume = 0
        self.market_sentiment = random.uniform(-1, 1)
        self.schedule = CustomScheduler(self)
        self.running = True
        self.max_steps = max_steps
        self.step_count = 0
        self.buy_volume = 0
        self.sell_volume = 0

        for i in range(self.num_agents):
            agent = TraderAgent(i, self)
            self.schedule.add(agent)
        
        self.datacollector = DataCollector(
            model_reporters={"Stock Price": "current_price"},
            agent_reporters={"Wealth": "wealth", "Strategy": lambda a: a.strategy}
        )

    def step(self):
        self.market_sentiment = random.uniform(-1, 1)
        self.schedule.step()
        self.datacollector.collect(self)
        self.step_count += 1

        if self.step_count >= self.max_steps:
            self.running = False

    def update_stock_price(self, price_change):
        self.current_price = max(1, self.current_price + price_change)
