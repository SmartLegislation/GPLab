import random
import numpy as np
from typing import Dict, Any, List
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class HousingMarketSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Market parameters
        self.initial_price = config.get("initial_avg_price_per_sqm", 15000)
        self.price_volatility = config.get("price_volatility", 0.02)
        self.current_price = self.initial_price
        self.price_history = [self.initial_price]
        
        # Policy parameters
        self.purchase_restrictions = config.get("purchase_restrictions", {})
        
        # Agent housing ownership tracking
        self.agent_properties = {}  # agent_id -> number of properties owned
        self.agent_purchase_history = {}  # agent_id -> list of purchase epochs
        
        # Market statistics
        self.monthly_transactions = 0
        self.monthly_demand = 0
        self.transaction_history = []
        self.demand_history = []
        
        # Track baseline transaction volume for comparison
        self.baseline_transactions = 0
        self.baseline_established = False
        
        # Opinion system integration
        self.opinion_system = None
        
        self.logger.info(f"HousingMarketSystem initialized with initial price: {self.initial_price} yuan/sqm")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent property ownership based on their economic status"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Initialize property ownership based on income and assets
            income_level = agent_data.get("economic_attributes", {}).get("income_level", "")
            assets = agent_data.get("economic_attributes", {}).get("assets", [])
            
            # Simple heuristic for initial property ownership
            if "home" in str(assets).lower() or "house" in str(assets).lower():
                self.agent_properties[agent_id] = 1
            elif "high" in income_level.lower() or "wealthy" in income_level.lower():
                self.agent_properties[agent_id] = random.choice([1, 2])
            else:
                self.agent_properties[agent_id] = random.choice([0, 1])
            
            self.agent_purchase_history[agent_id] = []
        
        self.logger.info(f"Initialized {len(self.agent_properties)} agents with property ownership")
    
    def connect_to_opinion_system(self, opinion_system):
        """Connect to the opinion system for sentiment influence"""
        self.opinion_system = opinion_system
        self.logger.info("Connected to HousingOpinionSystem for sentiment influence")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide housing market information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_restrictions = self.purchase_restrictions.get(str(current_epoch), 
                                                             self.purchase_restrictions.get("0", {}))
        
        # Calculate price trend
        price_trend = "stable"
        if len(self.price_history) > 1:
            recent_change = (self.current_price - self.price_history[-2]) / self.price_history[-2]
            if recent_change > 0.01:
                price_trend = "rising"
            elif recent_change < -0.01:
                price_trend = "falling"
        
        # Agent's current property status
        properties_owned = self.agent_properties.get(agent_id, 0)
        can_purchase = properties_owned < current_restrictions.get("max_properties_per_family", 3)
        
        # Determine applicable down payment ratio
        if properties_owned == 0:
            down_payment_ratio = current_restrictions.get("down_payment_ratio_first", 0.3)
        else:
            down_payment_ratio = current_restrictions.get("down_payment_ratio_second", 0.5)
        
        return {
            "housing_prices": {
                "current_price_per_sqm": self.current_price,
                "price_trend": price_trend,
                "monthly_change_percent": ((self.current_price - self.price_history[-2]) / self.price_history[-2] * 100) if len(self.price_history) > 1 else 0
            },
            "purchase_restrictions": {
                "max_properties_allowed": current_restrictions.get("max_properties_per_family", 3),
                "your_current_properties": properties_owned,
                "can_purchase_more": can_purchase,
                "required_down_payment_ratio": down_payment_ratio
            },
            "mortgage_rates": {
                "first_home_rate": 0.049,  # 4.9% annual
                "second_home_rate": 0.059  # 5.9% annual
            },
            "market_trends": {
                "recent_transaction_volume": self.monthly_transactions,
                "market_heat": "hot" if self.monthly_demand > self.monthly_transactions * 1.5 else "normal"
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent housing decisions and update market state"""
        self.monthly_transactions = 0
        self.monthly_demand = 0
        
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_restrictions = self.purchase_restrictions.get(str(current_epoch), 
                                                             self.purchase_restrictions.get("0", {}))
        
        # Process each agent's decision
        for agent_id, decisions in agent_decisions.items():
            if "HousingMarketSystem" not in decisions:
                continue
                
            decision = decisions["HousingMarketSystem"]
            purchase_intention_value = decision.get("housing_purchase_intention")
            purchase_intention = float(purchase_intention_value) if purchase_intention_value is not None else 0.0
            housing_type = decision.get("housing_type_preference", "no_preference")
            
            # Count demand
            if purchase_intention > 0.4:
                self.monthly_demand += 1
            
            # Process actual purchases (simplified model)
            if purchase_intention > 0.6:
                properties_owned = self.agent_properties.get(agent_id, 0)
                max_allowed = current_restrictions.get("max_properties_per_family", 3)
                
                if properties_owned < max_allowed:
                    # Simulate purchase success based on intention and market conditions
                    success_probability = purchase_intention * 0.8  # Adjust based on market heat
                    
                    if random.random() < success_probability:
                        self.agent_properties[agent_id] = properties_owned + 1
                        self.agent_purchase_history[agent_id].append(current_epoch)
                        self.monthly_transactions += 1
                        
                        self.logger.debug(f"Agent {agent_id} purchased property (type: {housing_type}), "
                                        f"now owns {self.agent_properties[agent_id]} properties")
        
        # Set baseline transactions in first epoch
        if not self.baseline_established and current_epoch == 0:
            self.baseline_transactions = max(self.monthly_transactions, 1)
            self.baseline_established = True
        
        # Update market price based on supply-demand dynamics with transaction volume consideration
        # When transactions drop significantly, prices should stabilize or decrease
        transaction_ratio = self.monthly_transactions / max(self.baseline_transactions, 1)
        
        # Calculate effective demand considering policy restrictions
        eligible_demand = 0
        for agent_id in agent_decisions:
            properties_owned = self.agent_properties.get(agent_id, 0)
            if properties_owned < current_restrictions.get("max_properties_per_family", 3):
                eligible_demand += 1
        
        # Adjust demand based on eligibility
        effective_demand = min(self.monthly_demand, eligible_demand)
        
        # Calculate price pressure with more weight on transaction volume
        demand_supply_ratio = effective_demand / max(self.monthly_transactions, 1)
        
        # Price pressure calculation with transaction volume consideration
        # When transactions are low, even high demand has limited price impact
        price_pressure = (demand_supply_ratio - 1) * 0.01 * transaction_ratio
        
        # Apply sentiment influence from opinion system if available
        sentiment_factor = 0.0
        if self.opinion_system:
            try:
                sentiment_factor = self.opinion_system.get_sentiment_influence()
                self.logger.debug(f"Applied sentiment factor: {sentiment_factor} from opinion system")
            except Exception as e:
                self.logger.warning(f"Failed to get sentiment influence: {e}")
        
        # Adjust price pressure based on sentiment
        price_pressure += sentiment_factor * 0.005
        
        # Add random market volatility
        random_factor = random.gauss(0, self.price_volatility * transaction_ratio)
        
        # Update price with bounds
        price_change = price_pressure + random_factor
        self.current_price = self.current_price * (1 + price_change)
        
        # More reasonable price bounds - prevent extreme price increases
        self.current_price = max(self.current_price, self.initial_price * 0.7)  # Floor at 70% of initial
        
        # Cap price increases more strictly when transactions are low
        max_price = self.initial_price * (1.0 + 0.1 * transaction_ratio)
        self.current_price = min(self.current_price, max_price)
        
        self.price_history.append(self.current_price)
        self.transaction_history.append(self.monthly_transactions)
        self.demand_history.append(self.monthly_demand)
        
        self.logger.info(f"Epoch {current_epoch}: Price={self.current_price:.0f}, "
                        f"Transactions={self.monthly_transactions}, Demand={self.monthly_demand}, "
                        f"D/S Ratio={demand_supply_ratio:.2f}, Transaction Ratio={transaction_ratio:.2f}, "
                        f"Sentiment Factor={sentiment_factor:.2f}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the effectiveness of housing purchase restriction policy"""
        # Calculate key metrics
        price_change_percent = ((self.current_price - self.initial_price) / self.initial_price) * 100
        avg_transactions = np.mean(self.transaction_history) if self.transaction_history else 0
        
        # Analyze property distribution
        property_distribution = {}
        for agent_id, num_properties in self.agent_properties.items():
            property_distribution[num_properties] = property_distribution.get(num_properties, 0) + 1
        
        # Calculate market cooling effectiveness
        early_transactions = self.transaction_history[:2] if len(self.transaction_history) > 2 else []
        late_transactions = self.transaction_history[-2:] if len(self.transaction_history) > 2 else []
        
        transaction_reduction = 0
        if early_transactions and late_transactions:
            transaction_reduction = (1 - np.mean(late_transactions) / np.mean(early_transactions)) * 100
        
        # Calculate price stabilization (lower volatility in recent periods)
        early_price_volatility = np.std(self.price_history[:3]) if len(self.price_history) > 3 else 0
        late_price_volatility = np.std(self.price_history[-3:]) if len(self.price_history) > 3 else 0
        price_stabilization = late_price_volatility <= early_price_volatility
        
        evaluation_results = {
            "price_metrics": {
                "initial_price": self.initial_price,
                "final_price": self.current_price,
                "total_price_change_percent": price_change_percent,
                "price_volatility": np.std(self.price_history) if len(self.price_history) > 1 else 0
            },
            "transaction_metrics": {
                "total_transactions": sum(self.transaction_history),
                "average_monthly_transactions": avg_transactions,
                "transaction_reduction_percent": transaction_reduction
            },
            "property_distribution": property_distribution,
            "policy_effectiveness": {
                "market_cooling_achieved": bool(price_change_percent < 10 and transaction_reduction > 20),
                "price_stabilization": price_stabilization
            },
            "time_series": {
                "price_history": self.price_history,
                "transaction_history": self.transaction_history,
                "demand_history": self.demand_history
            }
        }
        
        self.logger.info(f"evaluation_results：{evaluation_results}")
        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "current_price": self.current_price,
            "monthly_transactions": self.monthly_transactions,
            "monthly_demand": self.monthly_demand,
            "total_property_owners": sum(1 for v in self.agent_properties.values() if v > 0),
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 