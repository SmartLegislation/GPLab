from typing import Dict, Any, List
import random
import json

from src.subsystems.base import SocialSystemBase
from src.simulation.time import SimulationTime
from src.utils.logger import get_logger

class EcoSystemExp2(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Any = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(self.name)
        
        # --- Attributes specific to EcoSystemExp2 ---
        # LLM drivers for business agent
        self.business_llm_index = config.get("business_llm_driver_index", 2)
        
        # Product pricing and market data
        self.product_prices = {
            "green_vehicle": 180000,  # Initial prices (can be adjusted by business) - Lowered
            "traditional_vehicle": 150000
        }
        
        # Business agent state
        self.business_state = {
            "profit_margin_green": 0.15,  # Profit margin for green products
            "profit_margin_traditional": 0.20,  # Profit margin for traditional products
            "rd_investment_green": 0.05,  # R&D investment in green technology
            "rd_investment_traditional": 0.03,  # R&D investment in traditional technology
            "production_cost": {
                "green_vehicle": 150000,  # Base cost before pricing - Lowered
                "traditional_vehicle": 120000
            },
            "historical_sales": {
                "green_vehicle": [],  # List of sales per epoch
                "traditional_vehicle": []
            },
            "decisions": []  # History of business decisions
        }
        
        # Government agent state
        self.government_state = {
            "carbon_tax_rate": 0.0,  # Initial carbon tax rate (percentage)
            "green_subsidy": 0,  # Subsidy for green products (absolute amount)
            "traditional_tax": 0.0,  # Additional tax on traditional products
            "green_tax": 0.0,  # Basic tax on green products
            "environmental_goals_met": False
        }
        
        # Consumer market data
        self.market_data = {
            "green_vehicle_demand": 0,
            "traditional_vehicle_demand": 0,
            "green_vehicle_purchases": 0,
            "traditional_vehicle_purchases": 0,
            "green_vehicle_market_share": 0.0,
            "traditional_vehicle_market_share": 0.0,
            "total_carbon_emissions": 0.0,
            "cumulative_carbon_emissions": 0.0,
            "potential_buyers_ratio": 0.3,  # Percentage of agents who can buy a vehicle per epoch
            "avg_green_intention": 0.0,
            "avg_traditional_intention": 0.0
        }
        
        # Economic metrics for evaluation
        self.economic_metrics = {
            "green_technology_advancement": 0.0,  # Advancement in green technology
            "market_transformation_rate": 0.0,  # Rate at which market is transforming to green
            "price_differential_ratio": 0.0,  # Ratio of green to traditional prices
            "consumer_economic_impact": 0.0  # Economic impact on consumers
        }
        
        # Store epoch-by-epoch metrics for evaluation
        self.epoch_metrics = {}
        
        # Carbon pricing policies (defined in config)
        self.carbon_pricing_policies = config.get("carbon_pricing_policies", {})
        self.current_carbon_policy_info: Dict[str, Any] = {}

        self.logger.info(f"EcoSystemExp2 initialized with config: {config}")

    def set_time(self, simulation_time: SimulationTime):
        """Override to update policies at appropriate epochs."""
        previous_epoch = self.current_time.get_current_epoch() if self.current_time else None
        super().set_time(simulation_time)
        
        # Update policy when epoch changes
        if previous_epoch != simulation_time.get_current_epoch():
            self._update_policy_for_current_epoch()

    def _update_policy_for_current_epoch(self):
        """Updates the current policy based on the simulation epoch."""
        current_epoch_str = ""
        if self.current_time:
            current_epoch_str = str(self.current_time.get_current_epoch())
        else:
            # If current_time is None (e.g., during initial call from init before set_time(epoch 0)),
            # try to load policy for epoch "0" by default.
            current_epoch_str = "0"
            self.logger.info(f"current_time not set, attempting to load policy for default epoch '{current_epoch_str}'.")

        if current_epoch_str in self.carbon_pricing_policies:
            self.current_carbon_policy_info = self.carbon_pricing_policies[current_epoch_str]
            
            # Apply the policy to government state
            if "vehicle_tax_rate" in self.current_carbon_policy_info:
                self.government_state["traditional_tax"] = self.current_carbon_policy_info["vehicle_tax_rate"]
            else: # Ensure a default if not in policy
                self.government_state["traditional_tax"] = self.government_state.get("traditional_tax", 0.0)

            if "green_vehicle_subsidy" in self.current_carbon_policy_info:
                self.government_state["green_subsidy"] = self.current_carbon_policy_info["green_vehicle_subsidy"]
            else: # Ensure a default if not in policy
                self.government_state["green_subsidy"] = self.government_state.get("green_subsidy", 0)
            
                self.logger.info(f"Epoch {current_epoch_str}: Loaded carbon policy: {self.current_carbon_policy_info}")
        else:
            self.logger.warning(f"Epoch {current_epoch_str}: No specific carbon policy found. Initial policy settings may be default or empty.")
            # Ensure government_state has default values if no policy is found for epoch "0"
            if current_epoch_str == "0":
                self.government_state["traditional_tax"] = self.government_state.get("traditional_tax", 0.0)
                self.government_state["green_subsidy"] = self.government_state.get("green_subsidy", 0)
            self.current_carbon_policy_info = self.current_carbon_policy_info or {"description": "No specific carbon policy active for this epoch."}
        # Ensure government_state always has these keys initialized after policy update attempt
        self.government_state.setdefault("traditional_tax", 0.0)
        self.government_state.setdefault("green_subsidy", 0)

    def _process_consumer_decisions(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """
        Process consumer agent decisions about vehicle purchases based on purchase intentions.
        Purchase intentions are numerical values from 0.0 to 1.0 indicating likelihood of purchase.
        Higher values represent stronger purchase intentions.
        """
        # Reset purchase counts for this epoch
        self.market_data["green_vehicle_purchases"] = 0
        self.market_data["traditional_vehicle_purchases"] = 0
        
        # Track aggregate intention metrics
        total_green_intention = 0
        total_traditional_intention = 0
        num_agents_with_decisions = 0

        # Purchase probability thresholds - agents with intention above these values will make a purchase
        # This creates a probabilistic model where higher intentions are more likely to convert to sales
        GREEN_PURCHASE_THRESHOLD = 0.7  # Higher threshold as green products are typically considered more carefully
        TRADITIONAL_PURCHASE_THRESHOLD = 0.6  # Slightly lower threshold for traditional products

        for agent_id, decisions in agent_decisions.items():
            if not decisions:
                continue
                
            # Ensure EcoSystemExp2 decisions exist and are valid
            if 'EcoSystemExp2' not in decisions or not decisions['EcoSystemExp2']:
                self.logger.debug(f"Agent {agent_id} has no EcoSystemExp2 decisions, skipping")
                continue
                
            num_agents_with_decisions += 1
            eco_decisions = decisions['EcoSystemExp2']
            
            # Process vehicle purchase intentions with proper null checks
            green_intention_raw = eco_decisions.get("green_product_purchase_intention", 0.0)
            traditional_intention_raw = eco_decisions.get("traditional_product_purchase_intention", 0.0)
            
            # Handle None values and convert to float safely
            try:
                green_intention = float(green_intention_raw) if green_intention_raw is not None else 0.0
            except (ValueError, TypeError):
                self.logger.warning(f"Agent {agent_id} has invalid green_product_purchase_intention: {green_intention_raw}, using 0.0")
                green_intention = 0.0
                
            try:
                traditional_intention = float(traditional_intention_raw) if traditional_intention_raw is not None else 0.0
            except (ValueError, TypeError):
                self.logger.warning(f"Agent {agent_id} has invalid traditional_product_purchase_intention: {traditional_intention_raw}, using 0.0")
                traditional_intention = 0.0
            
            # Accumulate total intentions for market analysis
            if isinstance(green_intention, (int, float)):
                total_green_intention += green_intention
            if isinstance(traditional_intention, (int, float)):
                total_traditional_intention += traditional_intention
            
            # Determine actual purchases based on intention levels
            # Higher intention values have higher probability of resulting in a purchase
            
            # Green vehicle purchase decision (probabilistic based on intention)
            if isinstance(green_intention, (int, float)) and green_intention > 0:
                # Direct threshold approach - intention above threshold results in purchase
                if green_intention >= GREEN_PURCHASE_THRESHOLD:
                    self.market_data["green_vehicle_purchases"] += 1
                    self.logger.debug(f"Agent {agent_id} purchased a green vehicle with intention level {green_intention:.2f}")
                # Alternatively, could use probabilistic approach:
                # if random.random() < green_intention:
                #     self.market_data["green_vehicle_purchases"] += 1
            
            # Traditional vehicle purchase decision (probabilistic based on intention)
            if isinstance(traditional_intention, (int, float)) and traditional_intention > 0:
                # Only purchase traditional if green wasn't purchased (assuming one car per agent)
                if traditional_intention >= TRADITIONAL_PURCHASE_THRESHOLD and green_intention < GREEN_PURCHASE_THRESHOLD:
                    self.market_data["traditional_vehicle_purchases"] += 1
                    self.logger.debug(f"Agent {agent_id} purchased a traditional vehicle with intention level {traditional_intention:.2f}")
        
        # Calculate average intentions if there are agents with decisions
        if num_agents_with_decisions > 0:
            avg_green_intention = total_green_intention / num_agents_with_decisions
            avg_traditional_intention = total_traditional_intention / num_agents_with_decisions
            self.logger.info(f"Average intentions - Green: {avg_green_intention:.2f}, Traditional: {avg_traditional_intention:.2f}")
            
            # Store average intentions in market data for analysis
            self.market_data["avg_green_intention"] = avg_green_intention
            self.market_data["avg_traditional_intention"] = avg_traditional_intention
        
        # Update market shares
        total_purchases = self.market_data["green_vehicle_purchases"] + self.market_data["traditional_vehicle_purchases"]
        if total_purchases > 0:
            self.market_data["green_vehicle_market_share"] = self.market_data["green_vehicle_purchases"] / total_purchases
            self.market_data["traditional_vehicle_market_share"] = self.market_data["traditional_vehicle_purchases"] / total_purchases
        
        # Update business historical sales
        self.business_state["historical_sales"]["green_vehicle"].append(self.market_data["green_vehicle_purchases"])
        self.business_state["historical_sales"]["traditional_vehicle"].append(self.market_data["traditional_vehicle_purchases"])
        
        # Calculate carbon emissions from purchases
        # Assume traditional vehicles emit more carbon
        green_emissions = self.market_data["green_vehicle_purchases"] * 0.1  # Lower emissions factor - Made cleaner
        traditional_emissions = self.market_data["traditional_vehicle_purchases"] * 1.2  # Higher emissions factor - Made dirtier
        self.market_data["total_carbon_emissions"] = green_emissions + traditional_emissions
        self.market_data["cumulative_carbon_emissions"] += self.market_data["total_carbon_emissions"]
        
        self.logger.info(f"Processed consumer decisions: {self.market_data['green_vehicle_purchases']} green vehicles and {self.market_data['traditional_vehicle_purchases']} traditional vehicles purchased")
    
    def _make_business_decisions(self):
        """
        Simulate business agent decisions using LLM.
        In a full implementation, this would make an LLM API call.
        For now, we'll use rule-based decision-making.
        """
        # Analyze recent sales trends (last 3 epochs or all if fewer)
        green_trend = []
        traditional_trend = []
        
        if self.business_state["historical_sales"]["green_vehicle"]:
            green_trend = self.business_state["historical_sales"]["green_vehicle"][-3:]
            traditional_trend = self.business_state["historical_sales"]["traditional_vehicle"][-3:]
        
        # Calculate average sales
        avg_green_sales = sum(green_trend) / len(green_trend) if green_trend else 0
        avg_traditional_sales = sum(traditional_trend) / len(traditional_trend) if traditional_trend else 0
        
        # Business logic for price adjustment
        # If green sales are high, can increase prices or invest more in R&D
        # If traditional sales are declining, may need to lower prices
        
        # Adjust R&D investment based on market trends
        if avg_green_sales > avg_traditional_sales:
            # Increase investment in green technology
            self.business_state["rd_investment_green"] = min(0.20, self.business_state["rd_investment_green"] * 1.1)
            self.business_state["rd_investment_traditional"] = max(0.01, self.business_state["rd_investment_traditional"] * 0.9)
        else:
            # Balance investments
            self.business_state["rd_investment_green"] = max(0.05, self.business_state["rd_investment_green"] * 0.95)
            self.business_state["rd_investment_traditional"] = min(0.10, self.business_state["rd_investment_traditional"] * 1.05)
        
        # Adjust production costs based on R&D investment (long-term cost reduction)
        self.business_state["production_cost"]["green_vehicle"] *= (1 - self.business_state["rd_investment_green"] * 0.05)
        self.business_state["production_cost"]["traditional_vehicle"] *= (1 - self.business_state["rd_investment_traditional"] * 0.03)
        
        # Adjust prices based on costs, demand, and competitive factors
        # Include government taxes/subsidies in pricing considerations
        green_base_price = self.business_state["production_cost"]["green_vehicle"] * (1 + self.business_state["profit_margin_green"])
        traditional_base_price = self.business_state["production_cost"]["traditional_vehicle"] * (1 + self.business_state["profit_margin_traditional"])
        
        # Apply price adjustment based on market conditions for GREEN VEHICLES
        # Default to a slight decrease for green vehicles if no specific growth conditions met.
        # This replaces the original logic that increased price on growth.
        green_market_adjustment = 0.98 
        self.logger.debug(f"Initial green_market_adjustment set to: {green_market_adjustment}")

        # Parameters for "booming sales" leading to significant price drop for green vehicles
        # These values can be tuned or moved to a configuration file if more flexibility is desired.
        SALES_BOOM_GROWTH_THRESHOLD_GREEN = 1.5  # Example: Sales grew by 50% compared to the start of the trend window
        SIGNIFICANT_PRICE_DROP_FACTOR_GREEN = 0.15 # Example: Reduce price by 15% (adjustment factor becomes 1.0 - 0.15 = 0.85)
        MODERATE_GROWTH_PRICE_DROP_FACTOR_GREEN = 0.05 # Example: Reduce price by 5% for moderate growth (adjustment factor becomes 0.95)

        if len(green_trend) >= 2: # Need at least two data points to establish a trend for growth
            baseline_sales = float(green_trend[0]) # Ensure float for division
            current_sales = float(green_trend[-1])

            if baseline_sales > 0: # Avoid division by zero and ensure there were baseline sales
                growth_ratio = current_sales / baseline_sales
                if growth_ratio >= SALES_BOOM_GROWTH_THRESHOLD_GREEN:
                    green_market_adjustment = 1.0 - SIGNIFICANT_PRICE_DROP_FACTOR_GREEN
                    self.logger.info(f"Green vehicle sales booming (current: {current_sales}, baseline: {baseline_sales}, ratio: {growth_ratio:.2f}). Applying significant price drop. Adjustment factor: {green_market_adjustment:.2f}")
                elif growth_ratio > 1.0: # Any growth, but not booming
                    green_market_adjustment = 1.0 - MODERATE_GROWTH_PRICE_DROP_FACTOR_GREEN
                    self.logger.info(f"Green vehicle sales growing moderately (current: {current_sales}, baseline: {baseline_sales}, ratio: {growth_ratio:.2f}). Applying moderate price drop. Adjustment factor: {green_market_adjustment:.2f}")
                else: # Sales are stagnant (growth_ratio == 1.0) or declining (growth_ratio < 1.0)
                    # If not growing, set adjustment to 1.0 (no change from base price calculation due to sales trend)
                    # The initial default of 0.98 will be overridden if conditions above are not met.
                    green_market_adjustment = 1.0 
                    self.logger.info(f"Green vehicle sales stagnant or declining (current: {current_sales}, baseline: {baseline_sales}, ratio: {growth_ratio:.2f}). No growth-based price drop. Adjustment factor: {green_market_adjustment:.2f}")
            elif current_sales > 0: # Baseline sales were zero, but current sales are positive (e.g., new product launch)
                 green_market_adjustment = 1.0 - MODERATE_GROWTH_PRICE_DROP_FACTOR_GREEN # Consider initial sales as moderate growth
                 self.logger.info(f"Green vehicle has initial sales (current: {current_sales}, baseline: 0). Applying moderate price drop. Adjustment factor: {green_market_adjustment:.2f}")
            # If baseline is 0 and current sales are 0, the initial green_market_adjustment (0.98) remains.
        else:
            # Not enough trend data (e.g., first epoch of sales, green_trend has 0 or 1 item)
            # The initial green_market_adjustment (0.98) will apply.
            self.logger.info(f"Not enough green vehicle sales trend data (length: {len(green_trend)}) to apply boom-based pricing. Using default adjustment: {green_market_adjustment}")
        
        # Apply price adjustment for TRADITIONAL VEHICLES (original logic preserved)
        traditional_market_adjustment = 1.0
        if len(traditional_trend) > 1 and traditional_trend[-1] < traditional_trend[0]:
            traditional_market_adjustment = 0.97  # Price decrease to maintain market share
        else:
            traditional_market_adjustment = 1.01  # Slight price increase if stable/growing
        
        # Final price calculations
        self.product_prices["green_vehicle"] = green_base_price * green_market_adjustment
        self.product_prices["traditional_vehicle"] = traditional_base_price * traditional_market_adjustment
        
        # Update system state with new prices
        self.system_state["current_prices"] = self.product_prices
        
        # Log the business decisions
        business_decision = {
            "epoch": self.current_time.get_current_epoch() if self.current_time else 0,
            "rd_investment_green": self.business_state["rd_investment_green"],
            "rd_investment_traditional": self.business_state["rd_investment_traditional"],
            "green_vehicle_price": self.product_prices["green_vehicle"],
            "traditional_vehicle_price": self.product_prices["traditional_vehicle"],
            "profit_margin_green": self.business_state["profit_margin_green"],
            "profit_margin_traditional": self.business_state["profit_margin_traditional"]
        }
        
        self.business_state["decisions"].append(business_decision)
        self.logger.info(f"Business made pricing decisions: green vehicle: {self.product_prices['green_vehicle']:.2f}, traditional vehicle: {self.product_prices['traditional_vehicle']:.2f}")
    
    def _calculate_market_impacts(self):
        """Calculate overall market impacts after all decisions."""
        # Update market demand prediction for next epoch based on current prices and policies
        # This influences business decisions in the next step
        
        # Calculate effective prices after taxes and subsidies
        effective_green_price = self.product_prices["green_vehicle"] * (1 + self.government_state["green_tax"]) - self.government_state["green_subsidy"]
        effective_traditional_price = self.product_prices["traditional_vehicle"] * (1 + self.government_state["traditional_tax"])
        
        # Price ratio is a key factor in consumer purchasing decisions
        price_ratio = effective_green_price / effective_traditional_price if effective_traditional_price > 0 else 999
        
        # Store effective prices in system state for agent information
        self.system_state["effective_prices"] = {
            "green_vehicle": effective_green_price,
            "traditional_vehicle": effective_traditional_price,
            "price_ratio": price_ratio
        }
        
        # Predict demand for next epoch based on current trends
        # This is simplified - a real model would use more sophisticated prediction
        self.market_data["green_vehicle_demand"] = int(
            self.market_data["green_vehicle_purchases"] * 
            (1.1 if price_ratio < 1.2 else 0.9)  # Price sensitivity
        )
        
        self.market_data["traditional_vehicle_demand"] = int(
            self.market_data["traditional_vehicle_purchases"] * 
            (1.1 if self.government_state["traditional_tax"] < 0.15 else 0.9)  # Tax sensitivity
        )
        
        self.logger.debug(f"Market impacts calculated: Effective prices - green: {effective_green_price:.2f}, traditional: {effective_traditional_price:.2f}")
    
    def _update_economic_metrics(self):
        """Update economic metrics for evaluation."""
        # Technology advancement based on R&D investment
        self.economic_metrics["green_technology_advancement"] = min(1.0, self.economic_metrics.get("green_technology_advancement", 0) + self.business_state["rd_investment_green"] * 0.1)
        
        # Market transformation rate
        # How fast is the market shifting to green vehicles?
        if hasattr(self, "_previous_green_market_share"):
            transformation_rate = self.market_data["green_vehicle_market_share"] - self._previous_green_market_share
            self.economic_metrics["market_transformation_rate"] = transformation_rate
        else:
            self.economic_metrics["market_transformation_rate"] = 0
            
        self._previous_green_market_share = self.market_data["green_vehicle_market_share"]
        
        # Price differential ratio
        self.economic_metrics["price_differential_ratio"] = self.product_prices["green_vehicle"] / self.product_prices["traditional_vehicle"] if self.product_prices["traditional_vehicle"] > 0 else 1.5
        
        # Consumer economic impact (conceptual)
        # Higher value means less economic burden on consumers
        price_burden = 1.0 - min(1.0, (self.system_state["effective_prices"]["green_vehicle"] / (self.system_state["consumer_economic_distribution"]["median_income"] * 3)))
        self.economic_metrics["consumer_economic_impact"] = price_burden
        
        self.logger.debug(f"Updated economic metrics: Market transformation rate: {self.economic_metrics['market_transformation_rate']:.2f}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """
        Returns economic system information relevant to a specific agent.
        This includes product prices, government policies, and market trends.
        """
        # Calculate what the agent would actually pay after taxes and subsidies
        green_vehicle_final_price = self.system_state["effective_prices"]["green_vehicle"]
        traditional_vehicle_final_price = self.system_state["effective_prices"]["traditional_vehicle"]
        
        # # Prepare market information
        market_info = {
            "green_vehicle_market_share": self.market_data["green_vehicle_market_share"],
            "traditional_vehicle_market_share": self.market_data["traditional_vehicle_market_share"],
            "total_carbon_emissions_current": self.market_data["total_carbon_emissions"],
            "price_trend_green": "stable",
            "price_trend_traditional": "stable"
        }
        
        # Determine price trends if we have historical data
        if self.current_time and self.current_time.get_current_epoch() > 0:
            previous_epoch = self.current_time.get_current_epoch() - 1
            if previous_epoch in self.epoch_metrics:
                prev_green_price = self.epoch_metrics[previous_epoch].get("green_price", 0)
                prev_traditional_price = self.epoch_metrics[previous_epoch].get("traditional_price", 0)
                
                # Calculate price trends
                if self.product_prices["green_vehicle"] > prev_green_price * 1.05:
                    market_info["price_trend_green"] = "increasing"
                elif self.product_prices["green_vehicle"] < prev_green_price * 0.95:
                    market_info["price_trend_green"] = "decreasing"
                    
                if self.product_prices["traditional_vehicle"] > prev_traditional_price * 1.05:
                    market_info["price_trend_traditional"] = "increasing"
                elif self.product_prices["traditional_vehicle"] < prev_traditional_price * 0.95:
                    market_info["price_trend_traditional"] = "decreasing"
        
        # Technology advancement affects perceived value
        green_technology_value = "standard"
        if self.economic_metrics["green_technology_advancement"] > 0.7:
            green_technology_value = "cutting-edge"
        elif self.economic_metrics["green_technology_advancement"] > 0.4:
            green_technology_value = "advanced"
            
        # Information package for the agent
        agent_info = {
            "vehicle_prices": {
                "green_vehicle_Market price (after subsidy)": round(green_vehicle_final_price),  # After tax/subsidy
                "traditional_vehicle_Market price (after subsidy)": round(traditional_vehicle_final_price),  # After tax
            },
            "government_policy": {
                "carbon_tax_rate": self.government_state["traditional_tax"],
                "green_subsidy_amount": self.government_state["green_subsidy"],
            },
            # "market_information": market_info,
            # "product_information": {
            #     "green_vehicle_technology": green_technology_value,
            #     "traditional_vehicle_reliability": "proven"  # Traditional technology is usually more mature
            # }
        }
        
        return agent_info

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the state of the economic system at the end of the simulation.
        Returns comprehensive economic metrics and trends.
        """
        self.logger.info("Evaluating EcoSystemExp2...")
        
        # Calculate final market transformation
        initial_green_market_share = 0
        if -1 in self.epoch_metrics: # Use epoch -1 for true initial state
            initial_green_market_share = self.epoch_metrics[-1].get("green_market_share", 0)
        elif 0 in self.epoch_metrics: # Fallback to epoch 0 if -1 is not available for some reason
            initial_green_market_share = self.epoch_metrics[0].get("green_market_share", 0)
        
        current_green_market_share = self.market_data["green_vehicle_market_share"]
        market_transformation = current_green_market_share - initial_green_market_share
        
        # Calculate price evolution
        price_evolution = {}
        for epoch in sorted(self.epoch_metrics.keys()):
            price_evolution[epoch] = {
                "green_price": self.epoch_metrics[epoch].get("green_price", 0),
                "traditional_price": self.epoch_metrics[epoch].get("traditional_price", 0),
                "price_ratio": self.epoch_metrics[epoch].get("green_price", 0) / self.epoch_metrics[epoch].get("traditional_price", 1) if self.epoch_metrics[epoch].get("traditional_price", 0) > 0 else 0
            }
        
        # Calculate sales evolution
        sales_evolution = {}
        for epoch in sorted(self.epoch_metrics.keys()):
            sales_evolution[epoch] = {
                "green_sales": self.epoch_metrics[epoch].get("green_vehicle_sales", 0),
                "traditional_sales": self.epoch_metrics[epoch].get("traditional_vehicle_sales", 0),
                "total_sales": self.epoch_metrics[epoch].get("green_vehicle_sales", 0) + self.epoch_metrics[epoch].get("traditional_vehicle_sales", 0)
            }
        
        # Calculate purchase intentions and carbon emissions evolution
        purchase_intentions_and_emissions = {}
        for epoch in sorted(self.epoch_metrics.keys()):
            # Only include epochs >= 0 (exclude initial state at epoch -1)
            if epoch >= 0:
                purchase_intentions_and_emissions[epoch] = {
                    "avg_green_intention": self.epoch_metrics[epoch].get("avg_green_intention", 0.0),
                    "avg_traditional_intention": self.epoch_metrics[epoch].get("avg_traditional_intention", 0.0),
                    "carbon_emissions": self.epoch_metrics[epoch].get("carbon_emissions", 0.0),
                    "cumulative_emissions": self.epoch_metrics[epoch].get("cumulative_emissions", 0.0)
                }
        
        # Calculate policy impact
        policy_impact = {}
        for epoch in sorted(self.epoch_metrics.keys()):
            if epoch > 0 and epoch-1 in self.epoch_metrics:
                # Compare sales before and after policy changes
                prev_green_sales = self.epoch_metrics[epoch-1].get("green_vehicle_sales", 0)
                current_green_sales = self.epoch_metrics[epoch].get("green_vehicle_sales", 0)
                sales_change = (current_green_sales - prev_green_sales) / max(1, prev_green_sales)
                
                # Look for policy changes
                prev_subsidy = self.epoch_metrics[epoch-1].get("green_subsidy", 0)
                current_subsidy = self.epoch_metrics[epoch].get("green_subsidy", 0)
                subsidy_change = current_subsidy - prev_subsidy
                
                prev_tax = self.epoch_metrics[epoch-1].get("traditional_tax", 0)
                current_tax = self.epoch_metrics[epoch].get("traditional_tax", 0)
                tax_change = current_tax - prev_tax
                
                policy_impact[epoch] = {
                    "subsidy_change": subsidy_change,
                    "tax_change": tax_change,
                    "sales_impact": sales_change,
                    "effectiveness": sales_change / (abs(subsidy_change) + abs(tax_change) + 0.001)  # Effectiveness ratio
                }
        
        # Final evaluation results
        evaluation = {
            "market_transformation": {
                "initial_green_market_share": initial_green_market_share,
                "final_green_market_share": current_green_market_share,
                "transformation_magnitude": market_transformation,
                "transformation_rate": market_transformation / max(1, len(self.epoch_metrics))  # Per epoch
            },
            "environmental_impact": {
                "final_carbon_emissions": self.market_data["total_carbon_emissions"],
                "cumulative_carbon_emissions": self.market_data["cumulative_carbon_emissions"],
                "emission_reduction_rate": 1.0 - (self.market_data["total_carbon_emissions"] / (self.system_state["total_agents"] * 0.5))  # Compared to baseline
            },
            "price_dynamics": price_evolution,
            "sales_evolution": sales_evolution,
            "purchase_intentions_and_emissions": purchase_intentions_and_emissions,
            "policy_impact_analysis": policy_impact,
            "business_decisions": self.business_state["decisions"]
        }
        
        self.evaluation_results = evaluation
        self.logger.info(f"EcoSystemExp2 evaluation complete: Market transformation: {market_transformation:.2f}, Green technology advancement: {self.economic_metrics['green_technology_advancement']:.2f}")
        return evaluation

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return system state suitable for saving to DB."""
        state = super().get_state_for_persistence()
        state.update({
            "product_prices": self.product_prices,
            "business_state": self.business_state,
            "government_state": self.government_state,
            "market_data": self.market_data,
            "economic_metrics": self.economic_metrics,
            "epoch_metrics": self.epoch_metrics
        })
        return state 

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initializes the economic system."""
        self.logger.info("Initializing EcoSystemExp2...")
        
        # Initialize system_state
        self.system_state["total_agents"] = len(all_agent_data)
        self.system_state["market_conditions"] = "stable"
        self.system_state["consumer_economic_distribution"] = self._analyze_consumer_economics(all_agent_data)
        
        # Calculate potential market size based on economic distribution
        potential_buyers = int(len(all_agent_data) * self.market_data["potential_buyers_ratio"])
        self.system_state["potential_buyers_per_epoch"] = potential_buyers
        
        # Initialize prices
        self.system_state["current_prices"] = self.product_prices
        
        # --- BEGIN MODIFICATION: Store true initial state as epoch -1 ---
        # Capture state before any epoch 0 policies are applied
        self.epoch_metrics[-1] = {
            "green_price": self.product_prices["green_vehicle"],
            "traditional_price": self.product_prices["traditional_vehicle"],
            "green_subsidy": self.government_state.get("green_subsidy", 0), # Initial, pre-policy
            "traditional_tax": self.government_state.get("traditional_tax", 0.0), # Initial, pre-policy
            "green_market_share": 0.0,
            "traditional_market_share": 0.0,
            "green_vehicle_sales": 0,
            "traditional_vehicle_sales": 0,
            "carbon_emissions": 0.0,
            "cumulative_emissions": 0.0,
            "business_rd_green": self.business_state["rd_investment_green"], # Initial RD
            "avg_green_intention": 0.0,
            "avg_traditional_intention": 0.0
        }
        self.logger.info(f"Stored initial pre-simulation state in epoch_metrics[-1]: {self.epoch_metrics[-1]}")
        # --- END MODIFICATION ---

        # Initialize policy for the starting time (epoch 0)
        self._update_policy_for_current_epoch()
        
        # Initialize effective_prices in system_state
        # Ensure government_state values are numbers before calculation
        green_tax = self.government_state.get("green_tax", 0.0)
        green_subsidy = self.government_state.get("green_subsidy", 0)
        traditional_tax = self.government_state.get("traditional_tax", 0.0)

        initial_effective_green_price = self.product_prices["green_vehicle"] * (1 + green_tax) - green_subsidy
        initial_effective_traditional_price = self.product_prices["traditional_vehicle"] * (1 + traditional_tax)
        
        if initial_effective_traditional_price > 0:
            initial_price_ratio = initial_effective_green_price / initial_effective_traditional_price
        else:
            initial_price_ratio = 999 # A large number to indicate traditional price is zero or invalid

        self.system_state["effective_prices"] = {
            "green_vehicle": initial_effective_green_price,
            "traditional_vehicle": initial_effective_traditional_price,
            "price_ratio": initial_price_ratio
        }
        
        # Store initial state in epoch metrics
        initial_epoch_key = 0
        if self.current_time: # Check if current_time is already set
            initial_epoch_key = self.current_time.get_current_epoch()

        self.epoch_metrics[initial_epoch_key] = {
            "green_price": self.product_prices["green_vehicle"],
            "traditional_price": self.product_prices["traditional_vehicle"],
            "green_subsidy": self.government_state["green_subsidy"],
            "traditional_tax": self.government_state["traditional_tax"], # Changed from carbon_tax to traditional_tax
            "green_market_share": 0.0,
            "traditional_market_share": 0.0
        }
        
        self.logger.info("EcoSystemExp2 initialized.")

    def _analyze_consumer_economics(self, all_agent_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze economic distribution of consumers for market sizing."""
        incomes = []
        for agent in all_agent_data:
            # Path to income might vary depending on your data structure
            income_path = "Economic Attributes.Annual Household Income"
            income_parts = income_path.split(".")
            
            # Traverse the nested structure to get income
            income_value = agent
            for part in income_parts:
                if isinstance(income_value, dict) and part in income_value:
                    income_value = income_value[part]
                else:
                    income_value = None
                    break
            
            # Only append if income_value is not None and can be converted to float
            if income_value is not None:
                if isinstance(income_value, str):
                    # Remove non-numeric characters and convert to float
                    income_value = ''.join(c for c in income_value if c.isdigit() or c == '.')
                    try:
                        income_value = float(income_value) if income_value else None
                    except ValueError:
                        income_value = None
                elif isinstance(income_value, (int, float)):
                    income_value = float(income_value)
                else:
                    income_value = None
                    
                if income_value is not None:
                    incomes.append(income_value)
        
        # Calculate income distribution statistics
        if incomes:
            avg_income = sum(incomes) / len(incomes)
            incomes.sort()
            median_income = incomes[len(incomes) // 2]
            
            # Categorize by income level
            low_income = len([i for i in incomes if i < median_income * 0.5])
            mid_income = len([i for i in incomes if median_income * 0.5 <= i < median_income * 1.5])
            high_income = len([i for i in incomes if i >= median_income * 1.5])
            
            return {
                "average_income": avg_income,
                "median_income": median_income,
                "income_distribution": {
                    "low_income": low_income / len(incomes),
                    "middle_income": mid_income / len(incomes),
                    "high_income": high_income / len(incomes)
                }
            }
        else:
            return {
                "average_income": 0,
                "median_income": 0,
                "income_distribution": {
                    "low_income": 0,
                    "middle_income": 0,
                    "high_income": 0
                }
            }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """
        Processes decisions from all agents for this subsystem in the current epoch.
        Also handles LLM-driven business decisions.
        """
        self.logger.debug(f"EcoSystemExp2 stepping with agent decisions: {agent_decisions}")

        # 1. Process consumer agent decisions
        self._process_consumer_decisions(agent_decisions)
        
        # 2. Make business agent decisions (LLM-driven)
        self._make_business_decisions()
        
        # 3. Calculate market impacts and update prices
        self._calculate_market_impacts()
        
        # 4. Update economic metrics
        self._update_economic_metrics()
        
        # 5. Store epoch metrics
        if self.current_time:
            current_epoch = self.current_time.get_current_epoch()
            self.epoch_metrics[current_epoch] = {
                "green_price": self.product_prices["green_vehicle"],
                "traditional_price": self.product_prices["traditional_vehicle"],
                "green_subsidy": self.government_state["green_subsidy"],
                "traditional_tax": self.government_state["traditional_tax"],
                "green_market_share": self.market_data["green_vehicle_market_share"],
                "traditional_market_share": self.market_data["traditional_vehicle_market_share"],
                "green_vehicle_sales": self.market_data["green_vehicle_purchases"],
                "traditional_vehicle_sales": self.market_data["traditional_vehicle_purchases"],
                "carbon_emissions": self.market_data["total_carbon_emissions"],
                "cumulative_emissions": self.market_data["cumulative_carbon_emissions"],
                "business_rd_green": self.business_state["rd_investment_green"],
                "avg_green_intention": self.market_data.get("avg_green_intention", 0.0),
                "avg_traditional_intention": self.market_data.get("avg_traditional_intention", 0.0)
            }
        
        self.logger.debug("EcoSystemExp2 step completed.") 