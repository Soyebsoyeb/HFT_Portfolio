##  HFT Portfolio Management System
 
## System Overview
A comprehensive High-Frequency Trading Portfolio Management System implementing institutional-grade risk management, real-time analytics, and multi-asset trading capabilities. This system simulates a professional trading desk environment with advanced quantitative models and high-performance architecture.

Real-time portfolio management with institutional risk controls and high-frequency trading capabilities across multiple asset classes including equities, options, and derivatives.


## High-Level Component Diagram
<img width="838" height="363" alt="Screenshot 2025-10-20 011659" src="https://github.com/user-attachments/assets/dd29a0cc-159c-4e39-a97b-8933a9683f66" />

Market Data Feed triggers strategy or order logic.
Order Manager decides trades and routes them.
Execution Engine sends them to the market.
Portfolio Engine updates positions & PnL.
Risk Monitor ensures limits are respected.
Analytics Dashboard visualizes the entire system.

-------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------
## Core Modules Deep Dive

## 1. Market Data Infrastructure (MarketDataFeed)
Purpose:he MarketDataFeed class is essentially a high-frequency market simulation engine. Its main role is to generate realistic, live-like market data for assets, which can then be consumed by trading algorithms, risk engines, or backtesting frameworks.

```cpp
class MarketDataFeed {
    // Multi-threaded price generation engine
    void generate_market_data() {
        while (running) {
            // Brownian motion with stochastic volatility
            double ret = returns_dist(rng) * sqrt(volatility_factor);
            double new_price = price * exp(ret);
            
            // Dynamic spread modeling based on volume and volatility
            double spread = calculate_dynamic_spread(new_price, volume);
            update_market_data(symbol, new_price, spread);
            
            this_thread::sleep_for(microseconds(100)); // 10,000 updates/sec
        }
    }
    
    // Advanced features:
    // - Volume-weighted average price (VWAP) tracking
    // - Order book depth simulation
    // - Corporate action handling
    // - Latency modeling for exchange connectivity
};
```

## (i) Set up a continuous loop for real-time updates

Used while (running) to keep generating market data as long as the feed is active.
Ensures the simulation behaves like a live market feed.

## (ii) Model price changes using stochastic processes

Generated random returns: returns_dist(rng) → simulates unpredictable market movements.
Scaled by sqrt(volatility_factor) to include volatility in the price evolution.
Updated the price using geometric Brownian motion:

```cpp
new_price = price * exp(ret);
```
This is a standard method to simulate realistic stock or asset price movements in quantitative finance.

## (iii) Simulate liquidity with dynamic spreads

Calculated the bid-ask spread dynamically based on current price and volume:

```cpp
spread = calculate_dynamic_spread(new_price, volume);
```
Mimics real-world market microstructure where spreads widen during high volatility or low volume.

## (iv) Update market data for downstream systems

Fed the newly generated price and spread into the system using:
```cpp
update_market_data(symbol, new_price, spread);
```

This allows trading algorithms, risk engines, or portfolios to consume the data.

## (v) Throttle update frequency to simulate HFT environment

Introduced a small sleep:
```cpp
this_thread::sleep_for(microseconds(100));
```
Controls the rate of updates (here, 10,000 updates per second) to mimic a real-time exchange feed.

## (vi) Include hooks for advanced features (optional extensions)

VWAP tracking – calculates the volume-weighted average price.
Order book depth simulation – models liquidity at multiple price levels.
Corporate action handling – adjusts prices for dividends, splits, etc.
Latency modeling – simulates network or exchange delays for realism.

------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------


## 2. Order Management System (OrderManager)
Purpose: The OrderManager class is designed to manage the lifecycle of trading orders in a high-performance trading system. It ensures that orders are submitted, tracked, and executed efficiently, often in a low-latency environment like HFT (High-Frequency Trading).

It also supports advanced order types like Iceberg, TWAP, and VWAP.

```cpp
class OrderManager {
    // Lock-free order queue for maximum throughput
    CircularBuffer<Order, Config::ORDER_QUEUE_SIZE> order_queue;
    
    // Order lifecycle states:
    // PENDING → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED/CANCELLED/REJECTED
    
    // Advanced order types:
    bool submit_iceberg_order(const string& symbol, OrderSide side, 
                             int total_quantity, int display_quantity,
                             double price, string strategy_id);
    
    bool submit_twap_order(const string& symbol, OrderSide side,
                          int total_quantity, milliseconds duration);
    
    bool submit_vwap_order(const string& symbol, OrderSide side,
                          int total_quantity, const VWAPProfile& profile);
};
```
## Key Components

## a) Lock-free order queue
```cpp
CircularBuffer<Order, Config::ORDER_QUEUE_SIZE> order_queue;
```
Stores orders in a fixed-size, high-throughput circular buffer.

Lock-free design ensures minimal contention and maximum speed, crucial in real-time trading.

Orders flow through the queue from submission → execution → completion.

## b) Order lifecycle states
// PENDING → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED/CANCELLED/REJECTED


Every order passes through well-defined states:

(i)   PENDING – Order submitted but not yet acknowledged by the exchange.
(ii)  ACKNOWLEDGED – Exchange has confirmed the order.
(iii) PARTIALLY_FILLED – Only some quantity has been executed.
(iv) FILLED / CANCELLED / REJECTED – Final state of the order.

This helps track the status of every order in real-time.

## Advanced Order Types
## a) Iceberg Order
```cpp
bool submit_iceberg_order(const string& symbol, OrderSide side, 
                          int total_quantity, int display_quantity,
                          double price, string strategy_id);
```

Purpose: Breaks a large order into smaller “displayed” chunks while hiding the total quantity from the market.

Parameters:

total_quantity → total size of the order
display_quantity → quantity visible in the order book
strategy_id → identifies the trading strategy executing the order

## b) TWAP (Time-Weighted Average Price) Order

```cpp
bool submit_twap_order(const string& symbol, OrderSide side,
                       int total_quantity, milliseconds duration);
```
Purpose: Executes an order evenly over a fixed time duration to reduce market impact.

How it works: Divides the total quantity into smaller slices and submits them at regular intervals.

## c) VWAP (Volume-Weighted Average Price) Order
```cpp
bool submit_vwap_order(const string& symbol, OrderSide side,
                       int total_quantity, const VWAPProfile& profile);
```
Purpose: Executes an order based on market volume profile.

VWAPProfile → specifies how the order should be distributed according to expected trading volume during the day.

Ensures the order tracks the market’s average price and minimizes impact.

-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------

## 3. Advanced Financial Instruments Hierarchy

```cpp
// Base Asset class with comprehensive risk metrics
class Asset {
    // Position management
    virtual void on_trade(double price, int quantity, OrderSide side);
    
    // Risk metrics interface
    virtual double delta() const = 0;      // Price sensitivity
    virtual double gamma() const = 0;      // Delta sensitivity  
    virtual double vega() const = 0;       // Volatility sensitivity
    virtual double theta() const = 0;      // Time decay
    virtual double rho() const = 0;        // Interest rate sensitivity
    
    // Advanced metrics
    virtual double charm() const;          // Delta time decay
    virtual double vanna() const;          // Vega/price cross
    virtual double volga() const;          // Vega convexity
};

// Equity implementation
class Stock : public Asset {
    // Single-stock positions with corporate action handling
    void handle_dividend(double amount);
    void handle_split(int ratio);
    
    // Liquidity metrics
    double average_daily_volume() const;
    double bid_ask_spread() const;
    double short_borrow_cost() const;
};

// Options implementation with multiple pricing models
class EuropeanOption : public Asset {
private:
    // Multiple pricing model support
    double black_scholes_price() const;
    double binomial_tree_price(int steps = 100) const;
    double monte_carlo_price(int simulations = 10000) const;
    
    // Volatility surface modeling
    void update_implied_volatility(double market_price);
    double calculate_implied_volatility() const;
    
    // Dividend and borrow cost modeling
    double dividend_yield_;
    double borrow_cost_;
};
```

## Position management: on_trade() updates the asset’s position whenever a trade occurs (price, quantity, buy/sell).

Risk metrics (the Greeks):

delta() → sensitivity to underlying price
gamma() → sensitivity of delta to price
vega() → sensitivity to volatility
theta() → time decay of the asset’s value
rho() → sensitivity to interest rates


Advanced Greeks:

charm() → delta decay over time
vanna() → sensitivity of delta to volatility changes
volga() → sensitivity of vega to volatility changes


## Stock

(i) Corporate actions:

handle_dividend() → adjust positions and P&L for dividends
handle_split() → adjust position size and price for stock splits

(ii) 1Liquidity metrics:

average_daily_volume() → measures trading volume
bid_ask_spread() → measures liquidity cost
short_borrow_cost() → cost to short the stock


## EuropeanOption

(i) Pricing models:

black_scholes_price() → closed-form formula
binomial_tree_price() → lattice-based numerical method
monte_carlo_price() → simulation-based pricing

(ii) Volatility modeling:

update_implied_volatility() → adjusts model based on observed market prices
calculate_implied_volatility() → reverse-engineers volatility from market prices

(iii) Real-world adjustments:

dividend_yield_ → reduces option value due to expected payouts
borrow_cost_ → cost of holding/shorting underlying


------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

## (4) The AdvancedPortfolio

It is a class is designed as a central engine for managing risk and positions across multiple financial instruments.

It consolidates risk exposure, monitors performance, and allows for automated hedging across an entire portfolio.

It’s particularly useful in quantitative trading, portfolio management, and risk management systems.

``` cpp
4. Portfolio Engine (AdvancedPortfolio)
Purpose: Central risk and position management across all instruments.

cpp
class AdvancedPortfolio {
    // Multi-dimensional risk monitoring
    struct RiskExposure {
        double delta;           // Directional risk
        double gamma;           // Convexity risk  
        double vega;            // Volatility risk
        double theta;           // Time decay
        double correlation_risk;// Cross-asset dependencies
        double liquidity_risk;  // Market impact
        double jump_risk;       // Gap risk
    };
    
    // Advanced position analytics
    PositionAnalytics calculate_position_analytics() const;
    StressTestResults run_stress_tests() const;
    ScenarioAnalysis run_scenario_analysis() const;
    
    // Automated hedging strategies
    void hedge_delta_exposure();
    void hedge_vega_exposure();
    void hedge_gamma_exposure();
    void execute_basket_hedge(const vector<HedgeLeg>& legs);
};
```
## a) Multi-dimensional Risk Monitoring

```cpp
struct RiskExposure {
    double delta;           // Directional risk
    double gamma;           // Convexity risk  
    double vega;            // Volatility risk
    double theta;           // Time decay
    double correlation_risk;// Cross-asset dependencies
    double liquidity_risk;  // Market impact
    double jump_risk;       // Gap risk
};
```
Purpose: Capture all the key risks a portfolio faces.

Risks included:

Delta → sensitivity to price changes in underlying assets
Gamma → rate of change of delta (convexity)
Vega → sensitivity to volatility
Theta → sensitivity to time decay
Correlation risk → exposure due to dependencies between assets
Liquidity risk → potential market impact when trading large positions
Jump risk → risk from sudden price gaps

This struct allows the portfolio engine to calculate, track, and report risk holistically.

## b) Advanced Position Analytics

```cpp
PositionAnalytics calculate_position_analytics() const;
StressTestResults run_stress_tests() const;
ScenarioAnalysis run_scenario_analysis() const;
```

Purpose: Provide detailed insights into portfolio positions.

Key features:

(1) calculate_position_analytics() → computes Greeks, P&L attribution, and other position metrics.

(2) run_stress_tests() → simulates extreme market scenarios to see how the portfolio behaves.

(3) run_scenario_analysis() → evaluates the portfolio under hypothetical market conditions (e.g., rate shocks, volatility spikes).

These functions allow proactive risk management and informed decision-making.

## c) Automated Hedging Strategies
```cpp
void hedge_delta_exposure();
void hedge_vega_exposure();
void hedge_gamma_exposure();
void execute_basket_hedge(const vector<HedgeLeg>& legs);
```
Purpose: Automatically reduce unwanted risk exposures.

Types of hedging:

hedge_delta_exposure() → neutralize directional risk.
hedge_vega_exposure() → neutralize volatility risk.
hedge_gamma_exposure() → stabilize convexity exposure.
execute_basket_hedge() → hedge across multiple instruments simultaneously using a basket of hedge legs.

This ensures the portfolio stays balanced and protected against adverse market moves.

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

## Advanced Risk Management System

## 1. Value at Risk (VaR) Engine

RiskMetrics: Stores different measures of portfolio risk, both traditional and advanced.

VaREngine: Computes Value-at-Risk (VaR) and related metrics for a portfolio using various methodologies.

Together, they are part of a risk management framework for financial portfolios.

```cpp
struct RiskMetrics {
    // Traditional VaR measures
    double var_95;
    double var_99;
    double expected_shortfall_95;
    double expected_shortfall_99;
    
    // Advanced risk metrics
    double incremental_var;         // Marginal contribution to portfolio VaR
    double component_var;           // Component VaR decomposition
    double conditional_var;         // CVaR with extreme value theory
    
    // Liquidity-adjusted VaR
    double lvar;                    // Incorporating market impact
    double endogenous_lvar;         // Position-size dependent liquidity
    
    // Stress testing
    double historical_stress_var;   // Based on worst historical periods
    double hypothetical_stress_var; // User-defined scenarios
};

class VaREngine {
public:
    // Multiple VaR methodologies
    RiskMetrics calculate_historical_var(int lookback_days = 252);
    RiskMetrics calculate_parametic_var(double confidence = 0.95);
    RiskMetrics calculate_monte_carlo_var(int simulations = 100000);
    
    // Advanced features
    void backtest_var();            // Model validation
    double calculate_var_break_frequency(); // Exceedance rate
    void calculate_var_decomposition(); // Risk contribution analysis
};
```
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

## (2) Stress Testing Framework

The StressTestingEngine is designed to assess how a portfolio behaves under extreme market conditions.

(i) It goes beyond standard risk measures (like VaR) by simulating rare, severe events.
(ii) Helps risk managers identify vulnerabilities and plan hedging strategies.

```cpp
class StressTestingEngine {
    // Historical scenarios
    void run_historical_stress_test(const string& period);
    
    // Hypothetical scenarios
    void run_hypothetical_stress_test(const StressScenario& scenario);
    
    // Reverse stress testing
    vector<Scenario> find_breaking_scenarios(double loss_threshold);
    
    // Common stress scenarios:
    // - 1987 Black Monday (-20% equity crash)
    // - 2008 Financial Crisis (Lehman failure)
    // - 2020 COVID Crash (volatility explosion)
    // - 2010 Flash Crash (liquidity disappearance)
};
```
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------

## (3) Greeks Exposure Management

The GreeksMonitor class is designed to track and manage a portfolio’s exposure to the “Greeks” in real time.

Greeks are sensitivity measures used in options and derivatives trading to understand risk.

This class ensures that portfolio exposures stay within risk limits and helps generate hedging recommendations.

## a) Real-time exposure tracking
```cpp
map<string, double> position_greeks;
map<string, double> portfolio_greeks;
```

position_greeks → tracks Greeks (delta, gamma, vega, theta, rho, etc.) for each individual position.

portfolio_greeks → aggregates all position Greeks to show total portfolio exposure.

This is critical for real-time risk monitoring and managing derivative portfolios.

## b) Sensitivity analysis
``` cpp
void calculate_greeks_sensitivities();
void monitor_greeks_limits();
void generate_hedging_recommendations();
```
calculate_greeks_sensitivities() → computes how small changes in market variables (price, volatility, time) affect positions.

monitor_greeks_limits() → checks if exposures exceed predefined risk limits.

generate_hedging_recommendations() → suggests trades to neutralize or reduce unwanted Greek exposures.

## c) Cross-Greek monitoring
```cpp
void monitor_delta_gamma_relationship();
void monitor_vega_theta_tradeoff();
```

Delta-Gamma monitoring → ensures that adjusting delta doesn’t create excessive gamma risk.

Vega-Theta monitoring → balances sensitivity to volatility (vega) against time decay (theta), since these often trade off in options positions.

Helps optimize risk management and avoid unintended exposure shifts.

-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------

## Quantitative Finance Models

## 1. Option Pricing Models

## Black-Scholes-Merton Extension
```cpp
class AdvancedOptionPricer {
    // Enhanced Black-Scholes with real-world adjustments
    double black_scholes_with_borrow(double S, double K, double T, 
                                    double r, double q, double sigma, 
                                    bool is_call, double borrow_cost);
    
    // Stochastic volatility models
    double heston_model_price(int simulations = 10000);
    double sabel_rough_volatility_price();
    
    // Jump diffusion models
    double merton_jump_diffusion_price(double jump_intensity, 
                                      double jump_mean, double jump_std);
};
```

## a) black_scholes_with_borrow

Computes option prices using the classic Black-Scholes formula, but also accounts for borrowing costs (important if you’re shorting the underlying stock).

Input: stock price, strike, time, volatility, interest rate, dividend yield, borrow cost.

Output: option price (call or put).
Use case: quick, realistic option pricing for standard market conditions.

## b) heston_model_price

Uses the Heston stochastic volatility model.
Volatility is not constant, it changes randomly over time.
Simulates the random paths of stock prices and volatility (via Monte Carlo or other methods) to compute the expected discounted payoff.

Use case: more accurate pricing for options in markets with volatility smiles/skews.

## c) sabel_rough_volatility_price

Models rough or irregular volatility, capturing long-memory effects in the market.
Uses advanced stochastic methods to simulate price paths.

Use case: pricing options where market volatility behaves “roughly” over time (common in real-world equities and FX markets).

## d) merton_jump_diffusion_price

Adds jump events to the stock price (sudden market moves, e.g., news or earnings).
Computes the expected payoff by considering both regular diffusion and random jumps.

Use case: pricing options for assets prone to sudden jumps, e.g., small-cap stocks, commodities, or crypto.

## Overall What It’s Doing

Models the stock price realistically:->

(i)   Constant volatility → Black-Scholes
(ii)  Changing volatility → Heston
(iii) Rough paths → Sabel
(iv)  Jumps → Merton

Simulates or solves for the expected option payoff under these models.
Discounts the expected payoff to today’s price (risk-neutral pricing).
Returns the fair option price for the given model and parameters.

<img width="872" height="697" alt="Screenshot 2025-10-20 020846" src="https://github.com/user-attachments/assets/b5bebf68-98a0-415e-a26e-1527886da6ac" />

<img width="821" height="735" alt="Screenshot 2025-10-20 020855" src="https://github.com/user-attachments/assets/03313782-1ef4-4672-81cf-4c34831bbd6a" />


<img width="676" height="499" alt="Screenshot 2025-10-20 020935" src="https://github.com/user-attachments/assets/95fc6272-e1e9-423e-99f3-f9f5949094a5" />
<img width="676" height="763" alt="Screenshot 2025-10-20 020924" src="https://github.com/user-attachments/assets/2ac25500-980c-49ab-966c-d70f8a82ab9b" />
<img width="780" height="666" alt="Screenshot 2025-10-20 020904" src="https://github.com/user-attachments/assets/43e51a50-8b1f-4404-8255-ba6eb8149e46" />





































