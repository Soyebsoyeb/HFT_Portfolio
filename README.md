## High-Frequency Trading Portfolio Management System
Project Overview

This project implements a High-Frequency Trading (HFT) Portfolio Management System in modern C++. It is designed for real-time portfolio management, risk analytics, and trading execution.

The system simulates market data, handles orders in a multi-threaded environment, evaluates derivative instruments, and computes advanced risk metrics like Value at Risk (VaR), Expected Shortfall (ES), and stress loss.

It demonstrates proficiency in modern C++, concurrent programming, and quantitative finance concepts.

## Problems Addressed

Real-Time Market Simulation: Generating realistic price movements for multiple instruments.

High-Throughput Order Management: Processing thousands of orders per second without delays.

Portfolio Risk Analytics: Measuring portfolio risk in real-time across multiple assets.

Derivative Valuation: Pricing European options and computing Greeks.

Hedging and Exposure Control: Automatic delta and gamma exposure monitoring.

Low-Latency Reporting: Real-time updates of portfolio metrics without affecting trade execution.


## Key Features

Market simulation with bid/ask prices and spreads.

Thread-safe order submission, cancellation, and execution.

Asset classes: stocks and European options.

Portfolio aggregation with real-time PnL and Greek computation.

Monte Carlo-based risk metrics: VaR, ES, and stress testing.

Low-latency concurrent architecture using threads, atomic operations, and lock-free data structures.

Detailed reporting of portfolio and risk metrics.



Core System Functions
1. Mathematical & Utility Functions
Math::normal_pdf(double x)
Purpose: Calculate Probability Density Function of standard normal distribution
Usage: Black-Scholes option pricing and Greek calculations

cpp
// Used in option pricing for d1/d2 calculations
double pdf_d1 = Math::normal_pdf(d1);  // Probability density at d1
Math::normal_cdf(double x)
Purpose: Calculate Cumulative Distribution Function of standard normal
Usage: Option probability calculations in Black-Scholes

cpp
// Calculate probability for option exercise
double Nd1 = Math::normal_cdf(d1);  // P(Z ≤ d1) where Z ~ N(0,1)
double Nd2 = Math::normal_cdf(d2);  // P(Z ≤ d2)
Math::normal_quantile(double p)
Purpose: Inverse normal CDF (Acklam's algorithm)
Usage: VaR calculations and critical value determination

cpp
// Calculate z-score for confidence levels
double z_95 = Math::normal_quantile(0.95);  // ≈1.645 for 95% VaR
double z_99 = Math::normal_quantile(0.99);  // ≈2.326 for 99% VaR
Math::CholeskyDecomp Class
Purpose: Perform Cholesky decomposition for correlated simulations
Usage: Generate correlated random variables for Monte Carlo VaR

cpp
Math::CholeskyDecomp cholesky(covariance_matrix);
if (cholesky.is_valid()) {
    vector<double> correlated = cholesky.generate_correlated_normal(uncorrelated);
    // Use for correlated asset price simulations
}
gaussian()
Purpose: Generate standard normal random variables
Usage: Monte Carlo simulations and price path generation

cpp
// Generate random shocks for price simulations
double random_shock = gaussian();  // ~N(0,1)
double price_change = volatility * sqrt(dt) * random_shock;
2. Market Data Functions
MarketDataFeed::add_symbol(string symbol, double price)
Purpose: Register a financial instrument for market data updates
Usage: Initialize stocks/options with starting prices

cpp
market_data.add_symbol("AAPL", 150.0);   // Add Apple at $150
market_data.add_symbol("GOOGL", 2800.0); // Add Google at $2800
MarketDataFeed::get_last_price(string symbol)
Purpose: Retrieve current market price for an instrument
Usage: Portfolio valuation and risk calculations

cpp
auto price = market_data.get_last_price("AAPL");
if (price) {
    double current_value = price.value() * quantity;
}
MarketDataFeed::get_bid_price() / get_ask_price()
Purpose: Access bid-ask spread information
Usage: Realistic order pricing and market impact modeling

cpp
double bid = market_data.get_bid_price("AAPL").value_or(0.0);
double ask = market_data.get_ask_price("AAPL").value_or(0.0);
double spread = ask - bid;  // Calculate market liquidity
3. Order Management Functions
OrderManager::submit_order()
Purpose: Place new trading orders (market/limit)
Usage: Execute portfolio rebalancing and risk management trades

cpp
// Market order: immediate execution at current price
order_manager.submit_order("AAPL", OrderType::MARKET, OrderSide::BUY, 0.0, 100);

// Limit order: execution at specified price or better  
order_manager.submit_order("GOOGL", OrderType::LIMIT, OrderSide::SELL, 2810.0, 10);
OrderManager::cancel_order(string order_id)
Purpose: Cancel pending orders
Usage: Risk control and strategy adjustment

cpp
bool success = order_manager.cancel_order("ORD_12345");
if (success) {
    cout << "Order cancelled successfully" << endl;
}
OrderManager::process_order_queue()
Purpose: Process all pending orders through execution callback
Usage: Simulate order execution and update positions

cpp
order_manager.process_order_queue([&](const Order& order) -> bool {
    auto asset = portfolio.get_asset(order.symbol);
    if (asset) {
        asset->on_trade(order.price, order.quantity, order.side);
        return true;  // Order executed successfully
    }
    return false;  // Order rejected
});
4. Asset Management Functions
Stock::Stock(string symbol, double price, double vol, ...)
Purpose: Create stock position with specified parameters
Usage: Portfolio construction and instrument setup

cpp
// Create Apple stock: symbol, price, volatility, dividend yield
auto aapl = make_shared<Stock>("AAPL", 150.0, 0.25, 0.005);
portfolio.add_asset(aapl);
EuropeanOption::EuropeanOption(...)
Purpose: Create option position with full specification
Usage: Derivatives trading and complex strategies

cpp
// Create call option: symbol, underlying, strike, expiry, is_call, implied_vol
auto call = make_shared<EuropeanOption>("AAPL_150C", aapl_stock, 
                                       150.0, 0.1, true, 0.26);
portfolio.add_asset(call);
Asset::on_trade(double price, int quantity, OrderSide side)
Purpose: Update position when trade occurs
Usage: Maintain accurate position tracking and P&L

cpp
// Called when order executes
asset->on_trade(152.50, 100, OrderSide::BUY);  // Buy 100 shares at $152.50
// Updates average cost, quantity, and realized P&L
Asset::update_pnl()
Purpose: Recalculate unrealized profit/loss
Usage: Real-time performance monitoring

cpp
asset->update_pnl();  // Updates based on current market price vs average cost
cout << "Current PnL: $" << asset->unrealized_pnl << endl;
5. Portfolio Analysis Functions
AdvancedPortfolio::total_market_value()
Purpose: Calculate total portfolio value
Usage: Portfolio sizing and performance measurement

cpp
double portfolio_value = portfolio.total_market_value();
cout << "Portfolio Value: $" << portfolio_value << endl;
AdvancedPortfolio::total_delta() / total_gamma() / total_vega() / total_theta()
Purpose: Aggregate Greek exposures across all positions
Usage: Portfolio risk management and hedging decisions

cpp
double net_delta = portfolio.total_delta();    // Price sensitivity
double net_gamma = portfolio.total_gamma();    // Delta sensitivity  
double net_vega = portfolio.total_vega();      // Volatility sensitivity
double net_theta = portfolio.total_theta();    // Time decay

if (abs(net_delta) > risk_limit) {
    cout << "WARNING: Delta exposure limit exceeded!" << endl;
}
AdvancedPortfolio::compute_risk_metrics()
Purpose: Calculate comprehensive risk measures (VaR, Expected Shortfall, Stress Tests)
Usage: Regulatory reporting and risk monitoring

cpp
auto risk = portfolio.compute_risk_metrics(10000);  // 10,000 MC simulations

cout << "VaR 95%: $" << risk.var_95 << endl;        // 95% confidence loss
cout << "ES 95%: $" << risk.expected_shortfall_95 << endl;  // Average tail loss
cout << "Stress Loss: $" << risk.stress_loss << endl; // 5-sigma extreme scenario
6. Risk Management Functions
AdvancedPortfolio::enforce_risk_limits()
Purpose: Check and enforce portfolio risk limits
Usage: Automated risk control and compliance

cpp
// Called after each trade or price update
portfolio.enforce_risk_limits();

// Automatically triggers:
// - Delta hedging if delta > max_delta
// - Vega warnings if vega > max_vega
// - Gamma monitoring if gamma > max_gamma
AdvancedPortfolio::hedge_delta_exposure()
Purpose: Automatically hedge delta risk
Usage: Maintain market-neutral or controlled delta exposure

cpp
// When delta exceeds limits:
portfolio.hedge_delta_exposure();
// Would execute offsetting trades to bring delta within limits
AdvancedPortfolio::update_correlation_matrix()
Purpose: Maintain correlation structure for risk calculations
Usage: Accurate covariance modeling in VaR calculations

cpp
// Called when assets are added/removed
portfolio.update_correlation_matrix();
// Updates correlation assumptions between all instruments
7. Trading Execution Functions
AdvancedPortfolio::submit_market_order()
Purpose: Execute immediate trades at current market prices
Usage: Quick position adjustments and urgent risk management

cpp
// Buy 100 AAPL shares at current market price
bool success = portfolio.submit_market_order("AAPL", OrderSide::BUY, 100);
if (success) {
    cout << "Market order submitted" << endl;
}
AdvancedPortfolio::submit_limit_order()
Purpose: Place conditional trades at specified price levels
Usage: Cost-controlled execution and patient order placement

cpp
// Sell 10 GOOGL shares if price reaches $2810
bool success = portfolio.submit_limit_order("GOOGL", OrderSide::SELL, 2810.0, 10);
8. Monitoring & Reporting Functions
AdvancedPortfolio::print_detailed_report()
Purpose: Generate comprehensive portfolio analysis
Usage: Risk reporting, performance review, and regulatory compliance

cpp
portfolio.print_detailed_report();
// Outputs:
// - Portfolio summary (value, Greeks)
// - Risk metrics (VaR, ES, stress tests)  
// - Detailed positions with P&L
// - Active orders status
PortfolioMonitor::monitoring_loop()
Purpose: Continuous real-time portfolio monitoring
Usage: Live risk management and performance tracking

cpp
// Runs at 10Hz (100ms intervals)
// - Updates all prices and calculations
// - Processes pending orders
// - Enforces risk limits
// - Displays current status
🛠️ Problem-Solving Applications
Scenario 1: Risk Limit Breach Detection
cpp
// Problem: Portfolio delta exceeds $250,000 limit
// Solution: Automatic detection and hedging

void handle_delta_breach() {
    double current_delta = portfolio.total_delta();
    if (abs(current_delta) > 250000.0) {
        // 1. Detect the breach
        cout << "ALERT: Delta limit breached: " << current_delta << endl;
        
        // 2. Auto-hedge
        portfolio.hedge_delta_exposure();
        
        // 3. Verify resolution
        double new_delta = portfolio.total_delta();
        cout << "Delta after hedging: " << new_delta << endl;
    }
}
Scenario 2: VaR-Based Position Sizing
cpp
// Problem: Determine maximum position size within risk limits
// Solution: Use VaR calculations for position sizing

double calculate_max_position_size(const string& symbol, double confidence = 0.99) {
    auto risk = portfolio.compute_risk_metrics();
    double current_var = risk.var_99;  // 99% VaR
    
    // Calculate incremental VaR for new position
    double max_additional_risk = risk_limit - current_var;
    
    // Estimate position size that stays within risk budget
    double asset_vol = portfolio.get_asset(symbol)->volatility();
    double position_size = max_additional_risk / (asset_vol * 2.33); // 99% z-score
    
    return position_size;
}
Scenario 3: Options Strategy Risk Analysis
cpp
// Problem: Understand risk profile of complex options positions
// Solution: Comprehensive Greek analysis and scenario testing

void analyze_options_strategy() {
    // Add option positions
    auto call = make_shared<EuropeanOption>("CALL", stock, 150.0, 0.25, true, 0.30);
    auto put = make_shared<EuropeanOption>("PUT", stock, 140.0, 0.25, false, 0.28);
    
    portfolio.add_asset(call);
    portfolio.add_asset(put);
    
    // Analyze risk profile
    cout << "Strategy Greeks:" << endl;
    cout << "Net Delta: " << portfolio.total_delta() << endl;
    cout << "Net Gamma: " << portfolio.total_gamma() << endl; 
    cout << "Net Vega: " << portfolio.total_vega() << endl;
    cout << "Net Theta: " << portfolio.total_theta() << endl;
    
    // Stress test
    auto risk = portfolio.compute_risk_metrics();
    cout << "Max 1-day loss (99%): $" << risk.var_99 << endl;
}
Scenario 4: Real-time Portfolio Monitoring
cpp
// Problem: Monitor large portfolio in real-time for risk control
// Solution: Automated 10Hz monitoring system

void setup_real_time_monitoring() {
    PortfolioMonitor monitor(portfolio);
    monitor.start();  // Begins 10Hz monitoring loop
    
    // Monitor continuously runs:
    // - Price updates and P&L calculation
    // - Risk metric recomputation  
    // - Order processing
    // - Limit enforcement
    // - Live reporting
    
    // After 30 seconds...
    this_thread::sleep_for(seconds(30));
    monitor.stop();
    
    // Final risk report
    portfolio.print_detailed_report();
}
Scenario 5: Correlation-Aware Risk Management
cpp
// Problem: Account for asset correlations in risk calculations  
// Solution: Cholesky-based Monte Carlo simulations

void correlated_risk_analysis() {
    // Update correlations based on market regime
    portfolio.update_correlation_matrix();
    
    // Run correlated Monte Carlo VaR
    auto risk = portfolio.compute_risk_metrics(10000);
    
    cout << "Correlation-aware Risk Metrics:" << endl;
    cout << "VaR 95%: $" << risk.var_95 << endl;
    cout << "Expected Shortfall 95%: $" << risk.expected_shortfall_95 << endl;
    
    // Compare with uncorrelated assumption
    // (Diversification benefit quantification)
}
📈 Practical Usage Examples
Daily Risk Report Generation
cpp
void generate_daily_risk_report() {
    // 1. Update all market data
    market_data.start();
    
    // 2. Calculate overnight risk
    auto overnight_risk = portfolio.compute_risk_metrics(5000);
    
    // 3. Generate comprehensive report
    portfolio.print_detailed_report();
    
    // 4. Check compliance with limits
    portfolio.enforce_risk_limits();
    
    // 5. Log results for regulatory purposes
    portfolio.logSnapshot("daily_risk_report.csv");
}
Pre-Trade Analysis
cpp
bool pre_trade_analysis(const string& symbol, OrderSide side, int quantity) {
    // 1. Check current risk metrics
    auto current_risk = portfolio.compute_risk_metrics();
    
    // 2. Simulate proposed trade
    auto temp_portfolio = portfolio;  // Copy for simulation
    temp_portfolio.submit_market_order(symbol, side, quantity);
    temp_portfolio.process_trades();
    
    // 3. Analyze impact
    auto new_risk = temp_portfolio.compute_risk_metrics();
    
    // 4. Decision criteria
    if (new_risk.var_99 > risk_limit * 1.1) {
        cout << "REJECTED: Trade exceeds risk limits" << endl;
        return false;
    }
    
    cout << "APPROVED: Trade within risk parameters" << endl;
    return true;
}



