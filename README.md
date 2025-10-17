# High-Frequency Trading Portfolio Management System

## Project Overview

This project implements a High-Frequency Trading (HFT) Portfolio Management System in modern C++. It is designed for real-time portfolio management, risk analytics, and trading execution.

The system simulates market data, handles orders in a multi-threaded environment, evaluates derivative instruments, and computes advanced risk metrics like Value at Risk (VaR), Expected Shortfall (ES), and stress loss.

It demonstrates proficiency in modern C++, concurrent programming, and quantitative finance concepts.

---

## Problems Addressed

- **Real-Time Market Simulation:** Generating realistic price movements for multiple instruments.
- **High-Throughput Order Management:** Processing thousands of orders per second without delays.
- **Portfolio Risk Analytics:** Measuring portfolio risk in real-time across multiple assets.
- **Derivative Valuation:** Pricing European options and computing Greeks.
- **Hedging and Exposure Control:** Automatic delta and gamma exposure monitoring.
- **Low-Latency Reporting:** Real-time updates of portfolio metrics without affecting trade execution.

---

## Key Features

- Market simulation with bid/ask prices and spreads.
- Thread-safe order submission, cancellation, and execution.
- Asset classes: stocks and European options.
- Portfolio aggregation with real-time PnL and Greek computation.
- Monte Carlo-based risk metrics: VaR, ES, and stress testing.
- Low-latency concurrent architecture using threads, atomic operations, and lock-free data structures.
- Detailed reporting of portfolio and risk metrics.

---

## Core System Functions

### 1. Mathematical & Utility Functions

- **`Math::normal_pdf(double x)`**  
  **Purpose:** Calculate Probability Density Function of standard normal distribution  
  **Usage:** Black-Scholes option pricing and Greek calculations  

```cpp
double pdf_d1 = Math::normal_pdf(d1);  // Probability density at d1
Math::normal_cdf(double x)
Purpose: Calculate Cumulative Distribution Function of standard normal
Usage: Option probability calculations in Black-Scholes

cpp
Copy code
double Nd1 = Math::normal_cdf(d1);  
double Nd2 = Math::normal_cdf(d2);
Math::normal_quantile(double p)
Purpose: Inverse normal CDF (Acklam's algorithm)
Usage: VaR calculations and critical value determination

cpp
Copy code
double z_95 = Math::normal_quantile(0.95);  
double z_99 = Math::normal_quantile(0.99);
Math::CholeskyDecomp
Purpose: Perform Cholesky decomposition for correlated simulations
Usage: Generate correlated random variables for Monte Carlo VaR

cpp
Copy code
Math::CholeskyDecomp cholesky(covariance_matrix);
if (cholesky.is_valid()) {
    vector<double> correlated = cholesky.generate_correlated_normal(uncorrelated);
}
gaussian()
Purpose: Generate standard normal random variables
Usage: Monte Carlo simulations and price path generation

cpp
Copy code
double random_shock = gaussian();  
double price_change = volatility * sqrt(dt) * random_shock;
2. Market Data Functions
MarketDataFeed::add_symbol(string symbol, double price)
Purpose: Register a financial instrument for market data updates

cpp
Copy code
market_data.add_symbol("AAPL", 150.0);   
market_data.add_symbol("GOOGL", 2800.0);
MarketDataFeed::get_last_price(string symbol)
Purpose: Retrieve current market price for an instrument

cpp
Copy code
auto price = market_data.get_last_price("AAPL");
if (price) {
    double current_value = price.value() * quantity;
}
MarketDataFeed::get_bid_price() / get_ask_price()
Purpose: Access bid-ask spread information

cpp
Copy code
double bid = market_data.get_bid_price("AAPL").value_or(0.0);
double ask = market_data.get_ask_price("AAPL").value_or(0.0);
double spread = ask - bid;
3. Order Management Functions
OrderManager::submit_order()
Purpose: Place new trading orders (market/limit)

cpp
Copy code
order_manager.submit_order("AAPL", OrderType::MARKET, OrderSide::BUY, 0.0, 100);
order_manager.submit_order("GOOGL", OrderType::LIMIT, OrderSide::SELL, 2810.0, 10);
OrderManager::cancel_order(string order_id)
Purpose: Cancel pending orders

cpp
Copy code
bool success = order_manager.cancel_order("ORD_12345");
if (success) {
    cout << "Order cancelled successfully" << endl;
}
OrderManager::process_order_queue()
Purpose: Process all pending orders through execution callback

cpp
Copy code
order_manager.process_order_queue([&](const Order& order) -> bool {
    auto asset = portfolio.get_asset(order.symbol);
    if (asset) {
        asset->on_trade(order.price, order.quantity, order.side);
        return true;
    }
    return false;
});
4. Asset Management Functions
Stock::Stock(string symbol, double price, double vol, ...)
Purpose: Create stock position with specified parameters

cpp
Copy code
auto aapl = make_shared<Stock>("AAPL", 150.0, 0.25, 0.005);
portfolio.add_asset(aapl);
EuropeanOption::EuropeanOption(...)
Purpose: Create option position with full specification

cpp
Copy code
auto call = make_shared<EuropeanOption>("AAPL_150C", aapl_stock, 150.0, 0.1, true, 0.26);
portfolio.add_asset(call);
Asset::on_trade(double price, int quantity, OrderSide side)
Purpose: Update position when trade occurs

cpp
Copy code
asset->on_trade(152.50, 100, OrderSide::BUY);
Asset::update_pnl()
Purpose: Recalculate unrealized profit/loss

cpp
Copy code
asset->update_pnl();  
cout << "Current PnL: $" << asset->unrealized_pnl << endl;
5. Portfolio Analysis Functions
AdvancedPortfolio::total_market_value()
Purpose: Calculate total portfolio value

cpp
Copy code
double portfolio_value = portfolio.total_market_value();
cout << "Portfolio Value: $" << portfolio_value << endl;
AdvancedPortfolio::total_delta() / total_gamma() / total_vega() / total_theta()
Purpose: Aggregate Greek exposures

cpp
Copy code
double net_delta = portfolio.total_delta();
double net_gamma = portfolio.total_gamma();
double net_vega = portfolio.total_vega();
double net_theta = portfolio.total_theta();

if (abs(net_delta) > risk_limit) {
    cout << "WARNING: Delta exposure limit exceeded!" << endl;
}
AdvancedPortfolio::compute_risk_metrics()
Purpose: Calculate VaR, Expected Shortfall, and stress tests

cpp
Copy code
auto risk = portfolio.compute_risk_metrics(10000);

cout << "VaR 95%: $" << risk.var_95 << endl;
cout << "ES 95%: $" << risk.expected_shortfall_95 << endl;
cout << "Stress Loss: $" << risk.stress_loss << endl;
6. Risk Management Functions
AdvancedPortfolio::enforce_risk_limits()
Purpose: Automated risk control

cpp
Copy code
portfolio.enforce_risk_limits();
AdvancedPortfolio::hedge_delta_exposure()
Purpose: Automatically hedge delta risk

cpp
Copy code
portfolio.hedge_delta_exposure();
AdvancedPortfolio::update_correlation_matrix()
Purpose: Maintain correlation structure for risk calculations

cpp
Copy code
portfolio.update_correlation_matrix();
7. Trading Execution Functions
AdvancedPortfolio::submit_market_order()

cpp
Copy code
bool success = portfolio.submit_market_order("AAPL", OrderSide::BUY, 100);
if (success) cout << "Market order submitted" << endl;
AdvancedPortfolio::submit_limit_order()

cpp
Copy code
bool success = portfolio.submit_limit_order("GOOGL", OrderSide::SELL, 2810.0, 10);
8. Monitoring & Reporting Functions
AdvancedPortfolio::print_detailed_report()

cpp
Copy code
portfolio.print_detailed_report();
PortfolioMonitor::monitoring_loop()

cpp
Copy code
PortfolioMonitor monitor(portfolio);
monitor.start();
this_thread::sleep_for(seconds(30));
monitor.stop();
portfolio.print_detailed_report();
Problem-Solving Applications
Risk Limit Breach Detection

VaR-Based Position Sizing

Options Strategy Risk Analysis

Real-time Portfolio Monitoring

Correlation-Aware Risk Management
