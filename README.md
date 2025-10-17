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


