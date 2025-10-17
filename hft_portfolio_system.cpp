// Compile: g++ -std=c++17 -O3 -march=native -pthread -DNDEBUG hft_portfolio_system.cpp -o hft_portfolio

#include <bits/stdc++.h>
#include <atomic>
#include <cmath>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <optional>

using namespace std;
using namespace chrono;

// ==================== Configuration Constants ====================
namespace Config {
    constexpr bool ENABLE_LOGGING = true;
    constexpr bool ENABLE_METRICS = true;
    constexpr size_t MAX_PORTFOLIO_SIZE = 1000;
    constexpr size_t ORDER_QUEUE_SIZE = 10000;
    constexpr double RISK_FREE_RATE = 0.02; // 2% annual
    constexpr double INITIAL_MARGIN_REQ = 0.50; // 50% initial margin
    constexpr double MAINTENANCE_MARGIN = 0.25; // 25% maintenance margin
    constexpr double MAX_POSITION_SIZE = 1000000.0; // $1M max position
    constexpr double MAX_DELTA_EXPOSURE = 500000.0; // $500K max delta
    constexpr double MAX_VEGA_EXPOSURE = 200000.0; // $200K max vega
    constexpr int MC_SIMS_FAST = 1000;
    constexpr int MC_SIMS_ACCURATE = 10000;
    constexpr double VAR_CONFIDENCE = 0.99;
    constexpr int VAR_HORIZON_DAYS = 1;
}

// ==================== Global RNG ====================
mt19937_64& get_global_rng() {
    static mt19937_64 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
    return rng;
}

double gaussian() {
    static normal_distribution<double> dist(0.0, 1.0);
    return dist(get_global_rng());
}

// ==================== High Precision Timing ====================
class NanoTimer {
    steady_clock::time_point start;
public:
    NanoTimer() : start(steady_clock::now()) {}
    double elapsed() const {
        return duration_cast<nanoseconds>(steady_clock::now() - start).count() * 1e-9;
    }
    void reset() { start = steady_clock::now(); }
};

// ==================== Lock-Free Data Structures ====================
template<typename T, size_t Size>
class CircularBuffer {
    array<T, Size> buffer;
    atomic<size_t> head{0}, tail{0};
    
public:
    CircularBuffer() = default; // Add default constructor
    
    bool push(const T& item) {
        size_t current_tail = tail.load(memory_order_relaxed);
        size_t next_tail = (current_tail + 1) % Size;
        if (next_tail == head.load(memory_order_acquire)) return false;
        buffer[current_tail] = item;
        tail.store(next_tail, memory_order_release);
        return true;
    }
    
    bool pop(T& item) {
        size_t current_head = head.load(memory_order_relaxed);
        if (current_head == tail.load(memory_order_acquire)) return false;
        item = buffer[current_head];
        head.store((current_head + 1) % Size, memory_order_release);
        return true;
    }
    
    bool empty() const {
        return head.load(memory_order_acquire) == tail.load(memory_order_acquire);
    }
};

// ==================== Advanced Math Utilities ====================
namespace Math {
    constexpr double PI = 3.14159265358979323846;
    constexpr double SQRT_2PI = 2.506628274631000502415765284811;
    constexpr double INV_SQRT_2PI = 0.398942280401432677939946059934;
    
    inline double normal_pdf(double x) {
        return INV_SQRT_2PI * exp(-0.5 * x * x);
    }
    
    inline double normal_cdf(double x) {
        return 0.5 * (1.0 + erf(x / sqrt(2.0)));
    }
    
    // Fast inverse normal CDF (Acklam's algorithm)
    inline double normal_quantile(double p) {
        static const double a1 = -3.969683028665376e+01;
        static const double a2 =  2.209460984245205e+02;
        static const double a3 = -2.759285104469687e+02;
        static const double a4 =  1.383577518672690e+02;
        static const double a5 = -3.066479806614716e+01;
        static const double a6 =  2.506628277459239e+00;
        
        static const double b1 = -5.447609879822406e+01;
        static const double b2 =  1.615858368580409e+02;
        static const double b3 = -1.556989798598866e+02;
        static const double b4 =  6.680131188771972e+01;
        static const double b5 = -1.328068155288572e+01;
        
        static const double c1 = -7.784894002430293e-03;
        static const double c2 = -3.223964580411365e-01;
        static const double c3 = -2.400758277161838e+00;
        static const double c4 = -2.549732539343734e+00;
        static const double c5 =  4.374664141464968e+00;
        static const double c6 =  2.938163982698783e+00;
        
        static const double d1 =  7.784695709041462e-03;
        static const double d2 =  3.224671290700398e-01;
        static const double d3 =  2.445134137142996e+00;
        static const double d4 =  3.754408661907416e+00;
        
        double q, r;
        
        if (p < 0.02425) {
            q = sqrt(-2.0 * log(p));
            return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                   ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
        } else if (p > 0.97575) {
            q = sqrt(-2.0 * log(1.0 - p));
            return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                    ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
        } else {
            q = p - 0.5;
            r = q * q;
            return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
                   (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
        }
    }
    
    class CholeskyDecomp {
        vector<vector<double>> L;
        bool valid;
    public:
        CholeskyDecomp(const vector<vector<double>>& A) : valid(false) {
            const size_t n = A.size();
            if (n == 0) return;
            
            L.assign(n, vector<double>(n, 0.0));
            
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    double sum = A[i][j];
                    for (size_t k = 0; k < j; ++k) {
                        sum -= L[i][k] * L[j][k];
                    }
                    if (i == j) {
                        if (sum <= 0.0) return;
                        L[i][j] = sqrt(sum);
                    } else {
                        L[i][j] = sum / L[j][j];
                    }
                }
            }
            valid = true;
        }
        
        bool is_valid() const { return valid; }
        
        vector<double> solve(const vector<double>& b) const {
            if (!valid || L.empty()) return {};
            const size_t n = L.size();
            vector<double> y(n, 0.0), x(n, 0.0);
            
            // Forward substitution: L * y = b
            for (size_t i = 0; i < n; ++i) {
                double sum = b[i];
                for (size_t j = 0; j < i; ++j) {
                    sum -= L[i][j] * y[j];
                }
                y[i] = sum / L[i][i];
            }
            
            // Backward substitution: L^T * x = y
            for (int i = n-1; i >= 0; --i) {
                double sum = y[i];
                for (size_t j = i+1; j < n; ++j) {
                    sum -= L[j][i] * x[j];
                }
                x[i] = sum / L[i][i];
            }
            return x;
        }
        
        vector<double> generate_correlated_normal(const vector<double>& uncorrelated) const {
            if (!valid || L.empty()) return {};
            const size_t n = L.size();
            vector<double> correlated(n, 0.0);
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    correlated[i] += L[i][j] * uncorrelated[j];
                }
            }
            return correlated;
        }
    };
}

// ==================== Market Data Feed Simulation ====================
class MarketDataFeed {
    mutable shared_mutex mutex_;
    unordered_map<string, double> last_prices;
    unordered_map<string, double> bid_prices;
    unordered_map<string, double> ask_prices;
    unordered_map<string, double> volumes;
    atomic<bool> running{false};
    thread feed_thread;
    
    void generate_market_data() {
        static mt19937_64 rng(chrono::high_resolution_clock::now().time_since_epoch().count());
        normal_distribution<double> returns_dist(0.0, 0.01); // 1% daily vol
        
        while (running.load(memory_order_acquire)) {
            {
                unique_lock<shared_mutex> lock(mutex_);
                for (auto& [symbol, price] : last_prices) {
                    double ret = returns_dist(rng);
                    double new_price = price * exp(ret);
                    
                    // Update market data with realistic spreads
                    double spread = new_price * 0.0001; // 1bp spread
                    last_prices[symbol] = new_price;
                    bid_prices[symbol] = new_price - spread/2;
                    ask_prices[symbol] = new_price + spread/2;
                    volumes[symbol] = abs(normal_distribution<double>(100000, 50000)(rng));
                }
            }
            this_thread::sleep_for(milliseconds(100)); // 10 updates per second
        }
    }
    
public:
    MarketDataFeed() = default;
    ~MarketDataFeed() { stop(); }
    
    void start() {
        if (running.load()) return;
        running.store(true, memory_order_release);
        feed_thread = thread(&MarketDataFeed::generate_market_data, this);
    }
    
    void stop() {
        running.store(false, memory_order_release);
        if (feed_thread.joinable()) feed_thread.join();
    }
    
    void add_symbol(const string& symbol, double initial_price) {
        unique_lock<shared_mutex> lock(mutex_);
        last_prices[symbol] = initial_price;
        bid_prices[symbol] = initial_price * 0.9995;
        ask_prices[symbol] = initial_price * 1.0005;
        volumes[symbol] = 100000;
    }
    
    optional<double> get_last_price(const string& symbol) const {
        shared_lock<shared_mutex> lock(mutex_);
        auto it = last_prices.find(symbol);
        return it != last_prices.end() ? optional<double>(it->second) : nullopt;
    }
    
    optional<double> get_bid_price(const string& symbol) const {
        shared_lock<shared_mutex> lock(mutex_);
        auto it = bid_prices.find(symbol);
        return it != bid_prices.end() ? optional<double>(it->second) : nullopt;
    }
    
    optional<double> get_ask_price(const string& symbol) const {
        shared_lock<shared_mutex> lock(mutex_);
        auto it = ask_prices.find(symbol);
        return it != ask_prices.end() ? optional<double>(it->second) : nullopt;
    }
    
    unordered_map<string, double> snapshot() const {
        shared_lock<shared_mutex> lock(mutex_);
        return last_prices;
    }
};

// ==================== Order Management System ====================
enum class OrderType { MARKET, LIMIT, STOP };
enum class OrderSide { BUY, SELL };

struct Order {
    string id;
    string symbol;
    OrderType type;
    OrderSide side;
    double price;
    int quantity;
    long timestamp;
    string strategy_id;
    
    // Default constructor for CircularBuffer
    Order() : id(""), symbol(""), type(OrderType::MARKET), side(OrderSide::BUY), 
              price(0.0), quantity(0), timestamp(0), strategy_id("") {}
    
    Order(string id, string sym, OrderType t, OrderSide s, double p, int q, string strat = "")
        : id(move(id)), symbol(move(sym)), type(t), side(s), price(p), quantity(q), 
          timestamp(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count()),
          strategy_id(move(strat)) {}
};

class OrderManager {
    mutable shared_mutex mutex_;
    unordered_map<string, Order> active_orders;
    unordered_map<string, vector<Order>> order_history;
    CircularBuffer<Order, Config::ORDER_QUEUE_SIZE> order_queue;
    atomic<long> order_id_counter{0};
    
    string generate_order_id() {
        return "ORD_" + to_string(++order_id_counter) + "_" + to_string(
            duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
    }
    
public:
    OrderManager() = default; // Add default constructor
    
    bool submit_order(const string& symbol, OrderType type, OrderSide side, double price, int quantity, const string& strategy_id = "") {
        string id = generate_order_id();
        Order order(id, symbol, type, side, price, quantity, strategy_id);
        
        if (!order_queue.push(order)) return false;
        
        {
            unique_lock<shared_mutex> lock(mutex_);
            active_orders[id] = order;
        }
        return true;
    }
    
    bool cancel_order(const string& order_id) {
        unique_lock<shared_mutex> lock(mutex_);
        return active_orders.erase(order_id) > 0;
    }
    
    vector<Order> get_active_orders() const {
        shared_lock<shared_mutex> lock(mutex_);
        vector<Order> orders;
        for (const auto& [id, order] : active_orders) {
            orders.push_back(order);
        }
        return orders;
    }
    
    void process_order_queue(function<bool(const Order&)> execution_callback) {
        Order order;
        while (order_queue.pop(order)) {
            bool executed = execution_callback(order);
            {
                unique_lock<shared_mutex> lock(mutex_);
                active_orders.erase(order.id);
                order_history[order.symbol].push_back(order);
            }
        }
    }
};

// ==================== Advanced Financial Instruments ====================
class Asset {
protected:
    mutable shared_mutex mutex_;
    
public:
    string name;
    string symbol;
    double quantity;
    double average_cost;
    double unrealized_pnl;
    double realized_pnl;
    mutable atomic<double> last_market_price{0.0};
    mutable atomic<double> bid_price{0.0};
    mutable atomic<double> ask_price{0.0};
    
    Asset(string name, string symbol, double qty = 0.0, double cost = 0.0)
        : name(move(name)), symbol(move(symbol)), quantity(qty), average_cost(cost),
          unrealized_pnl(0.0), realized_pnl(0.0) {}
    
    virtual ~Asset() = default;
    
    virtual double get_market_price() const = 0;
    virtual double market_value() const { 
        return get_market_price() * quantity; 
    }
    
    virtual double volatility() const = 0;
    virtual double delta() const { return 0.0; }
    virtual double gamma() const { return 0.0; }
    virtual double vega() const { return 0.0; }
    virtual double theta() const { return 0.0; }
    virtual double rho() const { return 0.0; }
    
    virtual void update_pnl() {
        double current_price = get_market_price();
        unrealized_pnl = (current_price - average_cost) * quantity;
    }
    
    virtual void on_trade(double price, int trade_quantity, OrderSide side) {
        unique_lock<shared_mutex> lock(mutex_);
        double trade_value = price * trade_quantity;
        
        if ((side == OrderSide::BUY && trade_quantity > 0) || (side == OrderSide::SELL && trade_quantity < 0)) {
            // Adding to position
            double total_cost = average_cost * this->quantity + trade_value;
            this->quantity += trade_quantity;
            average_cost = this->quantity != 0 ? total_cost / this->quantity : 0.0;
        } else {
            // Reducing position
            double realized = (price - average_cost) * trade_quantity;
            realized_pnl += realized;
            this->quantity += trade_quantity;
        }
        update_pnl();
    }
    
    virtual string info() const = 0;
    virtual string risk_report() const = 0;
};

class Stock : public Asset {
    double price_;
    double vol_;
    double dividend_yield_;
    
public:
    Stock(string symbol, double price, double vol, double div_yield = 0.0, double qty = 0.0, double cost = 0.0)
        : Asset(move(symbol), move(symbol), qty, cost), price_(price), vol_(vol), dividend_yield_(div_yield) {
        last_market_price.store(price, memory_order_relaxed);
    }
    
    double get_market_price() const override { 
        return price_; 
    }
    
    void set_market_price(double price) {
        unique_lock<shared_mutex> lock(mutex_);
        price_ = price;
        last_market_price.store(price, memory_order_relaxed);
        update_pnl();
    }
    
    double volatility() const override { return vol_; }
    double delta() const override { return 1.0; } // Stock delta is always 1.0
    
    string info() const override {
        shared_lock<shared_mutex> lock(mutex_);
        ostringstream oss;
        oss << "STOCK " << symbol << " | Price: $" << fixed << setprecision(2) << price_
            << " | Qty: " << quantity << " | Vol: " << vol_
            << " | MV: $" << market_value() << " | PnL: $" << unrealized_pnl;
        return oss.str();
    }
    
    string risk_report() const override {
        ostringstream oss;
        oss << "Stock " << symbol << " | Delta: 1.0 | Gamma: 0.0 | Vega: 0.0";
        return oss.str();
    }
};

class EuropeanOption : public Asset {
    weak_ptr<Stock> underlying_;
    double strike_;
    double expiry_; // years
    double impl_vol_;
    double rf_rate_;
    bool is_call_;
    
    // Black-Scholes cached calculations
    mutable atomic<double> cached_premium_{0.0};
    mutable atomic<double> cached_delta_{0.0};
    mutable atomic<double> cached_gamma_{0.0};
    mutable atomic<double> cached_vega_{0.0};
    mutable atomic<double> cached_theta_{0.0};
    mutable atomic<bool> cache_valid_{false};
    
    void compute_greeks() const {
        auto underlying = underlying_.lock();
        if (!underlying || expiry_ <= 0.0) {
            cache_valid_.store(false, memory_order_relaxed);
            return;
        }
        
        double S = underlying->get_market_price();
        double K = strike_;
        double T = expiry_;
        double r = rf_rate_;
        double q = 0.0; // underlying->dividend_yield_; // Commented out as dividend_yield_ is private
        double sigma = (impl_vol_ > 0.0) ? impl_vol_ : underlying->volatility();
        
        if (S <= 0.0 || sigma <= 0.0 || T <= 0.0) {
            cache_valid_.store(false, memory_order_relaxed);
            return;
        }
        
        double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
        double d2 = d1 - sigma * sqrt(T);
        
        double Nd1 = Math::normal_cdf(d1);
        double Nd2 = Math::normal_cdf(d2);
        double Nmd1 = Math::normal_cdf(-d1);
        double Nmd2 = Math::normal_cdf(-d2);
        double pdf_d1 = Math::normal_pdf(d1);
        
        // Premium
        double premium;
        if (is_call_) {
            premium = S * exp(-q * T) * Nd1 - K * exp(-r * T) * Nd2;
        } else {
            premium = K * exp(-r * T) * Nmd2 - S * exp(-q * T) * Nmd1;
        }
        
        // Greeks
        double delta = is_call_ ? exp(-q * T) * Nd1 : exp(-q * T) * (Nd1 - 1.0);
        double gamma = exp(-q * T) * pdf_d1 / (S * sigma * sqrt(T));
        double vega = S * exp(-q * T) * pdf_d1 * sqrt(T) * 0.01; // per 1% vol change
        double theta = (-S * exp(-q * T) * pdf_d1 * sigma / (2.0 * sqrt(T))
                       - r * K * exp(-r * T) * (is_call_ ? Nd2 : Nmd2)
                       + q * S * exp(-q * T) * (is_call_ ? Nd1 : Nmd1)) / 365.0; // daily theta
        
        cached_premium_.store(premium, memory_order_relaxed);
        cached_delta_.store(delta, memory_order_relaxed);
        cached_gamma_.store(gamma, memory_order_relaxed);
        cached_vega_.store(vega, memory_order_relaxed);
        cached_theta_.store(theta, memory_order_relaxed);
        cache_valid_.store(true, memory_order_relaxed);
        last_market_price.store(premium, memory_order_relaxed);
    }
    
public:
    EuropeanOption(string symbol, shared_ptr<Stock> underlying, double strike, double expiry,
                   bool is_call, double impl_vol = 0.0, double rf_rate = Config::RISK_FREE_RATE,
                   double qty = 0.0, double cost = 0.0)
        : Asset(move(symbol), move(symbol), qty, cost), underlying_(underlying),
          strike_(strike), expiry_(expiry), impl_vol_(impl_vol), rf_rate_(rf_rate), is_call_(is_call) {
        compute_greeks(); // Initialize cache
    }
    
    double get_market_price() const override {
        if (!cache_valid_.load(memory_order_acquire)) {
            compute_greeks();
        }
        return cached_premium_.load(memory_order_relaxed);
    }
    
    double volatility() const override {
        if (impl_vol_ > 0.0) return impl_vol_;
        auto underlying = underlying_.lock();
        return underlying ? underlying->volatility() : 0.0;
    }
    
    double delta() const override {
        if (!cache_valid_.load(memory_order_acquire)) compute_greeks();
        return cached_delta_.load(memory_order_relaxed);
    }
    
    double gamma() const override {
        if (!cache_valid_.load(memory_order_acquire)) compute_greeks();
        return cached_gamma_.load(memory_order_relaxed);
    }
    
    double vega() const override {
        if (!cache_valid_.load(memory_order_acquire)) compute_greeks();
        return cached_vega_.load(memory_order_relaxed);
    }
    
    double theta() const override {
        if (!cache_valid_.load(memory_order_acquire)) compute_greeks();
        return cached_theta_.load(memory_order_relaxed);
    }
    
    void set_implied_vol(double vol) {
        unique_lock<shared_mutex> lock(mutex_);
        impl_vol_ = vol;
        cache_valid_.store(false, memory_order_release);
    }
    
    void set_time_to_expiry(double years) {
        unique_lock<shared_mutex> lock(mutex_);
        expiry_ = years;
        cache_valid_.store(false, memory_order_release);
    }
    
    string info() const override {
        auto underlying = underlying_.lock();
        ostringstream oss;
        oss << (is_call_ ? "CALL " : "PUT ") << symbol 
            << " | Underlying: " << (underlying ? underlying->symbol : "N/A")
            << " | Strike: $" << fixed << setprecision(2) << strike_
            << " | Expiry: " << fixed << setprecision(3) << expiry_ << "y"
            << " | Premium: $" << get_market_price()
            << " | Qty: " << quantity
            << " | IV: " << fixed << setprecision(4) << (impl_vol_ > 0.0 ? impl_vol_ : volatility());
        return oss.str();
    }
    
    string risk_report() const override {
        ostringstream oss;
        oss << "Option " << symbol << " | Delta: " << fixed << setprecision(4) << delta()
            << " | Gamma: " << gamma() << " | Vega: $" << vega()
            << " | Theta: $" << theta() << "/day";
        return oss.str();
    }
};

// ==================== Advanced Portfolio with Risk Management ====================
class AdvancedPortfolio {
    mutable shared_mutex mutex_;
    unordered_map<string, shared_ptr<Asset>> assets;
    vector<vector<double>> correlation_matrix;
    MarketDataFeed& market_data;
    OrderManager& order_manager;
    
    // Performance metrics
    atomic<double> total_pnl{0.0};
    atomic<double> daily_pnl{0.0};
    atomic<double> max_drawdown{0.0};
    atomic<double> peak_value{0.0};
    vector<double> historical_pnl;
    
    // Risk limits
    double max_portfolio_value_;
    double max_delta_;
    double max_vega_;
    double max_gamma_;
    
    void enforce_risk_limits() {
        double portfolio_delta = total_delta();
        double portfolio_vega = total_vega();
        double portfolio_gamma = total_gamma();
        
        if (abs(portfolio_delta) > max_delta_) {
            // Auto-hedge delta
            hedge_delta_exposure();
        }
        
        if (abs(portfolio_vega) > max_vega_) {
            // Alert for vega exposure
            cerr << "WARNING: Vega exposure limit exceeded: " << portfolio_vega << endl;
        }
    }
    
    void hedge_delta_exposure() {
        double delta = total_delta();
        // Simple delta hedging - would need access to underlying
        cout << "Delta hedging required. Current delta: " << delta << endl;
    }
    
public:
    AdvancedPortfolio(MarketDataFeed& md, OrderManager& om, 
                     double max_port_val = 10000000.0, double max_delta = Config::MAX_DELTA_EXPOSURE,
                     double max_vega = Config::MAX_VEGA_EXPOSURE, double max_gamma = 100000.0)
        : market_data(md), order_manager(om), max_portfolio_value_(max_port_val),
          max_delta_(max_delta), max_vega_(max_vega), max_gamma_(max_gamma) {}
    
    bool add_asset(shared_ptr<Asset> asset) {
        unique_lock<shared_mutex> lock(mutex_);
        if (assets.size() >= Config::MAX_PORTFOLIO_SIZE) return false;
        
        assets[asset->symbol] = asset;
        update_correlation_matrix();
        return true;
    }
    
    bool remove_asset(const string& symbol) {
        unique_lock<shared_mutex> lock(mutex_);
        bool removed = assets.erase(symbol) > 0;
        if (removed) update_correlation_matrix();
        return removed;
    }
    
    shared_ptr<Asset> get_asset(const string& symbol) const {
        shared_lock<shared_mutex> lock(mutex_);
        auto it = assets.find(symbol);
        return it != assets.end() ? it->second : nullptr;
    }
    
    vector<shared_ptr<Asset>> get_all_assets() const {
        shared_lock<shared_mutex> lock(mutex_);
        vector<shared_ptr<Asset>> result;
        for (const auto& [symbol, asset] : assets) {
            result.push_back(asset);
        }
        return result;
    }
    
    double total_market_value() const {
        shared_lock<shared_mutex> lock(mutex_);
        double total = 0.0;
        for (const auto& [symbol, asset] : assets) {
            total += asset->market_value();
        }
        return total;
    }
    
    double total_delta() const {
        shared_lock<shared_mutex> lock(mutex_);
        double delta = 0.0;
        for (const auto& [symbol, asset] : assets) {
            delta += asset->delta() * asset->quantity;
        }
        return delta;
    }
    
    double total_gamma() const {
        shared_lock<shared_mutex> lock(mutex_);
        double gamma = 0.0;
        for (const auto& [symbol, asset] : assets) {
            gamma += asset->gamma() * asset->quantity;
        }
        return gamma;
    }
    
    double total_vega() const {
        shared_lock<shared_mutex> lock(mutex_);
        double vega = 0.0;
        for (const auto& [symbol, asset] : assets) {
            vega += asset->vega() * asset->quantity;
        }
        return vega;
    }
    
    double total_theta() const {
        shared_lock<shared_mutex> lock(mutex_);
        double theta = 0.0;
        for (const auto& [symbol, asset] : assets) {
            theta += asset->theta() * asset->quantity;
        }
        return theta;
    }
    
    void update_correlation_matrix() {
        size_t n = assets.size();
        correlation_matrix.assign(n, vector<double>(n, 0.0));
        
        // Simple correlation model - in reality, this would use historical data
        for (size_t i = 0; i < n; ++i) {
            correlation_matrix[i][i] = 1.0;
            for (size_t j = i + 1; j < n; ++j) {
                // Basic correlation: same sector = 0.7, different = 0.3
                // This is simplified - real implementation would use proper correlation models
                double corr = 0.5; // Default moderate correlation
                correlation_matrix[i][j] = correlation_matrix[j][i] = corr;
            }
        }
    }
    
    // ==================== Advanced Risk Metrics ====================
    struct RiskMetrics {
        double var_95;
        double var_99;
        double expected_shortfall_95;
        double expected_shortfall_99;
        double stress_loss;
        double marginal_var;
    };
    
    RiskMetrics compute_risk_metrics(int num_simulations = Config::MC_SIMS_ACCURATE) const {
        shared_lock<shared_mutex> lock(mutex_);
        RiskMetrics metrics{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        
        if (assets.empty()) return metrics;
        
        // Build covariance matrix from volatilities and correlations
        vector<double> volatilities;
        vector<double> market_values;
        
        for (const auto& [symbol, asset] : assets) {
            volatilities.push_back(asset->volatility());
            market_values.push_back(asset->market_value());
        }
        
        size_t n = assets.size();
        vector<vector<double>> covariance(n, vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                covariance[i][j] = volatilities[i] * volatilities[j] * correlation_matrix[i][j];
            }
        }
        
        // Monte Carlo simulation for VaR and ES
        vector<double> pnl_samples;
        pnl_samples.reserve(num_simulations);
        
        Math::CholeskyDecomp cholesky(covariance);
        if (!cholesky.is_valid()) {
            cerr << "Cholesky decomposition failed - using diagonal approximation" << endl;
            // Fallback to uncorrelated simulation
            for (int sim = 0; sim < num_simulations; ++sim) {
                double pnl = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    double z = gaussian();
                    pnl += market_values[i] * volatilities[i] * z;
                }
                pnl_samples.push_back(pnl);
            }
        } else {
            // Correlated simulation
            for (int sim = 0; sim < num_simulations; ++sim) {
                vector<double> uncorrelated(n);
                for (size_t i = 0; i < n; ++i) {
                    uncorrelated[i] = gaussian();
                }
                vector<double> correlated = cholesky.generate_correlated_normal(uncorrelated);
                
                double pnl = 0.0;
                for (size_t i = 0; i < n; ++i) {
                    pnl += market_values[i] * volatilities[i] * correlated[i];
                }
                pnl_samples.push_back(pnl);
            }
        }
        
        // Compute VaR and Expected Shortfall
        sort(pnl_samples.begin(), pnl_samples.end());
        
        int idx_95 = max(0, min(num_simulations - 1, int(0.05 * num_simulations)));
        int idx_99 = max(0, min(num_simulations - 1, int(0.01 * num_simulations)));
        
        metrics.var_95 = -pnl_samples[idx_95];
        metrics.var_99 = -pnl_samples[idx_99];
        
        // Expected Shortfall (average of losses beyond VaR)
        double es_95 = 0.0, es_99 = 0.0;
        int count_95 = idx_95, count_99 = idx_99;
        
        if (count_95 > 0) {
            for (int i = 0; i < count_95; ++i) es_95 += pnl_samples[i];
            metrics.expected_shortfall_95 = -es_95 / count_95;
        }
        
        if (count_99 > 0) {
            for (int i = 0; i < count_99; ++i) es_99 += pnl_samples[i];
            metrics.expected_shortfall_99 = -es_99 / count_99;
        }
        
        // Stress test: 5 standard deviation move
        metrics.stress_loss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            metrics.stress_loss += market_values[i] * volatilities[i] * 5.0;
        }
        metrics.stress_loss = abs(metrics.stress_loss);
        
        return metrics;
    }
    
    // ==================== Trading Methods ====================
    bool submit_market_order(const string& symbol, OrderSide side, int quantity) {
        auto asset = get_asset(symbol);
        if (!asset) return false;
        
        double price = asset->get_market_price();
        // In real implementation, you'd get bid/ask from market data
        return order_manager.submit_order(symbol, OrderType::MARKET, side, price, quantity, "PORTFOLIO_MGMT");
    }
    
    bool submit_limit_order(const string& symbol, OrderSide side, double price, int quantity) {
        return order_manager.submit_order(symbol, OrderType::LIMIT, side, price, quantity, "PORTFOLIO_MGMT");
    }
    
    void process_trades() {
        order_manager.process_order_queue([this](const Order& order) -> bool {
            auto asset = get_asset(order.symbol);
            if (asset) {
                asset->on_trade(order.price, order.quantity, order.side);
                enforce_risk_limits();
                return true;
            }
            return false;
        });
    }
    
    // ==================== Reporting ====================
    void print_detailed_report() const {
        shared_lock<shared_mutex> lock(mutex_);
        
        cout << "\n" << string(100, '=') << "\n";
        cout << "ADVANCED PORTFOLIO REPORT\n";
        cout << string(100, '=') << "\n";
        
        // Summary
        cout << "PORTFOLIO SUMMARY:\n";
        cout << "Total Value: $" << fixed << setprecision(2) << total_market_value() << "\n";
        cout << "Total Delta: " << total_delta() << " | Gamma: " << total_gamma() 
             << " | Vega: $" << total_vega() << " | Theta: $" << total_theta() << "/day\n";
        
        // Risk metrics
        auto risk = compute_risk_metrics(Config::MC_SIMS_FAST);
        cout << "\nRISK METRICS:\n";
        cout << "VaR 95%: $" << risk.var_95 << " | VaR 99%: $" << risk.var_99 << "\n";
        cout << "ES 95%: $" << risk.expected_shortfall_95 << " | ES 99%: $" << risk.expected_shortfall_99 << "\n";
        cout << "Stress Loss (5σ): $" << risk.stress_loss << "\n";
        
        // Positions
        cout << "\nDETAILED POSITIONS:\n";
        for (const auto& [symbol, asset] : assets) {
            cout << asset->info() << "\n";
            cout << "  " << asset->risk_report() << "\n";
        }
        
        // Orders
        auto active_orders = order_manager.get_active_orders();
        if (!active_orders.empty()) {
            cout << "\nACTIVE ORDERS:\n";
            for (const auto& order : active_orders) {
                cout << order.symbol << " " << (order.side == OrderSide::BUY ? "BUY" : "SELL") 
                     << " " << order.quantity << " @ $" << order.price << " [" << order.id << "]\n";
            }
        }
        
        cout << string(100, '=') << "\n";
    }
};

// ==================== Real-time Monitoring System ====================
class PortfolioMonitor {
    AdvancedPortfolio& portfolio;
    atomic<bool> running{false};
    thread monitor_thread;
    mutable mutex cout_mutex;
    
    void monitoring_loop() {
        NanoTimer timer;
        int frame_count = 0;
        double total_latency = 0.0;
        
        while (running.load(memory_order_acquire)) {
            double start_time = timer.elapsed();
            
            {
                lock_guard<mutex> lock(cout_mutex);
#if defined(_WIN32)
                system("cls");
#else
                cout << "\033[2J\033[1;1H";
#endif
                
                cout << "\033[36m===== HFT PORTFOLIO MONITOR =====\033[0m\n";
                if (frame_count > 0) {
                    cout << "Frame: " << ++frame_count << " | Latency: " << fixed << setprecision(3) 
                         << (total_latency / frame_count * 1000) << "ms\n\n";
                } else {
                    cout << "Frame: " << ++frame_count << "\n\n";
                }
                
                portfolio.print_detailed_report();
            }
            
            portfolio.process_trades();
            
            double end_time = timer.elapsed();
            total_latency += (end_time - start_time);
            
            this_thread::sleep_for(milliseconds(100)); // 10 Hz update
        }
    }
    
public:
    PortfolioMonitor(AdvancedPortfolio& pf) : portfolio(pf) {}
    ~PortfolioMonitor() { stop(); }
    
    void start() {
        if (running.load()) return;
        running.store(true, memory_order_release);
        monitor_thread = thread(&PortfolioMonitor::monitoring_loop, this);
    }
    
    void stop() {
        running.store(false, memory_order_release);
        if (monitor_thread.joinable()) monitor_thread.join();
    }
};

// ==================== Utility Functions ====================
string now_iso8601() {
    auto now = system_clock::now();
    auto tt = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", localtime(&tt));
    
    string result(buffer);
    result += "." + to_string(ms.count());
    return result;
}

// ==================== Main Application ====================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cout << "Initializing Advanced HFT Portfolio System...\n";
    
    // Create market data feed and order manager
    MarketDataFeed market_data;
    OrderManager order_manager;
    
    // Create portfolio with risk limits
    AdvancedPortfolio portfolio(market_data, order_manager, 5000000.0, 250000.0, 100000.0);
    
    // Create monitor
    PortfolioMonitor monitor(portfolio);
    
    // Add some sample assets
    auto apple = make_shared<Stock>("AAPL", 150.0, 0.25, 0.005);
    auto google = make_shared<Stock>("GOOGL", 2800.0, 0.20, 0.0);
    auto msft = make_shared<Stock>("MSFT", 300.0, 0.22, 0.01);
    
    portfolio.add_asset(apple);
    portfolio.add_asset(google);
    portfolio.add_asset(msft);
    
    // Add some options
    auto aapl_call = make_shared<EuropeanOption>("AAPL_150C", apple, 150.0, 0.1, true, 0.26);
    auto googl_put = make_shared<EuropeanOption>("GOOGL_2800P", google, 2800.0, 0.2, false, 0.22);
    
    portfolio.add_asset(aapl_call);
    portfolio.add_asset(googl_put);
    
    // Start market data feed
    market_data.add_symbol("AAPL", 150.0);
    market_data.add_symbol("GOOGL", 2800.0);
    market_data.add_symbol("MSFT", 300.0);
    market_data.start();
    
    // Start monitoring
    monitor.start();
    
    // Demo trading activity
    this_thread::sleep_for(seconds(2));
    
    // Place some demo orders
    portfolio.submit_market_order("AAPL", OrderSide::BUY, 100);
    portfolio.submit_limit_order("GOOGL", OrderSide::SELL, 2810.0, 10);
    
    // Let it run for a while
    this_thread::sleep_for(seconds(10));
    
    // Clean shutdown
    cout << "\nInitiating shutdown...\n";
    monitor.stop();
    market_data.stop();
    
    // Final report
    portfolio.print_detailed_report();
    
    cout << "HFT Portfolio System shutdown complete.\n";
    return 0;
}
