//+------------------------------------------------------------------+
//|                        StatArb.mqh                                |
//|        Statistical Arbitrage (Cointegration) Module               |
//|        Pairs Trading via ADF Test + Z-Score                       |
//+------------------------------------------------------------------+
#ifndef SA_MQH
#define SA_MQH

#property copyright "StatArb Module"

//+------------------------------------------------------------------+
//| SECTION 1: ENUMS                                                  |
//+------------------------------------------------------------------+

enum ENUM_SA_TRADE_MODE
{
    SA_MODE_AUTO  = 0  /*Auto Trade*/,
    SA_MODE_ALERT = 1  /*Alert Only*/
};

enum ENUM_SA_SYMBOL_SOURCE
{
    SA_SRC_MARKETWATCH = 0  /*Market Watch*/,
    SA_SRC_MANUAL      = 1  /*Manual List*/
};

enum ENUM_SA_TIMEFRAME
{
    SA_TF_M1   = 0   /*M1*/,
    SA_TF_M2   = 1   /*M2*/,
    SA_TF_M3   = 2   /*M3*/,
    SA_TF_M4   = 3   /*M4*/,
    SA_TF_M5   = 4   /*M5*/,
    SA_TF_M6   = 5   /*M6*/,
    SA_TF_M10  = 6   /*M10*/,
    SA_TF_M12  = 7   /*M12*/,
    SA_TF_M15  = 8   /*M15*/,
    SA_TF_M20  = 9   /*M20*/,
    SA_TF_M30  = 10  /*M30*/,
    SA_TF_H1   = 11  /*H1*/,
    SA_TF_H2   = 12  /*H2*/,
    SA_TF_H3   = 13  /*H3*/,
    SA_TF_H4   = 14  /*H4*/,
    SA_TF_H6   = 15  /*H6*/,
    SA_TF_H8   = 16  /*H8*/,
    SA_TF_H12  = 17  /*H12*/,
    SA_TF_D1   = 18  /*D1*/,
    SA_TF_W1   = 19  /*W1*/,
    SA_TF_MN1  = 20  /*MN1*/,
    // Composite: first = entry (shorter), second = hedge/scan (longer)
    SA_TF_M1_M5    = 21  /*M1 + M5*/,
    SA_TF_M1_M15   = 22  /*M1 + M15*/,
    SA_TF_M5_M15   = 23  /*M5 + M15*/,
    SA_TF_M5_M30   = 24  /*M5 + M30*/,
    SA_TF_M5_H1    = 25  /*M5 + H1*/,
    SA_TF_M15_H1   = 26  /*M15 + H1*/,
    SA_TF_M15_H4   = 27  /*M15 + H4*/,
    SA_TF_M30_H1   = 28  /*M30 + H1*/,
    SA_TF_M30_H4   = 29  /*M30 + H4*/,
    SA_TF_H1_H4    = 30  /*H1 + H4*/,
    SA_TF_H1_D1    = 31  /*H1 + D1*/,
    SA_TF_H4_D1    = 32  /*H4 + D1*/
};

enum ENUM_SA_CALC_MODE
{
    SA_CALC_BAR  = 0  /*New Bar*/,
    SA_CALC_TICK = 1  /*Every Tick*/
};

enum ENUM_SA_PRICE
{
    SA_PRICE_CLOSE    = 0  /*Close*/,
    SA_PRICE_OHLC     = 1  /*OHLC avg (O+H+L+C)/4*/,
    SA_PRICE_TYPICAL  = 2  /*Typical (H+L+C)/3*/,
    SA_PRICE_MEDIAN   = 3  /*Median (H+L)/2*/,
    SA_PRICE_WEIGHTED = 4  /*Weighted (H+L+2C)/4*/,
    SA_PRICE_OHLC_RAW = 5  /*OHLC (4 pts/bar)*/
};

enum ENUM_SA_HEDGE_MODE
{
    SA_HEDGE_OLS = 0  /*OLS (standard)*/,
    SA_HEDGE_TLS = 1  /*TLS (Total Least Squares)*/
};

enum ENUM_SA_CORR_MODE
{
    SA_CORR_POSITIVE = 0  /*Positive Only*/,
    SA_CORR_BOTH     = 1  /*Positive & Negative*/
};

enum ENUM_SA_PROFIT_MODE
{
    SA_PROFIT_OFF       = 0  /*Off*/,
    SA_PROFIT_GATE_USD  = 1  /*Profit Gate (Dollar)*/,
    SA_PROFIT_GATE_PCT  = 2  /*Profit Gate (% Balance)*/,
    SA_PROFIT_TPSL_USD  = 3  /*TP & SL (Dollar)*/,
    SA_PROFIT_TPSL_PCT  = 4  /*TP & SL (% Balance)*/
};

enum ENUM_SA_HEDGE_ADAPT
{
    SA_ADAPT_NONE   = 0  /*Static (OLS/TLS)*/,
    SA_ADAPT_KALMAN = 1  /*Kalman Filter*/
};

enum ENUM_SA_BOUNDARY_MODE
{
    SA_BOUND_FIXED = 0  /*Fixed Z-Score*/,
    SA_BOUND_OU    = 1  /*OU Optimal*/
};

enum ENUM_SA_REGIME_MODE
{
    SA_REGIME_OFF   = 0  /*Off*/,
    SA_REGIME_VR    = 1  /*Variance Ratio*/,
    SA_REGIME_HURST = 2  /*Rolling Hurst*/
};

enum ENUM_SA_COINT_TEST
{
    SA_COINT_ADF      = 0  /*Engle-Granger (ADF)*/,
    SA_COINT_JOHANSEN = 1  /*Johansen Trace*/
};

//+------------------------------------------------------------------+
//| SECTION 2: INPUT PARAMETERS                                       |
//+------------------------------------------------------------------+

input group "=============== Statistical Arbitrage ==============="
input bool                    StatArb_Enabled         = true;           // Enable Statistical Arbitrage
input ENUM_SA_TRADE_MODE      StatArb_TradeMode       = SA_MODE_AUTO;   // Trade mode
input ENUM_SA_SYMBOL_SOURCE   StatArb_SymbolSource    = SA_SRC_MARKETWATCH; // Symbol source
input string                  StatArb_SymbolList      = "AUDCAD, AUDCHF, AUDJPY, AUDUSD, CADCHF, CADJPY, EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD, NZDCAD, NZDCHF, NZDJPY, NZDUSD, USDCAD, USDCHF, USDJPY";             // Manual symbol list (comma-separated)
input ENUM_SA_TIMEFRAME       StatArb_Timeframe       = SA_TF_M5;      // Timeframe
input ENUM_SA_CALC_MODE       StatArb_CalcMode        = SA_CALC_BAR;    // Calculation mode
input ENUM_LLM_OHLC_LOOKBACK  StatArb_LookbackMode   = LLM_OHLC_LB_BARS; // Pair scan lookback mode
input int                     StatArb_LookbackLength  = 600;           // Pair scan lookback length
input ENUM_LLM_OHLC_LOOKBACK  StatArb_ZScoreWinMode  = LLM_OHLC_LB_BARS; // Z-Score window mode
input int                     StatArb_ZScoreWinLength = 120;            // Z-Score window length
input ENUM_SA_PRICE           StatArb_PriceType       = SA_PRICE_CLOSE; // Price type
input ENUM_SA_HEDGE_MODE      StatArb_HedgeMode       = SA_HEDGE_OLS;  // Hedge ratio method
input int                     StatArb_ADF_Lags        = 4;             // ADF test lag order
input double                  StatArb_MinCorrelation  = 0.75;          // Min correlation pre-filter
input ENUM_SA_CORR_MODE       StatArb_CorrMode        = SA_CORR_POSITIVE; // Correlation filter mode
input double                  StatArb_ADF_CritLevel   = 0.01;          // ADF significance level
input int                     StatArb_RescanInterval  = 15;            // Rescan interval (minutes, 0=every bar, <0=never)
input int                     StatArb_HealthCheckBars = 50;            // Cointegration health check (bars, 0=off)
input double                  StatArb_EntryZ          = 3.0;           // Entry Z-score threshold
input double                  StatArb_ExitZ           = 0.0;           // Exit Z-score threshold
input double                  StatArb_StopZ           = 5.0;           // Stop Z-score threshold
input ENUM_SA_PROFIT_MODE     StatArb_MinProfitMode    = SA_PROFIT_OFF; // Profit management mode
input double                  StatArb_MinProfitValue   = 0.0;          // TP / gate level (0=off)
input double                  StatArb_MinStopLossValue = 0.0;          // SL level (0=off, TP&SL/Gate modes)
// --- Advanced Features ---
input ENUM_SA_HEDGE_ADAPT     StatArb_HedgeAdapt      = SA_ADAPT_NONE; // Adaptive hedge ratio
input double                  StatArb_KalmanQ          = 1e-5;         // Kalman process noise (Q diagonal)
input double                  StatArb_KalmanR          = 1e-2;         // Kalman observation noise (R)
input ENUM_SA_BOUNDARY_MODE   StatArb_BoundaryMode     = SA_BOUND_FIXED; // Boundary computation mode
input double                  StatArb_OUTransCost      = 0.0001;       // OU transaction cost (spread units)
input double                  StatArb_OUStopMultiple   = 1.75;         // OU stop = entry * this multiplier
input double                  StatArb_TimeExitHL       = 0.0;          // Time exit: N * half_life bars (0=off)
input bool                    StatArb_UseHurst         = false;        // Enable Hurst exponent filter
input double                  StatArb_MaxHurst         = 0.60;         // Max Hurst (H<0.5 = mean-reverting)
input ENUM_SA_REGIME_MODE     StatArb_RegimeMode       = SA_REGIME_OFF; // Regime detection mode
input int                     StatArb_RegimeWindow     = 50;           // Regime detection lookback
input double                  StatArb_RegimeThreshold  = 1.0;          // VR threshold (>1=trending) or Hurst (>0.5)
input ENUM_SA_COINT_TEST      StatArb_CointTest        = SA_COINT_ADF; // Cointegration test method
input int                     StatArb_JohansenLags     = 1;            // Johansen VAR lag order
input int                     StatArb_MaxPairs         = 1;            // Max simultaneous pairs (1=single)
input bool                    StatArb_NoOverlap        = true;         // Prevent symbol overlap between pairs
// --- Robustness Filters (from MQL5 article series research) ---
input bool                    StatArb_UseKPSS          = false;        // KPSS stationarity confirmation (dual-test)
input double                  StatArb_KPSS_Level       = 0.05;         // KPSS significance level (reject stationarity if p<this)
input bool                    StatArb_UseOOS           = false;        // In-sample/Out-of-sample validation
input double                  StatArb_OOS_Ratio        = 0.70;         // IS portion (0.70 = 70% IS, 30% OOS)
input double                  StatArb_OOS_MaxPValue    = 0.05;         // Max OOS ADF p-value to accept
input bool                    StatArb_UseDriftDetect   = false;        // Hedge ratio drift detection (RWEC)
input double                  StatArb_DriftMaxAngle    = 30.0;         // Max drift angle (degrees, 0-90)
input bool                    StatArb_UseCUSUM         = false;        // CUSUM structural break detection

input group "=============== StatArb Sizing & Display ==============="
input double                  StatArb_SpreadLimit     = 1.0;           // ADR-normalized max spread
input double                  StatArb_LotSize         = 0.01;          // Lot size per leg
input int                     StatArb_MagicOffset     = 1000;          // Magic number offset
input bool                    StatArb_ShowPanel       = true;          // Show panel
input int                     StatArb_PanelTopN       = 5;             // Top N pairs in panel
input int                     StatArb_PanelX          = 10;            // Panel X position
input int                     StatArb_PanelY          = 80;           // Panel Y position

//+------------------------------------------------------------------+
//| SECTION 3: CONSTANTS                                              |
//+------------------------------------------------------------------+

#define SA_MAX_CANDIDATES    200
#define SA_MAX_PRICE_CACHE   200
#define SA_PANEL_PREFIX      "SA_"

//+------------------------------------------------------------------+
//| SECTION 4: DATA STRUCTURES                                        |
//+------------------------------------------------------------------+

struct SA_PairResult
{
    string  symbolY;
    string  symbolX;
    double  correlation;
    double  hedge_ratio;
    double  intercept;
    double  adf_statistic;
    double  adf_pvalue;
    bool    is_cointegrated;
    double  half_life;
    double  spread_std;
    double  current_zscore;
    double  score;
    // Advanced feature fields
    double  hurst;
    double  ou_theta;
    double  ou_mu;
    double  ou_sigma;
    double  johansen_trace;
    double  johansen_beta;
    double  oos_pvalue;
    double  kpss_stat;

    void CopyFrom(const SA_PairResult &src)
    {
        symbolY = src.symbolY;
        symbolX = src.symbolX;
        correlation = src.correlation;
        hedge_ratio = src.hedge_ratio;
        intercept = src.intercept;
        adf_statistic = src.adf_statistic;
        adf_pvalue = src.adf_pvalue;
        is_cointegrated = src.is_cointegrated;
        half_life = src.half_life;
        spread_std = src.spread_std;
        current_zscore = src.current_zscore;
        score = src.score;
        hurst = src.hurst;
        ou_theta = src.ou_theta;
        ou_mu = src.ou_mu;
        ou_sigma = src.ou_sigma;
        johansen_trace = src.johansen_trace;
        johansen_beta = src.johansen_beta;
        oos_pvalue = src.oos_pvalue;
        kpss_stat = src.kpss_stat;
    }
};

//+------------------------------------------------------------------+
//| CUSUM (Cumulative Sum) structural break detection                  |
//| Tracks cumulative forecast errors from a simple AR(1) model.      |
//| When |CUSUM| exceeds critical limit, a structural break occurred.  |
//| Critical limit: 0.948 * (sqrt(n) + 2*n/sqrt(n)) at 95% conf.    |
//| Lightweight: only needs current spread and running state.          |
//| Defined here (before SA_ActivePair) because it's embedded as a    |
//| struct member. Functions also here to avoid forward-ref issues.    |
//+------------------------------------------------------------------+
struct SA_CUSUMState
{
    bool   initialized;
    double ar_beta;        // AR(1) slope
    double ar_alpha;       // AR(1) intercept
    double cumsum;         // Running CUSUM
    double prev_spread;    // Previous spread for prediction
    int    n_obs;          // Observations since initialization
    double sum_sq_resid;   // Running SSR for normalization

    void Reset()
    {
        initialized = false;
        ar_beta = 0; ar_alpha = 0;
        cumsum = 0; prev_spread = 0;
        n_obs = 0; sum_sq_resid = 0;
    }
};

void SA_CUSUM_Init(SA_CUSUMState &state, const double &spread[], int n)
{
    state.Reset();
    if(n < 30) return;

    // Fit AR(1): spread[t] = alpha + beta * spread[t-1]
    int m = n - 1;
    double sx = 0, sy = 0, sxy = 0, sxx = 0;
    for(int i = 1; i < n; i++)
    {
        double x = spread[i - 1];
        double y = spread[i];
        sx += x; sy += y; sxy += x * y; sxx += x * x;
    }
    double denom = (double)m * sxx - sx * sx;
    if(MathAbs(denom) < 1e-15) return;

    state.ar_beta = ((double)m * sxy - sx * sy) / denom;
    state.ar_alpha = (sy - state.ar_beta * sx) / (double)m;

    // Compute baseline residual variance
    double ssr = 0;
    for(int i = 1; i < n; i++)
    {
        double pred = state.ar_alpha + state.ar_beta * spread[i - 1];
        double resid = spread[i] - pred;
        ssr += resid * resid;
    }
    state.sum_sq_resid = ssr;
    state.n_obs = m;
    state.prev_spread = spread[n - 1];
    state.cumsum = 0;
    state.initialized = true;
}

bool SA_CUSUM_Update(SA_CUSUMState &state, double new_spread)
{
    if(!state.initialized) return false;

    double pred = state.ar_alpha + state.ar_beta * state.prev_spread;
    double resid = new_spread - pred;

    double sigma = (state.n_obs > 1) ?
        MathSqrt(state.sum_sq_resid / (double)state.n_obs) : 1.0;
    if(sigma < 1e-12) sigma = 1e-12;

    state.cumsum += resid / sigma;
    state.n_obs++;
    state.sum_sq_resid += resid * resid;
    state.prev_spread = new_spread;

    double sqrt_n = MathSqrt((double)state.n_obs);
    double crit = 0.948 * (sqrt_n + 2.0 * (double)state.n_obs / sqrt_n);

    return (MathAbs(state.cumsum) > crit);
}

struct SA_ActivePair
{
    bool     active;
    string   symbolY;
    string   symbolX;
    double   hedge_ratio;
    double   intercept;

    // Rolling Z-score ring buffer
    double   spread_ring[];
    int      ring_size;
    int      ring_head;
    int      ring_count;
    int      ring_updates;     // total pushes (for periodic recompute)
    double   rolling_sum;
    double   rolling_sum_sq;
    double   current_zscore;
    double   current_spread;

    // Position tracking
    ulong    ticketY;
    ulong    ticketX;
    int      position_dir;     // +1 = long Y/short X, -1 = short Y/long X, 0 = flat
    datetime entry_time;
    double   entry_zscore;
    datetime last_close_time;  // Fix 33: bar time of last close, for re-entry cooldown

    // Scan metadata
    datetime last_scan_time;
    double   adf_statistic;
    double   adf_pvalue;
    double   correlation;
    double   half_life;

    // Cointegration health tracking
    int      bars_since_health;   // counter for periodic ADF health check
    double   last_health_pvalue;  // Fix 64: actual p-value from most recent health check (always written)

    // Advanced feature state
    // Kalman filter
    double   kalman_beta;
    double   kalman_alpha;
    double   kalman_P00;
    double   kalman_P01;
    double   kalman_P10;
    double   kalman_P11;
    bool     kalman_initialized;
    // OU parameters & boundaries
    double   ou_theta;
    double   ou_mu;
    double   ou_sigma;
    double   ou_entry_long;
    double   ou_entry_short;
    double   ou_exit_long;
    double   ou_exit_short;
    double   ou_stop_long;
    double   ou_stop_short;
    // Time exit
    int      bars_in_trade;
    // Hurst
    double   hurst;
    // Regime
    double   regime_score;
    bool     regime_trending;
    // Hedge ratio drift detection
    double   initial_beta;        // Beta at activation, for RWEC cosine drift
    // CUSUM structural break detection
    SA_CUSUMState cusum_state;
    // Multi-pair: per-pair new-bar tracking
    datetime last_bar_time;

    void Reset()
    {
        active = false;
        symbolY = ""; symbolX = "";
        hedge_ratio = 0; intercept = 0;
        ring_size = 0; ring_head = 0; ring_count = 0; ring_updates = 0;
        rolling_sum = 0; rolling_sum_sq = 0;
        current_zscore = 0; current_spread = 0;
        ticketY = 0; ticketX = 0;
        position_dir = 0; entry_time = 0; entry_zscore = 0; last_close_time = 0;
        last_scan_time = 0;
        adf_statistic = 0; adf_pvalue = 1.0; correlation = 0; half_life = 0;
        bars_since_health = 0; last_health_pvalue = 1.0;
        // Advanced
        kalman_beta = 0; kalman_alpha = 0;
        kalman_P00 = 1.0; kalman_P01 = 0; kalman_P10 = 0; kalman_P11 = 1.0;
        kalman_initialized = false;
        ou_theta = 0; ou_mu = 0; ou_sigma = 0;
        ou_entry_long = 0; ou_entry_short = 0;
        ou_exit_long = 0; ou_exit_short = 0;
        ou_stop_long = 0; ou_stop_short = 0;
        bars_in_trade = 0;
        hurst = 0.5;
        regime_score = 0; regime_trending = false;
        initial_beta = 0;
        cusum_state.Reset();
        last_bar_time = 0;
        ArrayFree(spread_ring);
    }
};

struct SA_TopPairEntry
{
    string   symbolY;
    string   symbolX;
    double   correlation;
    double   adf_stat;
    double   adf_pvalue;
    double   hedge_ratio;
    double   half_life;
    double   current_zscore;
    double   score;
    int      position_dir;
    bool     is_active;
    double   hurst;
};

struct SA_PriceCache
{
    string          symbol;
    ENUM_TIMEFRAMES timeframe;
    datetime        last_bar_time;
    double          prices[];
    int             count;
    bool            valid;

    void Reset()
    {
        symbol = "";
        timeframe = PERIOD_CURRENT;
        last_bar_time = 0;
        ArrayFree(prices);
        count = 0;
        valid = false;
    }
};

#include "StatArb_Advanced.mqh"
#include "CMLEngine.mqh"
#include "StatArb_ML.mqh"

//+------------------------------------------------------------------+
//| SECTION 5: GLOBAL STATE                                           |
//+------------------------------------------------------------------+

SA_ActivePair    g_sa_pairs[];              // Array of pairs (size = StatArb_MaxPairs)
int              g_sa_current_idx = 0;      // Index of pair currently being processed
#define          g_sa_pair g_sa_pairs[g_sa_current_idx]
SA_TopPairEntry  g_sa_top_pairs[];
int              g_sa_top_count = 0;
SA_PriceCache    g_sa_price_cache[];
int              g_sa_price_cache_count = 0;
CTrade           sa_trade;
datetime         g_sa_last_scan_bar_time = 0;  // Scan cooldown (separate from trade new-bar)
bool             g_sa_initialized = false;
int              g_sa_deinit_reason = -1; // Fix 30: communicate deinit reason to Init

string           g_sa_candidates[];
int              g_sa_candidate_count = 0;

// Alert cooldown
datetime         g_sa_last_alert_time = 0;

// Scan stats (for panel)
int              g_sa_scan_total = 0;
int              g_sa_scan_corr_passed = 0;
int              g_sa_scan_cointegrated = 0;
int              g_sa_alerts_sent = 0;
int              g_sa_active_trades = 0;

// Clamped PanelTopN (inputs cannot be modified, so use runtime variable)
int              g_sa_panel_topn = 5;

//+------------------------------------------------------------------+
//| SECTION 6: TIMEFRAME RESOLUTION                                   |
//+------------------------------------------------------------------+

ENUM_TIMEFRAMES SA_EnumToTF(int idx)
{
    switch(idx)
    {
        case 0:  return PERIOD_M1;
        case 1:  return PERIOD_M2;
        case 2:  return PERIOD_M3;
        case 3:  return PERIOD_M4;
        case 4:  return PERIOD_M5;
        case 5:  return PERIOD_M6;
        case 6:  return PERIOD_M10;
        case 7:  return PERIOD_M12;
        case 8:  return PERIOD_M15;
        case 9:  return PERIOD_M20;
        case 10: return PERIOD_M30;
        case 11: return PERIOD_H1;
        case 12: return PERIOD_H2;
        case 13: return PERIOD_H3;
        case 14: return PERIOD_H4;
        case 15: return PERIOD_H6;
        case 16: return PERIOD_H8;
        case 17: return PERIOD_H12;
        case 18: return PERIOD_D1;
        case 19: return PERIOD_W1;
        case 20: return PERIOD_MN1;
        default: return PERIOD_M5;
    }
}

void SA_ResolveTimeframes(ENUM_TIMEFRAMES &entry_tf, ENUM_TIMEFRAMES &hedge_tf)
{
    int val = (int)StatArb_Timeframe;
    if(val <= 20)
    {
        // Single timeframe: both same
        entry_tf = SA_EnumToTF(val);
        hedge_tf = entry_tf;
    }
    else
    {
        // Composite: first = entry (shorter), second = hedge/scan (longer)
        switch(val)
        {
            case 21: entry_tf = PERIOD_M1;  hedge_tf = PERIOD_M5;  break;
            case 22: entry_tf = PERIOD_M1;  hedge_tf = PERIOD_M15; break;
            case 23: entry_tf = PERIOD_M5;  hedge_tf = PERIOD_M15; break;
            case 24: entry_tf = PERIOD_M5;  hedge_tf = PERIOD_M30; break;
            case 25: entry_tf = PERIOD_M5;  hedge_tf = PERIOD_H1;  break;
            case 26: entry_tf = PERIOD_M15; hedge_tf = PERIOD_H1;  break;
            case 27: entry_tf = PERIOD_M15; hedge_tf = PERIOD_H4;  break;
            case 28: entry_tf = PERIOD_M30; hedge_tf = PERIOD_H1;  break;
            case 29: entry_tf = PERIOD_M30; hedge_tf = PERIOD_H4;  break;
            case 30: entry_tf = PERIOD_H1;  hedge_tf = PERIOD_H4;  break;
            case 31: entry_tf = PERIOD_H1;  hedge_tf = PERIOD_D1;  break;
            case 32: entry_tf = PERIOD_H4;  hedge_tf = PERIOD_D1;  break;
            default: entry_tf = PERIOD_M5;  hedge_tf = PERIOD_M5;  break;
        }
    }
}

//+------------------------------------------------------------------+
//| Resolve lookback to bar count (reuses LLM_Bridge pattern)         |
//+------------------------------------------------------------------+

int SA_ResolveLookback(string symbol, ENUM_TIMEFRAMES tf, ENUM_LLM_OHLC_LOOKBACK mode, int length)
{
    int bars = 0;
    if(mode == LLM_OHLC_LB_BARS)
        bars = length;
    else if(mode == LLM_OHLC_LB_HOURS)
    {
        int tf_seconds = PeriodSeconds(tf);
        if(tf_seconds > 0)
            // Fix 45: Cast to long before multiply to prevent int overflow for large hour values
            bars = (int)(((long)length * 3600) / tf_seconds);
        else
            bars = length;
    }
    else
    {
        // Midnight modes: enum 2..6 → day shift 0..4
        int day_shift = (int)mode - 2;
        datetime midnight_time = iTime(symbol, PERIOD_D1, day_shift);
        if(midnight_time > 0)
        {
            int shift = iBarShift(symbol, tf, midnight_time);
            if(shift >= 0) bars = shift + 1;
            else bars = 50; // fallback
        }
        else
            bars = 50; // fallback
    }
    if(bars < 10) bars = 10;
    if(bars > 5000) bars = 5000;
    return bars;
}

//+------------------------------------------------------------------+
//| SECTION 7: MATHEMATICAL CORE                                      |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Pearson Correlation Coefficient                                    |
//+------------------------------------------------------------------+

double SA_PearsonCorrelation(const double &y[], const double &x[], int n)
{
    if(n < 3) return 0.0;

    double sum_x = 0, sum_y = 0;
    for(int i = 0; i < n; i++)
    {
        sum_x += x[i];
        sum_y += y[i];
    }
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;

    double cov = 0, var_x = 0, var_y = 0;
    for(int i = 0; i < n; i++)
    {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        cov   += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    double denom = MathSqrt(var_x * var_y);
    if(denom < 1e-15) return 0.0;
    return cov / denom;
}

//+------------------------------------------------------------------+
//| OLS Regression: Y = alpha + beta * X                              |
//+------------------------------------------------------------------+

void SA_OLS(const double &y[], const double &x[], int n,
            double &beta, double &alpha, double &residuals[])
{
    beta = 0; alpha = 0;
    ArrayResize(residuals, n);

    if(n < 3)
    {
        ArrayInitialize(residuals, 0);
        return;
    }

    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for(int i = 0; i < n; i++)
    {
        sum_x  += x[i];
        sum_y  += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }

    double denom = (double)n * sum_xx - sum_x * sum_x;
    if(MathAbs(denom) < 1e-15)
    {
        ArrayInitialize(residuals, 0);
        return;
    }

    beta  = ((double)n * sum_xy - sum_x * sum_y) / denom;
    alpha = (sum_y - beta * sum_x) / (double)n;

    for(int i = 0; i < n; i++)
        residuals[i] = y[i] - alpha - beta * x[i];
}

//+------------------------------------------------------------------+
//| TLS Regression (Total Least Squares): minimizes orthogonal error  |
//| Uses closed-form 2x2 eigendecomposition                          |
//+------------------------------------------------------------------+

void SA_TLS(const double &y[], const double &x[], int n,
            double &beta, double &alpha, double &residuals[])
{
    beta = 0; alpha = 0;
    ArrayResize(residuals, n);

    if(n < 3)
    {
        ArrayInitialize(residuals, 0);
        return;
    }

    // Center the data
    double mean_x = 0, mean_y = 0;
    for(int i = 0; i < n; i++)
    {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= n;
    mean_y /= n;

    // Build 2x2 covariance matrix [sxx, sxy; sxy, syy]
    double sxx = 0, sxy = 0, syy = 0;
    for(int i = 0; i < n; i++)
    {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    sxx /= n;
    sxy /= n;
    syy /= n;

    // Eigenvalues of 2x2 via quadratic formula
    // disc = (sxx-syy)^2 + 4*sxy^2 — always non-negative by construction
    // (algebraically equivalent to trace^2 - 4*det, but avoids FP cancellation)
    double trace = sxx + syy;
    double diff  = sxx - syy;
    double disc  = diff * diff + 4.0 * sxy * sxy;
    double sqrt_disc = MathSqrt(disc);

    // Smallest eigenvalue
    double lambda_min = (trace - sqrt_disc) / 2.0;

    // Eigenvector for smallest eigenvalue: (sxx - lambda_min) * v1 + sxy * v2 = 0
    // => v2/v1 = -(sxx - lambda_min) / sxy
    double a = sxx - lambda_min;
    double b = sxy;

    if(MathAbs(a) > 1e-15)
    {
        // Eigenvector: (sxx-lambda)*v1 + sxy*v2 = 0 => beta = -v1/v2 = sxy/(sxx-lambda)
        beta = b / a;
    }
    else if(MathAbs(b) > 1e-15)
    {
        // sxx == lambda_min, degenerate (near-vertical line)
        beta = 1e10 * ((b > 0) ? 1.0 : -1.0);
    }
    else
    {
        // Both near zero, fall back to OLS
        SA_OLS(y, x, n, beta, alpha, residuals);
        return;
    }

    alpha = mean_y - beta * mean_x;

    for(int i = 0; i < n; i++)
        residuals[i] = y[i] - alpha - beta * x[i];
}

//+------------------------------------------------------------------+
//| Gauss Elimination with Partial Pivoting (for ADF normal eqns)     |
//| Solves A*x = b where A is k×k                                    |
//+------------------------------------------------------------------+

bool SA_GaussSolve(double &A[], int k, double &b[], double &x_out[])
{
    // A is stored as flat array [k*k], row-major
    ArrayResize(x_out, k);
    ArrayInitialize(x_out, 0);

    // Make working copies
    double M[];
    ArrayResize(M, k * k);
    ArrayCopy(M, A);
    double rhs[];
    ArrayResize(rhs, k);
    ArrayCopy(rhs, b);

    // Forward elimination with partial pivoting
    for(int col = 0; col < k; col++)
    {
        // Find pivot
        int pivot_row = col;
        double max_val = MathAbs(M[col * k + col]);
        for(int row = col + 1; row < k; row++)
        {
            double v = MathAbs(M[row * k + col]);
            if(v > max_val)
            {
                max_val = v;
                pivot_row = row;
            }
        }

        if(max_val < 1e-14) return false; // Singular

        // Swap rows
        if(pivot_row != col)
        {
            for(int j = 0; j < k; j++)
            {
                double tmp = M[col * k + j];
                M[col * k + j] = M[pivot_row * k + j];
                M[pivot_row * k + j] = tmp;
            }
            double tmp = rhs[col];
            rhs[col] = rhs[pivot_row];
            rhs[pivot_row] = tmp;
        }

        // Eliminate below
        double pivot_val = M[col * k + col];
        for(int row = col + 1; row < k; row++)
        {
            double factor = M[row * k + col] / pivot_val;
            for(int j = col; j < k; j++)
                M[row * k + j] -= factor * M[col * k + j];
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    for(int i = k - 1; i >= 0; i--)
    {
        double s = rhs[i];
        for(int j = i + 1; j < k; j++)
            s -= M[i * k + j] * x_out[j];
        if(MathAbs(M[i * k + i]) < 1e-14) return false;
        x_out[i] = s / M[i * k + i];
    }

    return true;
}

//+------------------------------------------------------------------+
//| MacKinnon ADF Critical Values (constant, no trend)                |
//| Returns approximate p-value by interpolation                      |
//+------------------------------------------------------------------+

double SA_ADF_PValue(double t_stat, int n)
{
    // Critical values: rows = sample sizes, cols = significance (1%, 5%, 10%)
    // Source: MacKinnon (1994) for ADF with constant, no trend
    double cv_n[]    = {25, 50, 100, 250, 500, 10000};
    double cv_01[]   = {-3.75, -3.58, -3.51, -3.46, -3.44, -3.43};
    double cv_05[]   = {-3.00, -2.93, -2.89, -2.88, -2.87, -2.86};
    double cv_10[]   = {-2.63, -2.60, -2.58, -2.57, -2.57, -2.57};

    // Interpolate critical values for our sample size
    double c01 = cv_01[0], c05 = cv_05[0], c10 = cv_10[0];

    if(n <= (int)cv_n[0])
    {
        c01 = cv_01[0]; c05 = cv_05[0]; c10 = cv_10[0];
    }
    else if(n >= (int)cv_n[5])
    {
        c01 = cv_01[5]; c05 = cv_05[5]; c10 = cv_10[5];
    }
    else
    {
        // Find bracket
        int idx = 0;
        for(int i = 0; i < 5; i++)
        {
            if(n >= (int)cv_n[i] && n < (int)cv_n[i + 1])
            {
                idx = i;
                break;
            }
        }
        double frac = ((double)n - cv_n[idx]) / (cv_n[idx + 1] - cv_n[idx]);
        c01 = cv_01[idx] + frac * (cv_01[idx + 1] - cv_01[idx]);
        c05 = cv_05[idx] + frac * (cv_05[idx + 1] - cv_05[idx]);
        c10 = cv_10[idx] + frac * (cv_10[idx + 1] - cv_10[idx]);
    }

    // Map t-stat to approximate p-value (continuous extrapolation below c01)
    if(t_stat <= c01)      return MathMax(0.001, 0.01 + (t_stat - c01) / (c05 - c01) * 0.04); // <=1%
    else if(t_stat <= c05) return 0.01 + (t_stat - c01) / (c05 - c01) * 0.04; // 1-5%
    else if(t_stat <= c10) return 0.05 + (t_stat - c05) / (c10 - c05) * 0.05; // 5-10%
    // Fix 31: Slope 0.15 was too shallow — a t-stat of -1.0 (clearly non-stationary)
    // got p=0.34 instead of ~0.60. Slope 0.30 better matches the ADF distribution
    // tail and avoids inflating false-positive cointegration findings.
    else                   return MathMin(1.0, 0.10 + (t_stat - c10) * 0.30); // >10% (clamped)
}

//+------------------------------------------------------------------+
//| Augmented Dickey-Fuller Test                                       |
//| Returns t-statistic, fills pvalue                                 |
//+------------------------------------------------------------------+

double SA_ADF_Test(const double &spread[], int n, int lags, double &pvalue)
{
    pvalue = 1.0;
    if(n < lags + 10) return 0.0; // Not enough data

    // Step 1: First differences
    int nd = n - 1;
    double delta[];
    ArrayResize(delta, nd);
    for(int i = 0; i < nd; i++)
        delta[i] = spread[i + 1] - spread[i];

    // Step 2: Build regression
    // delta_s[t] = c + gamma * spread[t-1] + sum(phi_j * delta_s[t-j]) + eps
    // Effective sample starts at index = lags (to have lags lagged differences)
    int T = nd - lags; // Number of usable observations
    if(T < 10) return 0.0;

    int K = lags + 2; // Number of regressors: constant + level + lags

    // Guard: need T > K for degrees of freedom in sigma^2 = SSE/(T-K)
    if(T <= K) { pvalue = 1.0; return 0.0; }

    // Build X'X and X'Y using normal equations directly (avoids large matrix allocation)
    double XtX[];
    ArrayResize(XtX, K * K);
    ArrayInitialize(XtX, 0);
    double XtY[];
    ArrayResize(XtY, K);
    ArrayInitialize(XtY, 0);

    double row[];
    ArrayResize(row, K);

    for(int t = lags; t < nd; t++)
    {
        // Build regressor row for observation t
        row[0] = 1.0;                  // constant
        row[1] = spread[t];            // lagged level
        for(int j = 0; j < lags; j++)
            row[2 + j] = delta[t - 1 - j]; // lagged differences

        double y_val = delta[t]; // dependent variable

        // Accumulate X'X and X'Y
        for(int r = 0; r < K; r++)
        {
            XtY[r] += row[r] * y_val;
            for(int c = 0; c < K; c++)
                XtX[r * K + c] += row[r] * row[c];
        }
    }

    // Step 3: Solve normal equations X'X * beta_hat = X'Y
    double beta_hat[];
    if(!SA_GaussSolve(XtX, K, XtY, beta_hat))
    {
        pvalue = 1.0;
        return 0.0; // Singular matrix
    }

    double gamma = beta_hat[1]; // Coefficient on lagged level

    // Step 4: Compute residuals and sigma^2
    double sse = 0;
    for(int t = lags; t < nd; t++)
    {
        row[0] = 1.0;
        row[1] = spread[t];
        for(int j = 0; j < lags; j++)
            row[2 + j] = delta[t - 1 - j];

        double y_hat = 0;
        for(int j = 0; j < K; j++)
            y_hat += beta_hat[j] * row[j];

        double resid = delta[t] - y_hat;
        sse += resid * resid;
    }

    double sigma2 = sse / (double)(T - K);
    if(sigma2 < 1e-20)
    {
        pvalue = 0.001;
        return -10.0; // Perfect fit, definitely stationary
    }

    // Step 5: Compute SE(gamma) from inv(X'X)
    // We need element [1][1] of inv(X'X) * sigma^2
    // Solve X'X * e1 = [0,1,0,...,0] to get column 1 of inv(X'X)
    double e1[];
    ArrayResize(e1, K);
    ArrayInitialize(e1, 0);
    e1[1] = 1.0;

    double inv_col[];
    if(!SA_GaussSolve(XtX, K, e1, inv_col))
    {
        pvalue = 1.0;
        return 0.0;
    }

    double var_gamma = sigma2 * inv_col[1];
    if(var_gamma <= 0)
    {
        pvalue = 1.0;
        return 0.0;
    }

    double se_gamma = MathSqrt(var_gamma);
    double t_stat = gamma / se_gamma;

    // Step 6: P-value from MacKinnon tables
    pvalue = SA_ADF_PValue(t_stat, T);

    return t_stat;
}

//+------------------------------------------------------------------+
//| Rolling Z-Score: incremental ring buffer                          |
//+------------------------------------------------------------------+

double SA_UpdateZScore(double new_spread)
{
    if(g_sa_pair.ring_size <= 0) return 0.0;

    // If buffer is full, remove oldest
    if(g_sa_pair.ring_count >= g_sa_pair.ring_size)
    {
        double old_val = g_sa_pair.spread_ring[g_sa_pair.ring_head];
        g_sa_pair.rolling_sum -= old_val;
        g_sa_pair.rolling_sum_sq -= old_val * old_val;
    }

    // Add new value
    g_sa_pair.spread_ring[g_sa_pair.ring_head] = new_spread;
    g_sa_pair.rolling_sum += new_spread;
    g_sa_pair.rolling_sum_sq += new_spread * new_spread;

    // Advance head
    g_sa_pair.ring_head = (g_sa_pair.ring_head + 1) % g_sa_pair.ring_size;
    if(g_sa_pair.ring_count < g_sa_pair.ring_size)
        g_sa_pair.ring_count++;

    g_sa_pair.current_spread = new_spread;

    // Periodic recompute from scratch every 500 updates to prevent floating-point drift
    g_sa_pair.ring_updates++;
    if(g_sa_pair.ring_updates % 500 == 0)
    {
        g_sa_pair.rolling_sum = 0;
        g_sa_pair.rolling_sum_sq = 0;
        int cnt = (int)MathMin(g_sa_pair.ring_count, g_sa_pair.ring_size);
        for(int i = 0; i < cnt; i++)
        {
            g_sa_pair.rolling_sum += g_sa_pair.spread_ring[i];
            g_sa_pair.rolling_sum_sq += g_sa_pair.spread_ring[i] * g_sa_pair.spread_ring[i];
        }
    }

    if(g_sa_pair.ring_count < 2)
    {
        g_sa_pair.current_zscore = 0;
        return 0.0;
    }

    double cnt = (double)g_sa_pair.ring_count;
    double mean = g_sa_pair.rolling_sum / cnt;
    // Sample variance (n-1) instead of population variance (n)
    double variance = (g_sa_pair.rolling_sum_sq - cnt * mean * mean) / (cnt - 1.0);
    // Fix 28a: Negative variance from FP drift should trigger immediate recompute
    // from the ring buffer, not a silent clamp to 0. Clamping to 0 produces Z=0,
    // which can satisfy the exit condition (ExitZ=0) and close trades prematurely.
    if(variance < 0)
    {
        g_sa_pair.rolling_sum = 0;
        g_sa_pair.rolling_sum_sq = 0;
        int ring_cnt = (int)MathMin(g_sa_pair.ring_count, g_sa_pair.ring_size);
        for(int ii = 0; ii < ring_cnt; ii++)
        {
            g_sa_pair.rolling_sum += g_sa_pair.spread_ring[ii];
            g_sa_pair.rolling_sum_sq += g_sa_pair.spread_ring[ii] * g_sa_pair.spread_ring[ii];
        }
        mean = g_sa_pair.rolling_sum / cnt;
        variance = (g_sa_pair.rolling_sum_sq - cnt * mean * mean) / (cnt - 1.0);
        if(variance < 0) variance = 0; // True zero after recompute — safe to clamp
        g_sa_pair.ring_updates = 0;
    }
    double std = MathSqrt(variance);

    if(std < 1e-12)
    {
        g_sa_pair.current_zscore = 0;
        return 0.0;
    }

    double z = (new_spread - mean) / std;
    g_sa_pair.current_zscore = z;
    return z;
}

//+------------------------------------------------------------------+
//| Half-Life of Mean Reversion via AR(1)                             |
//+------------------------------------------------------------------+

double SA_HalfLife(const double &spread[], int n)
{
    if(n < 5) return -1;

    // Regress delta_s on s_lag WITH intercept: delta_s[t] = c + gamma * s[t-1] + eps
    // Omitting intercept biases gamma when spread has non-zero mean
    int m = n - 1;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for(int i = 1; i < n; i++)
    {
        double s_lag = spread[i - 1];
        double ds    = spread[i] - spread[i - 1];
        sum_x  += s_lag;
        sum_y  += ds;
        sum_xy += ds * s_lag;
        sum_xx += s_lag * s_lag;
    }

    double denom = (double)m * sum_xx - sum_x * sum_x;
    if(MathAbs(denom) < 1e-15) return -1;

    double gamma = ((double)m * sum_xy - sum_x * sum_y) / denom;

    // Mean reversion requires gamma < 0
    if(gamma >= 0) return -1;

    double rho = 1.0 + gamma;
    if(rho <= 0 || rho >= 1.0) return -1;

    return -MathLog(2.0) / MathLog(rho);
}

//+------------------------------------------------------------------+
//| Cointegration Health Check: lightweight ADF on current pair only   |
//| Fetches historical prices directly (self-contained, no ring buffer)|
//| Returns true if still healthy, false if degraded (triggers rescan) |
//+------------------------------------------------------------------+

bool SA_HealthCheck()
{
    if(StatArb_HealthCheckBars <= 0) return true; // Disabled
    if(g_sa_pair.bars_since_health < StatArb_HealthCheckBars) return true; // Not time yet

    g_sa_pair.bars_since_health = 0; // Reset counter

    // Fetch historical prices on hedge TF (same data the scan uses)
    ENUM_TIMEFRAMES hc_entry_tf_unused, hc_hedge_tf;
    SA_ResolveTimeframes(hc_entry_tf_unused, hc_hedge_tf);
    int lookback = SA_ResolveLookback(g_sa_pair.symbolY, hc_hedge_tf,
                                       StatArb_LookbackMode, StatArb_LookbackLength);
    if(lookback < 50) lookback = 50;

    double pricesY[], pricesX[];
    int nY = SA_GetPrices(g_sa_pair.symbolY, hc_hedge_tf, lookback + 1, pricesY);
    int nX = SA_GetPrices(g_sa_pair.symbolX, hc_hedge_tf, lookback + 1, pricesX);
    int ppb = SA_PointsPerBar();
    int n = (int)MathMin(nY, nX) - ppb; // exclude current bar (ppb data points)
    if(n < 50) return true; // Not enough data

    // Align arrays from the end (same pattern as SA_ActivatePair pre-fill)
    int baseY = nY - ppb - n;
    int baseX = nX - ppb - n;

    // Build aligned price arrays for OLS + correlation
    double alignY[], alignX[];
    ArrayResize(alignY, n);
    ArrayResize(alignX, n);
    for(int i = 0; i < n; i++)
    {
        alignY[i] = pricesY[i + baseY];
        alignX[i] = pricesX[i + baseX];
    }

    // Run OLS to get residuals for ADF test (and as fallback for metrics)
    double beta = 0, alpha = 0;
    double residuals[];
    SA_OLS(alignY, alignX, n, beta, alpha, residuals);

    // Cointegration test: Johansen or ADF
    double pvalue = 1.0;
    double adf_stat = 0;
    if(StatArb_CointTest == SA_COINT_JOHANSEN)
    {
        SA_JohansenResult jresult;
        if(SA_JohansenTest(alignY, alignX, n, StatArb_JohansenLags, StatArb_ADF_CritLevel, jresult))
        {
            pvalue = jresult.pvalue;
            adf_stat = -jresult.trace_stat_r0; // Negative for consistency with ADF convention
        }
        else
            pvalue = 1.0; // Test failed — treat as not cointegrated
    }
    else
    {
        adf_stat = SA_ADF_Test(residuals, n, StatArb_ADF_Lags, pvalue);
    }

    // Fix 65: Recompute residuals using the pair's ACTIVE hedge_ratio (which may be
    // Kalman-adapted or Johansen-derived) instead of the fresh OLS beta computed above.
    // Without this, half_life, Hurst, and OU re-estimation use residuals from a different
    // beta than the one actually being traded, producing inconsistent metrics.
    double active_beta = g_sa_pair.hedge_ratio;
    double active_alpha = g_sa_pair.intercept;
    if(MathAbs(active_beta - beta) > 1e-10 || MathAbs(active_alpha - alpha) > 1e-10)
    {
        for(int i = 0; i < n; i++)
            residuals[i] = alignY[i] - active_alpha - active_beta * alignX[i];
    }

    // Fix 64: Always store the actual health check p-value (even when Fix 54 defers
    // metadata writes). Used by Fix 61's CRITICAL warning to show the real failing value.
    g_sa_pair.last_health_pvalue = pvalue;

    // Fix 54: Guard metadata writes during alert-mode virtual positions.
    // Fix 43 principle: spread definition (including diagnostic metadata) must stay frozen
    // while a virtual position is open, for consistent entry/exit evaluation.
    bool can_update_metadata = !(g_sa_pair.position_dir != 0 && StatArb_TradeMode == SA_MODE_ALERT);
    if(can_update_metadata)
    {
        g_sa_pair.adf_statistic = adf_stat;
        g_sa_pair.adf_pvalue = pvalue;
        g_sa_pair.half_life = SA_HalfLife(residuals, n);
        g_sa_pair.correlation = SA_PearsonCorrelation(alignY, alignX, n);
        // Hurst re-check during health check
        if(StatArb_UseHurst)
            g_sa_pair.hurst = SA_HurstExponent(residuals, n);
        // OU re-estimation during health check
        if(StatArb_BoundaryMode == SA_BOUND_OU)
        {
            SA_EstimateOU(residuals, n, 1.0,
                          g_sa_pair.ou_theta, g_sa_pair.ou_mu, g_sa_pair.ou_sigma);
            SA_OUOptimalBoundaries(g_sa_pair.ou_theta, g_sa_pair.ou_mu, g_sa_pair.ou_sigma,
                                   StatArb_OUTransCost, StatArb_OUStopMultiple,
                                   g_sa_pair.ou_entry_long, g_sa_pair.ou_entry_short,
                                   g_sa_pair.ou_exit_long, g_sa_pair.ou_exit_short,
                                   g_sa_pair.ou_stop_long, g_sa_pair.ou_stop_short);
        }
    }

    // Check if still cointegrated (3x tolerance: allow temporary wobble but catch real breakdown)
    if(pvalue > StatArb_ADF_CritLevel * 3.0)
    {
        Print("StatArb: Health check FAILED -- ADF p=", DoubleToString(pvalue, 3),
              " (threshold=", DoubleToString(StatArb_ADF_CritLevel, 3),
              ", trigger=", DoubleToString(StatArb_ADF_CritLevel * 3.0, 3),
              "). Triggering full rescan.");
        return false; // Degraded
    }

    // KPSS confirmation during health check (if enabled)
    if(StatArb_UseKPSS && can_update_metadata)
    {
        double kpss_pval = 0;
        SA_KPSS_Test(residuals, n, kpss_pval);
        if(kpss_pval < StatArb_KPSS_Level)
        {
            Print("StatArb: Health check FAILED -- KPSS rejected stationarity (p=",
                  DoubleToString(kpss_pval, 3), "). Triggering full rescan.");
            return false;
        }
    }

    // Hedge ratio drift detection during health check
    if(StatArb_UseDriftDetect && can_update_metadata && g_sa_pair.initial_beta != 0)
    {
        double drift = SA_HedgeRatioDrift(g_sa_pair.initial_beta, beta);
        if(drift > StatArb_DriftMaxAngle)
        {
            Print("StatArb: Health check FAILED -- Hedge ratio drift ",
                  DoubleToString(drift, 1), "° > max ", DoubleToString(StatArb_DriftMaxAngle, 1),
                  "° (initial beta=", DoubleToString(g_sa_pair.initial_beta, 4),
                  " current=", DoubleToString(beta, 4), "). Triggering full rescan.");
            return false;
        }
    }

    return true; // Still healthy
}

//+------------------------------------------------------------------+
//| SECTION 8: PRICE DATA MANAGEMENT                                  |
//+------------------------------------------------------------------+

int SA_FindPriceCache(string symbol, ENUM_TIMEFRAMES tf)
{
    for(int i = 0; i < g_sa_price_cache_count; i++)
    {
        if(g_sa_price_cache[i].symbol == symbol && g_sa_price_cache[i].timeframe == tf)
            return i;
    }
    return -1;
}

int SA_GetOrCreateCacheEntry(string symbol, ENUM_TIMEFRAMES tf)
{
    int idx = SA_FindPriceCache(symbol, tf);
    if(idx >= 0) return idx;

    // Need a new slot
    if(g_sa_price_cache_count >= SA_MAX_PRICE_CACHE)
    {
        // LRU eviction: find oldest
        int oldest = 0;
        for(int i = 1; i < g_sa_price_cache_count; i++)
        {
            if(g_sa_price_cache[i].last_bar_time < g_sa_price_cache[oldest].last_bar_time)
                oldest = i;
        }
        g_sa_price_cache[oldest].Reset();
        g_sa_price_cache[oldest].symbol = symbol;
        g_sa_price_cache[oldest].timeframe = tf;
        return oldest;
    }

    ArrayResize(g_sa_price_cache, g_sa_price_cache_count + 1);
    idx = g_sa_price_cache_count++;
    g_sa_price_cache[idx].Reset();
    g_sa_price_cache[idx].symbol = symbol;
    g_sa_price_cache[idx].timeframe = tf;
    return idx;
}

//+------------------------------------------------------------------+
//| Data points per bar for current price type (4 for OHLC_RAW, else 1)
//+------------------------------------------------------------------+
int SA_PointsPerBar() { return (StatArb_PriceType == SA_PRICE_OHLC_RAW) ? 4 : 1; }

//+------------------------------------------------------------------+
//| Get prices for a symbol/timeframe, extracting by price type       |
//| Returns data-point count (= bars * SA_PointsPerBar()).            |
//| prices[] is oldest-first.                                         |
//+------------------------------------------------------------------+

int SA_GetPrices(string symbol, ENUM_TIMEFRAMES tf, int bars_needed, double &prices[])
{
    int cache_idx = SA_GetOrCreateCacheEntry(symbol, tf);
    datetime cur_bar = iTime(symbol, tf, 0);

    // Check cache validity
    // Fix 38: Reject cache when iTime returned 0 (history not synced). Without this,
    // a cache entry stored with last_bar_time=0 matches cur_bar=0 indefinitely,
    // serving stale data even after new bars form.
    if(cur_bar > 0 &&
       g_sa_price_cache[cache_idx].valid &&
       g_sa_price_cache[cache_idx].last_bar_time == cur_bar &&
       g_sa_price_cache[cache_idx].count >= ((StatArb_PriceType == SA_PRICE_OHLC_RAW) ? bars_needed * 4 : bars_needed))
    {
        int cnt = g_sa_price_cache[cache_idx].count;
        ArrayResize(prices, cnt);
        ArrayCopy(prices, g_sa_price_cache[cache_idx].prices);
        return cnt;
    }

    // Need to copy fresh data
    if(StatArb_PriceType == SA_PRICE_CLOSE)
    {
        double close_buf[];
        int copied = CopyClose(symbol, tf, 0, bars_needed, close_buf);
        if(copied <= 0) return 0;
        ArrayResize(prices, copied);
        ArrayCopy(prices, close_buf);
    }
    else
    {
        double o_buf[], h_buf[], l_buf[], c_buf[];
        int co = CopyOpen(symbol, tf, 0, bars_needed, o_buf);
        int ch = CopyHigh(symbol, tf, 0, bars_needed, h_buf);
        int cl = CopyLow(symbol, tf, 0, bars_needed, l_buf);
        int cc = CopyClose(symbol, tf, 0, bars_needed, c_buf);
        int copied = (int)MathMin(MathMin(co, ch), MathMin(cl, cc));
        if(copied <= 0) return 0;

        // Fix 37: Align OHLC buffers from the end (most-recent bar) when Copy calls
        // return different counts (race condition: new bar forms between calls).
        // Without this, o_buf[i] and h_buf[i] can refer to different calendar bars.
        int oOff = co - copied;
        int hOff = ch - copied;
        int lOff = cl - copied;
        int cOff = cc - copied;

        if(StatArb_PriceType == SA_PRICE_OHLC_RAW)
        {
            ArrayResize(prices, copied * 4);
            for(int i = 0; i < copied; i++)
            {
                prices[i*4    ] = o_buf[i+oOff];
                prices[i*4 + 1] = h_buf[i+hOff];
                prices[i*4 + 2] = l_buf[i+lOff];
                prices[i*4 + 3] = c_buf[i+cOff];
            }
        }
        else
        {
            ArrayResize(prices, copied);
            for(int i = 0; i < copied; i++)
            {
                switch(StatArb_PriceType)
                {
                    case SA_PRICE_OHLC:     prices[i] = (o_buf[i+oOff] + h_buf[i+hOff] + l_buf[i+lOff] + c_buf[i+cOff]) / 4.0; break;
                    case SA_PRICE_TYPICAL:  prices[i] = (h_buf[i+hOff] + l_buf[i+lOff] + c_buf[i+cOff]) / 3.0; break;
                    case SA_PRICE_MEDIAN:   prices[i] = (h_buf[i+hOff] + l_buf[i+lOff]) / 2.0; break;
                    case SA_PRICE_WEIGHTED: prices[i] = (h_buf[i+hOff] + l_buf[i+lOff] + 2.0 * c_buf[i+cOff]) / 4.0; break;
                    default:                prices[i] = c_buf[i+cOff]; break;
                }
            }
        }
    }

    int cnt = ArraySize(prices);

    // Update cache — only if we have a valid bar time reference (Fix 38)
    if(cur_bar > 0)
    {
        ArrayResize(g_sa_price_cache[cache_idx].prices, cnt);
        ArrayCopy(g_sa_price_cache[cache_idx].prices, prices);
        g_sa_price_cache[cache_idx].count = cnt;
        g_sa_price_cache[cache_idx].last_bar_time = cur_bar;
        g_sa_price_cache[cache_idx].valid = true;
    }

    return cnt;
}

//+------------------------------------------------------------------+
//| SECTION 9: CANDIDATE SYMBOL LIST                                  |
//+------------------------------------------------------------------+

void SA_BuildCandidateList()
{
    g_sa_candidate_count = 0;

    if(StatArb_SymbolSource == SA_SRC_MARKETWATCH)
    {
        int total = SymbolsTotal(true);
        ArrayResize(g_sa_candidates, (int)MathMin(total, SA_MAX_CANDIDATES));
        for(int i = 0; i < total && g_sa_candidate_count < SA_MAX_CANDIDATES; i++)
        {
            string s = SymbolName(i, true);
            if(SymbolInfoInteger(s, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED)
            {
                g_sa_candidates[g_sa_candidate_count++] = s;
            }
        }
    }
    else
    {
        string tempList = StatArb_SymbolList;
        StringTrimLeft(tempList);
        StringTrimRight(tempList);
        if(StringLen(tempList) == 0) return;

        string parts[];
        int count = StringSplit(tempList, ',', parts);
        ArrayResize(g_sa_candidates, (int)MathMin(count, SA_MAX_CANDIDATES));
        for(int i = 0; i < count && g_sa_candidate_count < SA_MAX_CANDIDATES; i++)
        {
            StringTrimLeft(parts[i]);
            StringTrimRight(parts[i]);
            if(StringLen(parts[i]) > 0 && SymbolSelect(parts[i], true))
            {
                g_sa_candidates[g_sa_candidate_count++] = parts[i];
            }
            else if(StringLen(parts[i]) > 0)
            {
                Print("StatArb: WARNING - Symbol '", parts[i], "' not found or cannot be selected");
            }
        }
    }

    Print("StatArb: Built candidate list with ", g_sa_candidate_count, " symbols");
}

//+------------------------------------------------------------------+
//| SECTION 10: PAIR SELECTION ENGINE                                  |
//+------------------------------------------------------------------+

bool SA_ScanForBestPair(string symbolY, SA_PairResult &best_result)
{
    ENUM_TIMEFRAMES entry_tf, hedge_tf;
    SA_ResolveTimeframes(entry_tf, hedge_tf);

    int lookback = SA_ResolveLookback(symbolY, hedge_tf, StatArb_LookbackMode, StatArb_LookbackLength);

    // Get Y prices
    double pricesY[];
    int barsY = SA_GetPrices(symbolY, hedge_tf, lookback, pricesY);
    if(barsY < 50)
    {
        Print("StatArb: Not enough data for ", symbolY, " (", barsY / SA_PointsPerBar(), " bars)");
        return false;
    }

    // Collect all valid results for top-N panel
    SA_PairResult results[];
    int result_count = 0;

    g_sa_scan_total = 0;
    g_sa_scan_corr_passed = 0;
    g_sa_scan_cointegrated = 0;

    for(int i = 0; i < g_sa_candidate_count; i++)
    {
        string symbolX = g_sa_candidates[i];
        if(symbolX == symbolY) continue;
        // Multi-pair: skip candidates whose X symbol is already in use by OTHER pairs.
        // Only check symbolX — symbolY is always the chart symbol and is shared by all pairs
        // by design (scanning always uses chart symbol as Y).
        // (skip_idx = g_sa_current_idx allows rescan to replace current pair's symbols)
        if(StatArb_NoOverlap && StatArb_MaxPairs > 1)
        {
            if(SA_IsSymbolInUse(g_sa_pairs, ArraySize(g_sa_pairs), symbolX, g_sa_current_idx))
                continue;
        }
        g_sa_scan_total++;

        // Get X prices
        double pricesX[];
        int barsX = SA_GetPrices(symbolX, hedge_tf, lookback, pricesX);
        if(barsX < 50) continue;

        // Align to shorter length
        int n = (int)MathMin(barsY, barsX);
        // Use the most recent n bars from each
        double alignedY[], alignedX[];
        ArrayResize(alignedY, n);
        ArrayResize(alignedX, n);
        ArrayCopy(alignedY, pricesY, 0, barsY - n, n);
        ArrayCopy(alignedX, pricesX, 0, barsX - n, n);

        // Step 1: Correlation pre-filter
        double corr = SA_PearsonCorrelation(alignedY, alignedX, n);
        if(StatArb_CorrMode == SA_CORR_POSITIVE)
        {
            if(corr < StatArb_MinCorrelation) continue;  // Positive only
        }
        else
        {
            if(MathAbs(corr) < StatArb_MinCorrelation) continue;  // Both directions
        }
        g_sa_scan_corr_passed++;

        // Step 2: Regression (OLS or TLS)
        double beta, alpha;
        double residuals[];
        if(StatArb_HedgeMode == SA_HEDGE_TLS)
            SA_TLS(alignedY, alignedX, n, beta, alpha, residuals);
        else
            SA_OLS(alignedY, alignedX, n, beta, alpha, residuals);

        if(MathAbs(beta) < 1e-10 || MathAbs(beta) > 100.0) continue; // Degenerate or unreasonable

        // Step 3: Cointegration test (ADF or Johansen)
        double adf_pvalue = 1.0;
        double adf_stat = 0;
        bool cointegrated = false;
        double johansen_trace_val = 0;
        double johansen_beta_val = beta;

        if(StatArb_CointTest == SA_COINT_JOHANSEN)
        {
            SA_JohansenResult jresult;
            if(SA_JohansenTest(alignedY, alignedX, n, StatArb_JohansenLags,
                                StatArb_ADF_CritLevel, jresult))
            {
                cointegrated = jresult.cointegrated;
                adf_pvalue = jresult.pvalue;
                adf_stat = -jresult.trace_stat_r0; // Negative for score compat
                johansen_trace_val = jresult.trace_stat_r0;
                johansen_beta_val = -jresult.eigenvec_2; // Cointegrating vector [1, -beta]
                // If Johansen provided a different beta, recompute residuals
                if(MathAbs(johansen_beta_val - beta) > 1e-10)
                {
                    beta = johansen_beta_val;
                    double sum_res = 0;
                    for(int j = 0; j < n; j++) sum_res += alignedY[j] - beta * alignedX[j];
                    alpha = sum_res / n;
                    for(int j = 0; j < n; j++) residuals[j] = alignedY[j] - alpha - beta * alignedX[j];
                }
            }
        }
        else
        {
            adf_stat = SA_ADF_Test(residuals, n, StatArb_ADF_Lags, adf_pvalue);
            cointegrated = (adf_pvalue <= StatArb_ADF_CritLevel);
        }

        if(!cointegrated) continue;
        g_sa_scan_cointegrated++;

        // Step 3b: KPSS stationarity confirmation (dual-test paradigm)
        // ADF H0: non-stationary (reject = good). KPSS H0: stationary (fail to reject = good).
        // Both must agree: ADF rejects AND KPSS fails to reject = robust stationarity.
        double kpss_stat_val = 0;
        if(StatArb_UseKPSS)
        {
            double kpss_pvalue = 0;
            kpss_stat_val = SA_KPSS_Test(residuals, n, kpss_pvalue);
            // KPSS: low p-value = reject stationarity = bad
            if(kpss_pvalue < StatArb_KPSS_Level)
            {
                if(g_sa_scan_cointegrated <= 10)
                    Print("StatArb: KPSS rejected stationarity for ", symbolY, "/", symbolX,
                          " (KPSS p=", DoubleToString(kpss_pvalue, 3),
                          " stat=", DoubleToString(kpss_stat_val, 4), ")");
                continue;
            }
        }

        // Step 3c: In-sample/Out-of-sample validation
        // Estimate cointegration on IS (first 70%), validate on OOS (last 30%)
        double oos_pval = 0;
        if(StatArb_UseOOS)
        {
            double oos_beta_est = 0, oos_adf_s = 0;
            oos_pval = SA_OOS_Validate(alignedY, alignedX, n,
                                        StatArb_OOS_Ratio, StatArb_ADF_Lags,
                                        oos_beta_est, oos_adf_s);
            if(oos_pval > StatArb_OOS_MaxPValue)
            {
                if(g_sa_scan_cointegrated <= 10)
                    Print("StatArb: OOS validation FAILED for ", symbolY, "/", symbolX,
                          " (OOS ADF p=", DoubleToString(oos_pval, 3),
                          " IS beta=", DoubleToString(beta, 4),
                          " OOS beta=", DoubleToString(oos_beta_est, 4), ")");
                continue;
            }

            // Step 3d: Hedge ratio drift detection (cosine similarity)
            if(StatArb_UseDriftDetect)
            {
                double drift_angle = SA_HedgeRatioDrift(oos_beta_est, beta);
                if(drift_angle > StatArb_DriftMaxAngle)
                {
                    if(g_sa_scan_cointegrated <= 10)
                        Print("StatArb: Hedge drift REJECTED for ", symbolY, "/", symbolX,
                              " (angle=", DoubleToString(drift_angle, 1),
                              "° > max=", DoubleToString(StatArb_DriftMaxAngle, 1), "°)");
                    continue;
                }
            }
        }

        // Step 4: Half-life
        double hl = SA_HalfLife(residuals, n);
        if(hl <= 1 || hl > lookback * 0.5) continue; // Unrealistic

        // Step 4b: Hurst exponent filter
        double hurst = 0.5;
        if(StatArb_UseHurst)
        {
            hurst = SA_HurstExponent(residuals, n);
            // Log all Hurst values for first 30 cointegrated pairs (diagnostics)
            if(g_sa_scan_cointegrated <= 30)
                Print("StatArb: Hurst ", symbolY, "/", symbolX,
                      " H=", DoubleToString(hurst, 3),
                      (hurst >= StatArb_MaxHurst ? " REJECTED" : " OK"),
                      " (max=", DoubleToString(StatArb_MaxHurst, 3), " n=", n, ")");
            if(hurst >= StatArb_MaxHurst) continue;
        }

        // Step 5: Current Z-score from spread
        int zwin = (int)MathMin(n, SA_ResolveLookback(symbolY, hedge_tf, StatArb_ZScoreWinMode, StatArb_ZScoreWinLength));
        if(zwin < 5) zwin = 5;
        double zmean = 0, zstd = 0;
        int zstart = (int)MathMax(0, n - zwin);
        int zcount = n - zstart;
        for(int j = zstart; j < n; j++) zmean += residuals[j];
        zmean /= zcount;
        for(int j = zstart; j < n; j++) zstd += (residuals[j] - zmean) * (residuals[j] - zmean);
        zstd = MathSqrt(zstd / MathMax(1, zcount - 1)); // Sample std
        double cur_z = (zstd > 1e-12) ? (residuals[n - 1] - zmean) / zstd : 0;

        // Step 6: Composite score
        // Z-bonus: reward z near entry threshold, linearly declining to 0 at stop.
        // Pairs with |z| beyond stop get no z-bonus (would immediately stop out).
        // Pairs with |z| below entry get a small bonus proportional to proximity.
        double abs_z = MathAbs(cur_z);
        double z_bonus = 0;
        if(abs_z >= StatArb_EntryZ && abs_z < StatArb_StopZ)
            z_bonus = (StatArb_StopZ - abs_z) / (StatArb_StopZ - StatArb_EntryZ) * 5.0;
        else if(abs_z > 0 && abs_z < StatArb_EntryZ)
            z_bonus = abs_z / StatArb_EntryZ * 2.0; // small bonus for approaching entry

        double hurst_bonus = StatArb_UseHurst ? (0.5 - hurst) * 20.0 : 0;
        // OOS stability bonus: lower OOS p-value = more stable relationship
        double oos_bonus = (StatArb_UseOOS && oos_pval < StatArb_OOS_MaxPValue) ?
                           (StatArb_OOS_MaxPValue - oos_pval) / StatArb_OOS_MaxPValue * 15.0 : 0;
        double score = (-adf_stat) * 10.0
                     + MathAbs(corr) * 20.0
                     - MathLog(MathMax(hl, 1.0)) * 5.0
                     + z_bonus
                     + hurst_bonus
                     + oos_bonus;

        ArrayResize(results, result_count + 1, 50);
        results[result_count].symbolY = symbolY;
        results[result_count].symbolX = symbolX;
        results[result_count].correlation = corr;
        results[result_count].hedge_ratio = beta;
        results[result_count].intercept = alpha;
        results[result_count].adf_statistic = adf_stat;
        results[result_count].adf_pvalue = adf_pvalue;
        results[result_count].is_cointegrated = true;
        results[result_count].half_life = hl;
        results[result_count].spread_std = zstd;
        results[result_count].current_zscore = cur_z;
        results[result_count].score = score;
        results[result_count].hurst = hurst;
        results[result_count].ou_theta = 0;
        results[result_count].ou_mu = 0;
        results[result_count].ou_sigma = 0;
        results[result_count].johansen_trace = johansen_trace_val;
        results[result_count].johansen_beta = johansen_beta_val;
        results[result_count].oos_pvalue = oos_pval;
        results[result_count].kpss_stat = kpss_stat_val;
        result_count++;
    }

    if(result_count == 0) return false;

    // Sort by score descending (simple selection sort for small N)
    for(int i = 0; i < result_count - 1; i++)
    {
        int best_idx = i;
        for(int j = i + 1; j < result_count; j++)
        {
            if(results[j].score > results[best_idx].score)
                best_idx = j;
        }
        if(best_idx != i)
        {
            SA_PairResult tmp;
            tmp.CopyFrom(results[i]);
            results[i].CopyFrom(results[best_idx]);
            results[best_idx].CopyFrom(tmp);
        }
    }

    // Update top-N for panel
    int topN = (int)MathMin(result_count, g_sa_panel_topn);
    ArrayResize(g_sa_top_pairs, topN);
    g_sa_top_count = topN;
    for(int i = 0; i < topN; i++)
    {
        g_sa_top_pairs[i].symbolY       = results[i].symbolY;
        g_sa_top_pairs[i].symbolX       = results[i].symbolX;
        g_sa_top_pairs[i].correlation   = results[i].correlation;
        g_sa_top_pairs[i].adf_stat      = results[i].adf_statistic;
        g_sa_top_pairs[i].adf_pvalue    = results[i].adf_pvalue;
        g_sa_top_pairs[i].hedge_ratio   = results[i].hedge_ratio;
        g_sa_top_pairs[i].half_life     = results[i].half_life;
        g_sa_top_pairs[i].current_zscore = results[i].current_zscore;
        g_sa_top_pairs[i].score         = results[i].score;
        g_sa_top_pairs[i].position_dir  = 0;
        g_sa_top_pairs[i].is_active     = false;
        g_sa_top_pairs[i].hurst         = results[i].hurst;
    }

    best_result.CopyFrom(results[0]);
    return true;
}

//+------------------------------------------------------------------+
//| SECTION 11: TRADING ENGINE                                         |
//+------------------------------------------------------------------+

double SA_ClampLotSize(string symbol, double lot)
{
    double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double step    = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

    lot = MathMax(lot, min_lot);
    lot = MathMin(lot, max_lot);
    if(step > 0)
    {
        // Add epsilon before floor to compensate for IEEE 754 representation error.
        // Without this, MathFloor(0.03 / 0.01) can give 2 instead of 3.
        lot = MathFloor(lot / step + 1e-9) * step;
    }
    // Derive decimal places from step size (not hardcoded to 2)
    // step=0.01→2, step=0.001→3, step=0.1→1, step=1.0→0, step=10→0
    int lot_digits = (step > 0) ? (int)MathMax(0, (int)MathCeil(-MathLog10(step + 1e-15))) : 2;
    return NormalizeDouble(lot, lot_digits);
}

bool SA_SpreadCheckBothLegs()
{
    string symY = g_sa_pair.symbolY;
    string symX = g_sa_pair.symbolX;

    if(StatArb_SpreadLimit <= 0) return true; // Disabled

    double askY = SymbolInfoDouble(symY, SYMBOL_ASK);
    double bidY = SymbolInfoDouble(symY, SYMBOL_BID);
    if(askY <= 0 || bidY <= 0) { Print("StatArb: Zero price on ", symY); return false; }
    double spreadY = normalization(symY, askY, bidY, adr_atr_norml_);
    // Reject zero (normalization returns 0 when ADR/ATR unavailable) and invalid values
    if(!MathIsValidNumber(spreadY) || spreadY <= 0 || spreadY > StatArb_SpreadLimit)
    {
        Print("StatArb: Spread invalid/too high on ", symY, ": ", DoubleToString(spreadY, 5));
        return false;
    }

    double askX = SymbolInfoDouble(symX, SYMBOL_ASK);
    double bidX = SymbolInfoDouble(symX, SYMBOL_BID);
    if(askX <= 0 || bidX <= 0) { Print("StatArb: Zero price on ", symX); return false; }
    double spreadX = normalization(symX, askX, bidX, adr_atr_norml_);
    if(!MathIsValidNumber(spreadX) || spreadX <= 0 || spreadX > StatArb_SpreadLimit)
    {
        Print("StatArb: Spread invalid/too high on ", symX, ": ", DoubleToString(spreadX, 5));
        return false;
    }

    return true;
}

void SA_RefreshTickets()
{
    ulong sa_magic = SA_PairMagic(g_sa_current_idx);
    g_sa_pair.ticketY = 0;
    g_sa_pair.ticketX = 0;

    // Fix 68: Count ALL StatArb positions across all pair slots, not just the current
    // pair's magic. Previously, g_sa_active_trades was reset to 0 here and only counted
    // positions matching one pair's magic — wrong in multi-pair mode.
    g_sa_active_trades = 0;
    int max_pairs = ArraySize(g_sa_pairs);

    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket <= 0) continue;
        ulong pos_magic = (ulong)PositionGetInteger(POSITION_MAGIC);

        // Count position if it belongs to ANY of our pair slots
        for(int p = 0; p < max_pairs; p++)
        {
            if(pos_magic == SA_PairMagic(p)) { g_sa_active_trades++; break; }
        }

        // Match current pair's tickets
        if(pos_magic != sa_magic) continue;
        string sym = PositionGetString(POSITION_SYMBOL);
        // Prefer highest ticket number (most recent) in case of duplicates
        if(sym == g_sa_pair.symbolY && ticket > g_sa_pair.ticketY)
            g_sa_pair.ticketY = ticket;
        else if(sym == g_sa_pair.symbolX && ticket > g_sa_pair.ticketX)
            g_sa_pair.ticketX = ticket;
    }
}

void SA_CloseOneLeg(string symbol)
{
    ulong sa_magic = SA_PairMagic(g_sa_current_idx);
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket <= 0) continue;
        if((ulong)PositionGetInteger(POSITION_MAGIC) != sa_magic) continue;
        if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
        sa_trade.PositionClose(ticket);
    }
}

//+------------------------------------------------------------------+
//| Open pair trade (both legs)                                       |
//+------------------------------------------------------------------+

bool SA_OpenPairTrade(int direction)
{
    string symY = g_sa_pair.symbolY;
    string symX = g_sa_pair.symbolX;

    // Stale tick check: ensure both symbols have recent data
    MqlTick tickY, tickX;
    if(!SymbolInfoTick(symY, tickY) || !SymbolInfoTick(symX, tickX))
    {
        Print("StatArb: Cannot get tick data for ", symY, " or ", symX);
        return false;
    }
    datetime now = TimeCurrent();
    if(now - tickY.time > 60 || now - tickX.time > 60)
    {
        Print("StatArb: Stale tick data. Y age=", (int)(now - tickY.time),
              "s  X age=", (int)(now - tickX.time), "s. Skipping trade.");
        return false;
    }

    if(!SA_SpreadCheckBothLegs()) return false;

    // X leg direction depends on both trade direction AND sign of hedge_ratio.
    // Spread S = Y - beta*X. Long spread = Long Y, Short(beta*X).
    // If beta > 0: Short(beta*X) = Short X.  If beta < 0: Short(beta*X) = Long X.
    int x_dir = (g_sa_pair.hedge_ratio >= 0) ? -direction : direction;

    // Calculate lots
    double lotY = SA_ClampLotSize(symY, StatArb_LotSize);

    // Use correct bid/ask based on actual trade direction of each leg
    double priceY = (direction > 0) ? SymbolInfoDouble(symY, SYMBOL_ASK) : SymbolInfoDouble(symY, SYMBOL_BID);
    double priceX = (x_dir > 0) ? SymbolInfoDouble(symX, SYMBOL_ASK) : SymbolInfoDouble(symX, SYMBOL_BID);

    if(priceY <= 0 || priceX <= 0)
    {
        Print("StatArb: Cannot open trade - zero price. Y=", priceY, " X=", priceX);
        return false;
    }

    // Dollar-neutral lot sizing for leg X
    double contractY = SymbolInfoDouble(symY, SYMBOL_TRADE_CONTRACT_SIZE);
    double contractX = SymbolInfoDouble(symX, SYMBOL_TRADE_CONTRACT_SIZE);
    if(contractY <= 0 || contractX <= 0)
    {
        Print("StatArb: Invalid contract size. Y=", contractY, " X=", contractX, ". Skipping.");
        return false;
    }
    double notionalY = contractY * priceY;
    double notionalX = contractX * priceX;
    double lotX = lotY;
    if(notionalX > 0)
        lotX = lotY * MathAbs(g_sa_pair.hedge_ratio) * notionalY / notionalX;
    lotX = SA_ClampLotSize(symX, lotX);

    // Guard against zero lots (restricted symbols) and extreme lot ratios
    if(lotY <= 0 || lotX <= 0)
    {
        Print("StatArb: Zero lot size after clamping. Y=", lotY, " X=", lotX, ". Skipping.");
        return false;
    }
    if(lotX / lotY > 20.0 || lotY / lotX > 20.0)
    {
        Print("StatArb: Lot ratio too extreme: Y=", lotY, " X=", lotX, ". Skipping.");
        return false;
    }

    sa_trade.SetDeviationInPoints(10);
    sa_trade.SetExpertMagicNumber(SA_PairMagic(g_sa_current_idx));

    string comment = "SA:" + symY + "/" + symX;
    bool successY = false, successX = false;

    successY = (direction > 0) ? sa_trade.Buy(lotY, symY, 0, 0, 0, comment)
                               : sa_trade.Sell(lotY, symY, 0, 0, 0, comment);
    if(successY)
        successX = (x_dir > 0) ? sa_trade.Buy(lotX, symX, 0, 0, 0, comment)
                               : sa_trade.Sell(lotX, symX, 0, 0, 0, comment);

    if(successY && successX)
    {
        g_sa_pair.position_dir = direction;
        g_sa_pair.entry_time = TimeCurrent();
        g_sa_pair.entry_zscore = g_sa_pair.current_zscore;
        g_sa_pair.bars_in_trade = 0;
        SA_RefreshTickets();
        Print("StatArb: Opened pair trade. Dir=", direction,
              " Y=", symY, "(", lotY, ") X=", symX, "(", lotX, ")",
              " Z=", DoubleToString(g_sa_pair.current_zscore, 2));
        return true;
    }
    else
    {
        if(successY && !successX)
        {
            Print("StatArb: WARNING - Leg X (", symX, ") failed: ", sa_trade.ResultRetcodeDescription(),
                  ". Closing leg Y.");
            SA_CloseOneLeg(symY);
            // Verify Y was actually closed — if not, mark position so orphan detection can retry
            ulong sa_magic = SA_PairMagic(g_sa_current_idx);
            bool y_still_open = false;
            for(int k = PositionsTotal() - 1; k >= 0; k--)
            {
                ulong t = PositionGetTicket(k);
                if(t > 0 && (ulong)PositionGetInteger(POSITION_MAGIC) == sa_magic
                   && PositionGetString(POSITION_SYMBOL) == symY)
                { y_still_open = true; break; }
            }
            if(y_still_open)
            {
                Print("StatArb: CRITICAL - Failed to unwind Y leg on ", symY,
                      ". Marking for orphan detection retry.");
                // Mark as position so orphan detection can close the dangling Y leg.
                // Set direction so the orphan block (position_dir != 0) fires next tick.
                g_sa_pair.position_dir = direction;
                g_sa_pair.entry_time = TimeCurrent();
                g_sa_pair.ticketX = 0; // X never opened — ensure orphan detector sees the asymmetry
                SA_RefreshTickets();
                return false;
            }
        }
        if(!successY)
        {
            Print("StatArb: WARNING - Leg Y (", symY, ") failed: ", sa_trade.ResultRetcodeDescription());
        }
        return false;
    }
}

//+------------------------------------------------------------------+
//| Close pair trade (both legs)                                      |
//+------------------------------------------------------------------+

void SA_ClosePairTrade(string reason)
{
    // Read PnL BEFORE closing (positions won't exist after close)
    // Fix 36: Include swap (overnight financing) and commission for accurate PnL.
    // POSITION_PROFIT alone excludes these costs, which can be significant on
    // pairs trades held across rollovers with two legs.
    double pnl = 0;
    if(g_sa_pair.ticketY > 0 && PositionSelectByTicket(g_sa_pair.ticketY))
        pnl += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);
    if(g_sa_pair.ticketX > 0 && PositionSelectByTicket(g_sa_pair.ticketX))
        pnl += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);

    // Now close both legs
    bool closedY = true, closedX = true;
    if(g_sa_pair.ticketY > 0)
    {
        closedY = sa_trade.PositionClose(g_sa_pair.ticketY);
        if(!closedY)
            Print("StatArb: Failed to close Y leg ticket ", g_sa_pair.ticketY,
                  ": ", sa_trade.ResultRetcodeDescription());
    }
    if(g_sa_pair.ticketX > 0)
    {
        closedX = sa_trade.PositionClose(g_sa_pair.ticketX);
        if(!closedX)
            Print("StatArb: Failed to close X leg ticket ", g_sa_pair.ticketX,
                  ": ", sa_trade.ResultRetcodeDescription());
    }

    int duration_sec = (int)(TimeCurrent() - g_sa_pair.entry_time);
    int hours = duration_sec / 3600;
    int mins = (duration_sec % 3600) / 60;

    Print("StatArb: Closed pair trade. Reason: ", reason,
          " Z=", DoubleToString(g_sa_pair.current_zscore, 2),
          " PnL=$", DoubleToString(pnl, 2),
          " Duration=", hours, "h ", mins, "m");

    if(closedY && closedX)
    {
        // ML hook: store experience and train on trade close
        // Called BEFORE zeroing tickets/position_dir so feature 17 (position_state)
        // reads correctly. Note: feature 13 (unrealized PnL) will be 0 because
        // positions are already closed — acceptable since close_features (next_state)
        // are only used in the non-terminal bootstrap branch (dead code for v1).
        if(StatArb_MLMode != SA_ML_OFF)
            SAML_OnTradeClose(g_sa_pair, pnl);

        g_sa_pair.ticketY = 0;
        g_sa_pair.ticketX = 0;
        g_sa_pair.position_dir = 0;
        // Fix 33: Record close bar time for re-entry cooldown. Without this,
        // after a stop-loss the z-score still exceeds EntryZ and the EA immediately
        // re-enters, creating a destructive churn cycle (stop → re-enter → stop → ...).
        g_sa_pair.last_close_time = TimeCurrent();

        // Fix 69: Recount active trades after close. SA_RefreshTickets only runs when
        // position_dir != 0 (line 2230), so after the LAST pair closes, the counter
        // goes stale — panel shows "Active trades: 1" when there are actually 0.
        g_sa_active_trades = 0;
        int cnt_pairs = ArraySize(g_sa_pairs);
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong t = PositionGetTicket(i);
            if(t <= 0) continue;
            ulong pm = (ulong)PositionGetInteger(POSITION_MAGIC);
            for(int p = 0; p < cnt_pairs; p++)
            {
                if(pm == SA_PairMagic(p)) { g_sa_active_trades++; break; }
            }
        }
    }
    else
    {
        // Partial close: only zero the leg that succeeded; orphan detector handles the other
        if(closedY) g_sa_pair.ticketY = 0;
        if(closedX) g_sa_pair.ticketX = 0;
        Print("StatArb: CRITICAL - Partial close! Y=", closedY, " X=", closedX,
              " — will retry on next tick via orphan detection.");
    }
}

//+------------------------------------------------------------------+
//| SECTION 12: ALERT ENGINE                                          |
//+------------------------------------------------------------------+

void SA_SendAlert(string msg)
{
    // Cooldown: one alert per bar (on entry timeframe)
    ENUM_TIMEFRAMES alert_tf, hedge_tf_unused;
    SA_ResolveTimeframes(alert_tf, hedge_tf_unused);
    datetime cur_bar = iTime(g_sa_pair.symbolY, alert_tf, 0);
    if(cur_bar == g_sa_last_alert_time) return;
    g_sa_last_alert_time = cur_bar;

    g_sa_alerts_sent++;

    if(send_alert)
        Alert(msg);
    if(send_notification)
        SendNotification(msg);
    if(send_telegram_message)
        SendTelegramMessage(TelegramApiUrl, TelegramBotToken, ChatId, msg, "");

    Print("StatArb Alert: ", msg);
}

string SA_DirectionStr(int dir)
{
    if(dir == 0) return "FLAT";
    string yStr = (dir > 0) ? "LONG " : "SHORT ";
    int x_d = (g_sa_pair.hedge_ratio >= 0) ? -dir : dir;
    string xStr = (x_d > 0) ? "LONG " : "SHORT ";
    return yStr + g_sa_pair.symbolY + " / " + xStr + g_sa_pair.symbolX;
}

//+------------------------------------------------------------------+
//| SECTION 13: MAIN TRADE MANAGEMENT                                 |
//+------------------------------------------------------------------+

void SA_ActivatePair(const SA_PairResult &result)
{
    ENUM_TIMEFRAMES entry_tf, hedge_tf_unused;
    SA_ResolveTimeframes(entry_tf, hedge_tf_unused);

    g_sa_pair.active = true;
    // Fix 62: Defensive reset of position tracking state. Callers should already ensure
    // position_dir==0 before calling (AUTO: rescan_allowed guard; ALERT: explicit void),
    // but SA_ActivatePair should be self-contained and not rely on caller discipline.
    g_sa_pair.ticketY = 0;
    g_sa_pair.ticketX = 0;
    g_sa_pair.position_dir = 0;
    g_sa_pair.entry_time = 0;
    g_sa_pair.entry_zscore = 0;
    // Fix 44: Reset cooldown from previous pair — stale last_close_time from
    // the old pair would block entries on the new pair for up to one bar.
    g_sa_pair.last_close_time = 0;
    g_sa_pair.symbolY = result.symbolY;
    g_sa_pair.symbolX = result.symbolX;
    g_sa_pair.hedge_ratio = result.hedge_ratio;
    g_sa_pair.intercept = result.intercept;
    g_sa_pair.adf_statistic = result.adf_statistic;
    g_sa_pair.adf_pvalue = result.adf_pvalue;
    g_sa_pair.correlation = result.correlation;
    g_sa_pair.half_life = result.half_life;
    g_sa_pair.hurst = result.hurst;
    g_sa_pair.initial_beta = result.hedge_ratio; // Save for drift detection
    g_sa_pair.last_scan_time = TimeCurrent();
    g_sa_pair.bars_in_trade = 0;

    // Reset ML pair state (prev_zscore seeded AFTER pre-fill below, not here —
    // current_zscore at this point is stale from the old pair / pre-rescan value)
    if(StatArb_MLMode != SA_ML_OFF && g_sa_ml_initialized)
    {
        int pidx = g_sa_current_idx;
        if(pidx >= 0 && pidx < ArraySize(g_sa_ml_pair_state))
            g_sa_ml_pair_state[pidx].Reset();
    }

    // Initialize Z-score ring buffer
    // Use entry_tf since ongoing spread pushes come at entry_tf frequency
    int zwin = SA_ResolveLookback(result.symbolY, entry_tf, StatArb_ZScoreWinMode, StatArb_ZScoreWinLength);
    if(zwin < 5) zwin = 5;
    g_sa_pair.ring_size = zwin;
    g_sa_pair.ring_head = 0;
    g_sa_pair.ring_count = 0;
    g_sa_pair.ring_updates = 0;
    g_sa_pair.rolling_sum = 0;
    g_sa_pair.rolling_sum_sq = 0;
    // Fix 63: Prevent stale z-score/spread from previous pair showing on the panel
    // if the pre-fill is too short (< 5 bars) to produce a valid z-score.
    g_sa_pair.current_zscore = 0;
    g_sa_pair.current_spread = 0;
    ArrayResize(g_sa_pair.spread_ring, zwin);
    ArrayInitialize(g_sa_pair.spread_ring, 0);

    // Pre-fill ring buffer with historical COMPLETED bar data so z-score is
    // immediately available — avoids a cold-start delay of ring_size bars
    // where no entry signals can fire.
    // Exclude the last element (current incomplete bar) for consistency with
    // SA_ManageTrade BAR mode which uses iClose(symbol, entry_tf, 1).
    double pfY[], pfX[];
    int pfnY = SA_GetPrices(result.symbolY, entry_tf, zwin + 1, pfY);
    int pfnX = SA_GetPrices(result.symbolX, entry_tf, zwin + 1, pfX);
    int ppb = SA_PointsPerBar();
    int pfn = (int)MathMin(pfnY, pfnX) - ppb; // exclude current (open) bar
    if(pfn >= 5)
    {
        // Fix 29a: Align arrays from the end (most-recent bars) when pfnY != pfnX.
        // Without this, pfY[k] and pfX[k] refer to different calendar bars if one
        // symbol has more available history than the other.
        int baseY = pfnY - ppb - pfn;  // offset so pfY[baseY + pfn - 1] = last completed bar
        int baseX = pfnX - ppb - pfn;
        int pfstart = (pfn > zwin) ? pfn - zwin : 0;
        for(int k = pfstart; k < pfn; k++)
            SA_UpdateZScore(pfY[k + baseY] - result.hedge_ratio * pfX[k + baseX] - result.intercept);

        // Kalman warm-up: feed pre-fill data through Kalman filter
        if(StatArb_HedgeAdapt == SA_ADAPT_KALMAN)
        {
            g_sa_pair.kalman_beta  = result.hedge_ratio;
            g_sa_pair.kalman_alpha = result.intercept;
            g_sa_pair.kalman_P00   = 1.0;
            g_sa_pair.kalman_P01   = 0;
            g_sa_pair.kalman_P10   = 0;
            g_sa_pair.kalman_P11   = 1.0;
            g_sa_pair.kalman_initialized = true;
            for(int k = pfstart; k < pfn; k++)
                SA_KalmanUpdate(g_sa_pair, pfY[k + baseY], pfX[k + baseX],
                                StatArb_KalmanQ, StatArb_KalmanR);
        }

        // OU parameter estimation from historical residuals
        if(StatArb_BoundaryMode == SA_BOUND_OU)
        {
            int res_count = pfn - pfstart;
            double ou_residuals[];
            ArrayResize(ou_residuals, res_count);
            for(int k = 0; k < res_count; k++)
            {
                int idx = pfstart + k;
                ou_residuals[k] = pfY[idx + baseY] - g_sa_pair.hedge_ratio * pfX[idx + baseX] - g_sa_pair.intercept;
            }
            SA_EstimateOU(ou_residuals, res_count, 1.0,
                          g_sa_pair.ou_theta, g_sa_pair.ou_mu, g_sa_pair.ou_sigma);
            SA_OUOptimalBoundaries(g_sa_pair.ou_theta, g_sa_pair.ou_mu, g_sa_pair.ou_sigma,
                                   StatArb_OUTransCost, StatArb_OUStopMultiple,
                                   g_sa_pair.ou_entry_long, g_sa_pair.ou_entry_short,
                                   g_sa_pair.ou_exit_long, g_sa_pair.ou_exit_short,
                                   g_sa_pair.ou_stop_long, g_sa_pair.ou_stop_short);
        }

        // Initialize CUSUM structural break detector from historical spreads
        if(StatArb_UseCUSUM && pfn >= 30)
        {
            int cu_count = pfn - pfstart;
            double cusum_spreads[];
            ArrayResize(cusum_spreads, cu_count);
            for(int k = 0; k < cu_count; k++)
            {
                int idx = pfstart + k;
                cusum_spreads[k] = pfY[idx + baseY] - g_sa_pair.hedge_ratio * pfX[idx + baseX] - g_sa_pair.intercept;
            }
            SA_CUSUM_Init(g_sa_pair.cusum_state, cusum_spreads, cu_count);
        }
    }

    // Reset health counter — scan just ran ADF, no need for immediate health check
    g_sa_pair.bars_since_health = 0;

    // Mark active in top pairs (check against all active pairs for multi-pair)
    for(int i = 0; i < g_sa_top_count; i++)
    {
        g_sa_top_pairs[i].is_active = false;
        int mp = ArraySize(g_sa_pairs);
        for(int p = 0; p < mp; p++)
        {
            if(!g_sa_pairs[p].active) continue;
            if(g_sa_top_pairs[i].symbolX == g_sa_pairs[p].symbolX &&
               g_sa_top_pairs[i].symbolY == g_sa_pairs[p].symbolY)
            { g_sa_top_pairs[i].is_active = true; break; }
        }
    }

    // Fix 39: Set last_bar_time so the first SA_ManageTrade call doesn't see a
    // false "new bar" and push a duplicate of the last completed bar (already in
    // the pre-fill). Without this, the ring buffer contains [..., bar_N, bar_N]
    // instead of [..., bar_{N-1}, bar_N], biasing mean/variance toward the latest bar.
    // Fix 59: Add iTime=0 fallback for entry_tf.
    // Without this, if iTime(entry_tf, 0) returns 0, g_sa_pair.last_bar_time=0 causes a false
    // is_new_bar on the first tick, duplicating the last pre-filled bar in the z-score ring.
    g_sa_pair.last_bar_time = iTime(result.symbolY, entry_tf, 0);
    if(g_sa_pair.last_bar_time == 0)
        g_sa_pair.last_bar_time = iTime(result.symbolY, entry_tf, 1);
    if(g_sa_pair.last_bar_time == 0)
        g_sa_pair.last_bar_time = TimeCurrent();  // Fix 58 sentinel: prevent 0 from ever persisting

    // Seed ML prev_zscore AFTER pre-fill so dz on the first bar uses the new pair's z-score
    // (not the stale pre-rescan value, which would cause a cross-pair velocity spike)
    if(StatArb_MLMode != SA_ML_OFF && g_sa_ml_initialized)
    {
        int pidx = g_sa_current_idx;
        if(pidx >= 0 && pidx < ArraySize(g_sa_ml_pair_state))
            g_sa_ml_pair_state[pidx].prev_zscore = g_sa_pair.current_zscore;
    }

    string oos_info = "";
    if(StatArb_UseOOS) oos_info = " OOS_p=" + DoubleToString(result.oos_pvalue, 3);
    string kpss_info = "";
    if(StatArb_UseKPSS) kpss_info = " KPSS=" + DoubleToString(result.kpss_stat, 4);
    Print("StatArb: Activated pair ", result.symbolY, "/", result.symbolX,
          " Beta=", DoubleToString(result.hedge_ratio, 4),
          " ADF=", DoubleToString(result.adf_statistic, 2),
          " p=", DoubleToString(result.adf_pvalue, 3),
          " Corr=", DoubleToString(result.correlation, 3),
          " HL=", DoubleToString(result.half_life, 1),
          " ZWin=", zwin, oos_info, kpss_info);
}

void SA_ManageTrade(string symbolY)
{
    ENUM_TIMEFRAMES entry_tf, hedge_tf_unused;
    SA_ResolveTimeframes(entry_tf, hedge_tf_unused);

    // New bar detection
    datetime curBarTime = iTime(symbolY, entry_tf, 0);
    if(curBarTime <= 0) return; // Symbol history not yet synced
    bool is_new_bar = (curBarTime != g_sa_pair.last_bar_time);
    if(is_new_bar) g_sa_pair.last_bar_time = curBarTime;

    // In bar mode, only process on new bars
    if(StatArb_CalcMode == SA_CALC_BAR && !is_new_bar) return;

    if(!g_sa_pair.active) return;

    // Calculate current spread: S = Y - beta*X - alpha
    // Cointegration relationship holds across frequencies, so entry_tf prices are valid
    // and provide more responsive spread readings with composite timeframes.
    double priceY = 0, priceX = 0;
    if(StatArb_CalcMode == SA_CALC_TICK)
    {
        // Use current bid for most responsive reading
        priceY = SymbolInfoDouble(symbolY, SYMBOL_BID);
        priceX = SymbolInfoDouble(g_sa_pair.symbolX, SYMBOL_BID);
    }
    else
    {
        // Use completed bar [1] on entry_tf for responsiveness
        priceY = iClose(symbolY, entry_tf, 1);
        priceX = iClose(g_sa_pair.symbolX, entry_tf, 1);
    }

    // Fix 56: If prices unavailable (history not synced or symbol never quoted), still
    // increment health counter so health checks can fire. SA_HealthCheck fetches its own
    // historical data via SA_GetPrices, so it works even when current tick prices are zero.
    // Note: actual symbol halts return stale non-zero prices (iClose/BID cache the last
    // known value), so this path handles the rarer case of completely absent price data.
    if(priceY == 0 || priceX == 0)
    {
        if(is_new_bar && g_sa_pair.active && StatArb_HealthCheckBars > 0)
        {
            g_sa_pair.bars_since_health++;
            if(!SA_HealthCheck())
                g_sa_pair.last_scan_time = 0;
        }
        return;
    }

    double spread = priceY - g_sa_pair.hedge_ratio * priceX - g_sa_pair.intercept;

    // Kalman filter: update hedge ratio on new bars
    if(StatArb_HedgeAdapt == SA_ADAPT_KALMAN && is_new_bar)
    {
        SA_KalmanUpdate(g_sa_pair, priceY, priceX,
                        StatArb_KalmanQ, StatArb_KalmanR);
        // Recompute spread with updated Kalman beta/alpha
        spread = priceY - g_sa_pair.hedge_ratio * priceX - g_sa_pair.intercept;
    }

    double z = 0;

    // In tick mode: only push to ring buffer on new bars, but use current spread for Z
    if(StatArb_CalcMode == SA_CALC_TICK)
    {
        if(is_new_bar)
            z = SA_UpdateZScore(spread);  // Update ring buffer on new bar only
        else if(g_sa_pair.ring_count >= 2)
        {
            // Compute Z from existing rolling stats without pushing
            double cnt = (double)((g_sa_pair.ring_count < g_sa_pair.ring_size)
                         ? g_sa_pair.ring_count : g_sa_pair.ring_size);
            double mean = g_sa_pair.rolling_sum / cnt;
            // Sample variance (n-1) consistent with SA_UpdateZScore
            double var  = (g_sa_pair.rolling_sum_sq - cnt * mean * mean) / (cnt - 1.0);
            // Fix 28b: Same FP drift fix as SA_UpdateZScore — recompute from ring
            // instead of clamping to 0 (which produces Z=0 → spurious exits).
            // This tick-mode path is hotter (runs every tick), making the impact worse.
            if(var < 0)
            {
                g_sa_pair.rolling_sum = 0;
                g_sa_pair.rolling_sum_sq = 0;
                int ring_cnt = (int)MathMin(g_sa_pair.ring_count, g_sa_pair.ring_size);
                for(int ii = 0; ii < ring_cnt; ii++)
                {
                    g_sa_pair.rolling_sum += g_sa_pair.spread_ring[ii];
                    g_sa_pair.rolling_sum_sq += g_sa_pair.spread_ring[ii] * g_sa_pair.spread_ring[ii];
                }
                mean = g_sa_pair.rolling_sum / cnt;
                var = (g_sa_pair.rolling_sum_sq - cnt * mean * mean) / (cnt - 1.0);
                if(var < 0) var = 0;
                g_sa_pair.ring_updates = 0;
            }
            double std  = MathSqrt(var);
            z = (std > 1e-12) ? (spread - mean) / std : 0.0;
            g_sa_pair.current_zscore = z;
            g_sa_pair.current_spread = spread;
        }
        else return; // Not enough data yet
    }
    else
        z = SA_UpdateZScore(spread);

    // CUSUM structural break detection (every new bar)
    if(StatArb_UseCUSUM && is_new_bar && g_sa_pair.cusum_state.initialized)
    {
        if(SA_CUSUM_Update(g_sa_pair.cusum_state, spread))
        {
            Print("StatArb: CUSUM BREAK detected! CUSUM=",
                  DoubleToString(g_sa_pair.cusum_state.cumsum, 2),
                  " (n=", g_sa_pair.cusum_state.n_obs,
                  "). Triggering full rescan.");
            g_sa_pair.last_scan_time = 0;  // Force rescan (same as health check failure)
        }
    }

    // Periodic cointegration health check (on entry_tf new bars)
    // SA_HealthCheck fetches its own historical data — self-contained, no ring buffer
    if(is_new_bar && g_sa_pair.active && StatArb_HealthCheckBars > 0)
    {
        g_sa_pair.bars_since_health++;
        if(!SA_HealthCheck())
            g_sa_pair.last_scan_time = 0;  // Force immediate rescan
    }

    // Time-based exit counter
    if(is_new_bar && g_sa_pair.position_dir != 0)
        g_sa_pair.bars_in_trade++;

    // Check rescan timer: when flat in AUTO mode, or anytime in ALERT mode
    // (alert mode has no real positions, so stale parameters are a bigger risk than rescan cost)
    bool rescan_allowed = (g_sa_pair.position_dir == 0) ||
                          (StatArb_TradeMode == SA_MODE_ALERT);
    // Fix 55: If health check failed while in AUTO-mode position, the rescan is blocked
    // (can't switch pairs with real positions open). Clear the health_forced flag and log
    // a warning so it doesn't persist indefinitely. The next health check will re-evaluate.
    if(!rescan_allowed && g_sa_pair.last_scan_time == 0 && g_sa_pair.position_dir != 0)
    {
        Print("StatArb: WARNING — Health check failed but position open in AUTO mode. ",
              "ADF p=", DoubleToString(g_sa_pair.last_health_pvalue, 3),
              ". Rescan deferred until position closes.");
        g_sa_pair.last_scan_time = TimeCurrent();  // Clear health_forced to prevent repeated attempts
    }
    // Fix 24: RescanInterval=0 means "every bar" (as documented in input comment).
    // RescanInterval<0 means "never rescan". Previously >0 guard blocked 0=every-bar.
    // Fix 27: Gate RescanInterval=0 with is_new_bar — without this, TICK calc mode
    // would fire a full pair scan on every tick (potentially dozens per second).
    // Fix 52: Health check failure (last_scan_time=0) overrides RescanInterval<0.
    // If cointegration degrades, we MUST rescan regardless of interval setting.
    // last_scan_time==0 is set exclusively by health check failure (post-activation).
    // Fix 53: Clause 2 must exclude health_forced — when last_scan_time==0 (epoch),
    // TimeCurrent()-0 is always huge, bypassing the interval gate on every tick.
    // Health-forced rescans are handled by Clause 3 (gated with is_new_bar).
    bool health_forced = (g_sa_pair.last_scan_time == 0);
    if(rescan_allowed &&
       (StatArb_RescanInterval >= 0 || health_forced) &&
       ((StatArb_RescanInterval == 0 && is_new_bar) ||
        (StatArb_RescanInterval > 0 && !health_forced && TimeCurrent() - g_sa_pair.last_scan_time > StatArb_RescanInterval * 60) ||
        (health_forced && is_new_bar)))
    {
        SA_PairResult new_result;
        if(SA_ScanForBestPair(symbolY, new_result))
        {
            if(new_result.symbolX != g_sa_pair.symbolX || new_result.symbolY != g_sa_pair.symbolY)
            {
                // Fix 22: In alert mode, a virtual position on the OLD pair must be voided
                // before switching — otherwise position_dir leaks to the new pair and
                // exit/stop signals fire on the wrong spread.
                if(g_sa_pair.position_dir != 0 && StatArb_TradeMode == SA_MODE_ALERT)
                {
                    // Fix 35: Use Print instead of SA_SendAlert for the CANCEL notification.
                    // SA_SendAlert consumes the per-bar alert cooldown, which suppresses the
                    // subsequent ENTRY alert on the new pair (both happen on the same bar).
                    Print("StatArb: SA CANCEL: Pair switched ", g_sa_pair.symbolY, "/", g_sa_pair.symbolX,
                          " -> ", new_result.symbolY, "/", new_result.symbolX,
                          " | Prior virtual position voided (dir=", g_sa_pair.position_dir, ")");
                    if(StatArb_MLMode != SA_ML_OFF) SAML_ResetPairML();
                    g_sa_pair.position_dir = 0;
                    g_sa_pair.entry_time = 0;
                    g_sa_pair.entry_zscore = 0;
                }
                SA_ActivatePair(new_result);
            }
            else
            {
                // Same pair: full re-activate to refresh parameters and z-score ring.
                // Fix 43: Defer when alert mode has a virtual position open —
                // exit/stop must use consistent spread definition as entry.
                if(g_sa_pair.position_dir != 0 && StatArb_TradeMode == SA_MODE_ALERT)
                {
                    Print("StatArb: Rescan found updated params for same pair, but virtual position active -- deferring");
                }
                else
                {
                    SA_ActivatePair(new_result);
                }
            }
        }
        else
        {
            Print("StatArb: Rescan found no cointegrated pairs for ", symbolY);
            // Fix 61: Explicit CRITICAL warning when health-forced rescan fails to find a replacement.
            // The pair continues trading with degraded cointegration — user needs visibility.
            if(health_forced)
            {
                // Fix 64: Use last_health_pvalue (always current) instead of adf_pvalue
                // (which may be stale in ALERT mode due to Fix 54's metadata guard).
                Print("StatArb: CRITICAL — Cointegration degraded (ADF p=",
                      DoubleToString(g_sa_pair.last_health_pvalue, 3),
                      ") and no replacement pair found. Pair ", g_sa_pair.symbolY,
                      "/", g_sa_pair.symbolX, " continues with existing parameters.");
                if(g_sa_pair.position_dir != 0)
                    Print("StatArb: WARNING — Position still open on degraded pair!");
            }
        }
        g_sa_pair.last_scan_time = TimeCurrent();
        // Fix 26: Rescan may have changed hedge_ratio/intercept/symbolX or fully
        // switched pairs. The local `z` was computed with old parameters and is now
        // stale — re-sync from the ring buffer's current z-score (updated during refill).
        z = g_sa_pair.current_zscore;
    }

    // Refresh position tickets (detect external closes) — AUTO mode only
    // In ALERT mode, position_dir tracks a virtual position with no real tickets
    if(g_sa_pair.position_dir != 0 && StatArb_TradeMode == SA_MODE_AUTO)
    {
        SA_RefreshTickets();
        if(g_sa_pair.ticketY == 0 || g_sa_pair.ticketX == 0)
        {
            bool orphan_closed = true;
            if(g_sa_pair.ticketY > 0) orphan_closed &= sa_trade.PositionClose(g_sa_pair.ticketY);
            if(g_sa_pair.ticketX > 0) orphan_closed &= sa_trade.PositionClose(g_sa_pair.ticketX);

            if(orphan_closed)
            {
                Print("StatArb: External close detected, closed orphaned leg");
                if(StatArb_MLMode != SA_ML_OFF) SAML_ResetPairML();
                g_sa_pair.position_dir = 0;
                g_sa_pair.ticketY = 0;
                g_sa_pair.ticketX = 0;
                // Fix 41: Set cooldown so Fix 33 blocks immediate re-entry after orphan
                // recovery. Without this, a partial-close → orphan-close sequence has no
                // cooldown, and z (still past EntryZ) triggers instant re-entry churn.
                g_sa_pair.last_close_time = TimeCurrent();
            }
            else
                Print("StatArb: Orphan close retry failed — will retry next tick");
        }
    }

    // Update top pairs Z-score for panel
    for(int i = 0; i < g_sa_top_count; i++)
    {
        if(g_sa_top_pairs[i].is_active)
        {
            g_sa_top_pairs[i].current_zscore = z;
            g_sa_top_pairs[i].position_dir = g_sa_pair.position_dir;
        }
    }

    // Not enough data in ring buffer yet — still update ML velocity tracking
    if(g_sa_pair.ring_count < 5)
    {
        if(is_new_bar && StatArb_MLMode != SA_ML_OFF) SAML_OnNewBar(g_sa_pair);
        return;
    }

    // ── TRADING LOGIC ──
    // Fix 33: Cooldown after close — require at least 1 new bar before re-entry.
    // Without this, a stop-loss exit is immediately followed by re-entry (z still
    // exceeds EntryZ), creating a destructive churn cycle burning transaction costs.
    // Note: last_close_time is server time (TimeCurrent), curBarTime is bar open time.
    // Using < ensures the next bar (whose open time >= close time) is allowed through.
    if(g_sa_pair.position_dir == 0 && g_sa_pair.last_close_time > 0 &&
       curBarTime < g_sa_pair.last_close_time)
    {
        // Still on the same bar as the last close — wait for next bar
        if(is_new_bar && StatArb_MLMode != SA_ML_OFF) SAML_OnNewBar(g_sa_pair);
        return;
    }

    if(g_sa_pair.position_dir == 0)
    {
        // Regime gate: skip entries in trending regime
        // Full RL mode bypasses — the NN has regime_score as feature #10 and learns when to trade
        if(StatArb_RegimeMode != SA_REGIME_OFF && StatArb_MLMode != SA_ML_FULL_RL)
        {
            if(!SA_IsRegimeMeanReverting(g_sa_pair, (int)StatArb_RegimeMode,
                                          StatArb_RegimeWindow, StatArb_RegimeThreshold))
            {
                // Trending regime — no entry, but still update ML velocity tracking
                if(is_new_bar && StatArb_MLMode != SA_ML_OFF) SAML_OnNewBar(g_sa_pair);
                return;
            }
        }

        // FLAT: check for entry (skip if Z already past stop — would immediately stop out)
        int bm = (int)StatArb_BoundaryMode;

        // Full RL mode: ML replaces Z-score entry entirely
        // Gate to new bars only — z-score ring only updates on new bars, so running
        // Forward() on every tick in TICK mode wastes CPU and floods counterfactuals
        if(StatArb_MLMode == SA_ML_FULL_RL)
        {
            if(!is_new_bar) {} // TICK mode: skip RL decision on non-bar ticks (no new data)
            else
            {
                int rl_action = SAML_FullRLDecision(g_sa_pair);
                if(rl_action == 1 || rl_action == 2)
            {
                // StopZ safety: prevent RL entries that would immediately stop out
                bool z_past_stop = (rl_action == 1 && z <= -StatArb_StopZ) ||
                                   (rl_action == 2 && z >= StatArb_StopZ);
                if(z_past_stop)
                {
                    SAML_ResetPairML();
                }
                else
                {
                    int dir = (rl_action == 1) ? +1 : -1;
                    if(StatArb_TradeMode == SA_MODE_AUTO)
                    {
                        if(!SA_OpenPairTrade(dir))
                            SAML_ResetPairML();  // Clear dangling has_entry on open failure
                    }
                    else
                    {
                        SA_SendAlert("SA ML-RL ENTRY: " + SA_DirectionStr(dir)
                            + " | Z=" + DoubleToString(z, 2)
                            + " | Beta=" + DoubleToString(g_sa_pair.hedge_ratio, 4));
                        g_sa_pair.position_dir = dir;
                        g_sa_pair.entry_zscore = z;
                        g_sa_pair.entry_time = TimeCurrent();
                        g_sa_pair.bars_in_trade = 0;
                    }
                }
            }
            } // end is_new_bar else block
        }
        // Classic or ML-gated entry (modes OFF, ENTRY_GATE, ENTRY_EXIT)
        else if(SA_ShouldEnterLong(z, spread, g_sa_pair, bm))
        {
            // ML gate: modes 1-2 filter classic Z-score entries
            bool ml_allow = true;
            if(StatArb_MLMode == SA_ML_ENTRY_GATE || StatArb_MLMode == SA_ML_ENTRY_EXIT)
                ml_allow = SAML_GateEntry(g_sa_pair, +1);

            if(ml_allow)
            {
                // Spread below mean: Y cheap relative to X → Long Y, Short X
                if(StatArb_TradeMode == SA_MODE_AUTO)
                {
                    if(!SA_OpenPairTrade(+1))
                        SAML_ResetPairML();  // Clear dangling has_entry on open failure
                }
                else
                {
                    SA_SendAlert("SA ENTRY: " + SA_DirectionStr(+1)
                        + " | Z=" + DoubleToString(z, 2)
                        + " | Beta=" + DoubleToString(g_sa_pair.hedge_ratio, 4)
                        + " | Corr=" + DoubleToString(g_sa_pair.correlation, 3)
                        + " | HL=" + DoubleToString(g_sa_pair.half_life, 1));
                    g_sa_pair.position_dir = +1;
                    g_sa_pair.entry_zscore = z;
                    g_sa_pair.entry_time = TimeCurrent();
                    g_sa_pair.bars_in_trade = 0;
                }
            }
        }
        else if(SA_ShouldEnterShort(z, spread, g_sa_pair, bm))
        {
            // ML gate: modes 1-2 filter classic Z-score entries
            bool ml_allow = true;
            if(StatArb_MLMode == SA_ML_ENTRY_GATE || StatArb_MLMode == SA_ML_ENTRY_EXIT)
                ml_allow = SAML_GateEntry(g_sa_pair, -1);

            if(ml_allow)
            {
                // Spread above mean: Y expensive relative to X → Short Y, Long X
                if(StatArb_TradeMode == SA_MODE_AUTO)
                {
                    if(!SA_OpenPairTrade(-1))
                        SAML_ResetPairML();  // Clear dangling has_entry on open failure
                }
                else
                {
                    SA_SendAlert("SA ENTRY: " + SA_DirectionStr(-1)
                        + " | Z=" + DoubleToString(z, 2)
                        + " | Beta=" + DoubleToString(g_sa_pair.hedge_ratio, 4)
                        + " | Corr=" + DoubleToString(g_sa_pair.correlation, 3)
                        + " | HL=" + DoubleToString(g_sa_pair.half_life, 1));
                    g_sa_pair.position_dir = -1;
                    g_sa_pair.entry_zscore = z;
                    g_sa_pair.entry_time = TimeCurrent();
                    g_sa_pair.bars_in_trade = 0;
                }
            }
        }
    }
    else
    {
        // IN POSITION: check exit / stop
        int bm = (int)StatArb_BoundaryMode;
        bool should_stop = false;
        bool should_exit = false;

        if(g_sa_pair.position_dir > 0)
        {
            should_exit = SA_ShouldExitLong(z, spread, g_sa_pair, bm);
            should_stop = SA_ShouldStopLong(z, spread, g_sa_pair, bm);
        }
        else
        {
            should_exit = SA_ShouldExitShort(z, spread, g_sa_pair, bm);
            should_stop = SA_ShouldStopShort(z, spread, g_sa_pair, bm);
        }

        // ML exit override (modes 2-3): replace classic Z-score exit with ML decision
        // StopZ is NEVER overridden — always respected as hard safety
        // During warmup, SAML_ShouldExitML defers to classic_exit (preserves z-score exits)
        if(!should_stop && (StatArb_MLMode == SA_ML_ENTRY_EXIT || StatArb_MLMode == SA_ML_FULL_RL))
        {
            should_exit = SAML_ShouldExitML(g_sa_pair, should_exit);
        }

        // Time-based exit: force close if trade hasn't reverted within N * half_life bars
        bool should_time_exit = false;
        if(StatArb_TimeExitHL > 0 && g_sa_pair.half_life > 0 && g_sa_pair.position_dir != 0)
        {
            int max_bars = (int)(StatArb_TimeExitHL * g_sa_pair.half_life);
            if(max_bars < 1) max_bars = 1;
            if(g_sa_pair.bars_in_trade >= max_bars)
                should_time_exit = true;
        }

        if(should_time_exit && !should_stop)
        {
            string reason = "TIME EXIT (" + IntegerToString(g_sa_pair.bars_in_trade)
                          + " bars >= " + DoubleToString(StatArb_TimeExitHL, 1)
                          + " * HL " + DoubleToString(g_sa_pair.half_life, 1) + ")";
            if(StatArb_TradeMode == SA_MODE_AUTO)
                SA_ClosePairTrade(reason);
            else
            {
                SA_SendAlert("SA TIME EXIT: " + SA_DirectionStr(g_sa_pair.position_dir) + " | " + reason);
                g_sa_pair.position_dir = 0;
                g_sa_pair.last_close_time = TimeCurrent();
                if(StatArb_MLMode != SA_ML_OFF) SAML_ResetPairML();
            }
        }
        else if(should_stop)
        {
            string reason = "STOP (Z=" + DoubleToString(z, 2) + " exceeded ±" + DoubleToString(StatArb_StopZ, 1) + ")";
            if(StatArb_TradeMode == SA_MODE_AUTO)
                SA_ClosePairTrade(reason);  // handles position_dir reset internally (conditional on both legs closing)
            else
            {
                SA_SendAlert("SA STOP: " + SA_DirectionStr(g_sa_pair.position_dir) + " | " + reason);
                g_sa_pair.position_dir = 0; // Reset in alert mode to prevent spam
                // Fix 40: Set last_close_time so Fix 33 cooldown also works for virtual
                // positions. Without this, a STOP is immediately followed by a re-ENTRY
                // alert on the same bar (z still exceeds EntryZ), creating alert churn.
                g_sa_pair.last_close_time = TimeCurrent();
                if(StatArb_MLMode != SA_ML_OFF) SAML_ResetPairML();
            }
        }
        else if(should_exit)
        {
            // Profit Gate mode: block z-score exit until PnL >= threshold (AUTO only).
            // StopZ is never gated (handled above). Only z-score reversion exits are gated.
            bool is_gate_mode = (StatArb_MinProfitMode == SA_PROFIT_GATE_USD || StatArb_MinProfitMode == SA_PROFIT_GATE_PCT);
            if(is_gate_mode && StatArb_TradeMode == SA_MODE_AUTO && StatArb_MinProfitValue > 0)
            {
                double pair_pnl = 0;
                if(g_sa_pair.ticketY > 0 && PositionSelectByTicket(g_sa_pair.ticketY))
                    pair_pnl += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);
                if(g_sa_pair.ticketX > 0 && PositionSelectByTicket(g_sa_pair.ticketX))
                    pair_pnl += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);

                double gate_threshold = StatArb_MinProfitValue;
                if(StatArb_MinProfitMode == SA_PROFIT_GATE_PCT)
                    gate_threshold = AccountInfoDouble(ACCOUNT_BALANCE) * StatArb_MinProfitValue / 100.0;

                if(pair_pnl < gate_threshold)
                    should_exit = false;  // Block exit — profit not reached yet
            }
        }

        if(should_exit)
        {
            string reason = "EXIT (Z=" + DoubleToString(z, 2) + " reverted)";
            if(StatArb_TradeMode == SA_MODE_AUTO)
                SA_ClosePairTrade(reason);  // handles position_dir reset internally (conditional on both legs closing)
            else
            {
                SA_SendAlert("SA EXIT: " + SA_DirectionStr(g_sa_pair.position_dir) + " | " + reason);
                g_sa_pair.position_dir = 0;
                // Fix 40: Same cooldown as STOP path above — prevent immediate re-entry alert.
                g_sa_pair.last_close_time = TimeCurrent();
                if(StatArb_MLMode != SA_ML_OFF) SAML_ResetPairML();
            }
        }

        // Dollar/percent-based TP & SL exits independent of z-score (AUTO only).
        // Works in both TPSL and Gate modes. Only fires if z-score exits above didn't already close.
        bool is_tpsl_mode = (StatArb_MinProfitMode == SA_PROFIT_TPSL_USD || StatArb_MinProfitMode == SA_PROFIT_TPSL_PCT);
        bool is_gate_mode = (StatArb_MinProfitMode == SA_PROFIT_GATE_USD || StatArb_MinProfitMode == SA_PROFIT_GATE_PCT);
        if((is_tpsl_mode || is_gate_mode) && g_sa_pair.position_dir != 0 && StatArb_TradeMode == SA_MODE_AUTO)
        {
            double pair_pnl = 0;
            if(g_sa_pair.ticketY > 0 && PositionSelectByTicket(g_sa_pair.ticketY))
                pair_pnl += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);
            if(g_sa_pair.ticketX > 0 && PositionSelectByTicket(g_sa_pair.ticketX))
                pair_pnl += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);

            bool is_pct = (StatArb_MinProfitMode == SA_PROFIT_TPSL_PCT || StatArb_MinProfitMode == SA_PROFIT_GATE_PCT);
            double balance = is_pct ? AccountInfoDouble(ACCOUNT_BALANCE) : 0;

            // Take profit (TPSL mode only — Gate mode handles TP via z-score gating above)
            if(is_tpsl_mode && StatArb_MinProfitValue > 0)
            {
                double tp_threshold = is_pct ? (balance * StatArb_MinProfitValue / 100.0) : StatArb_MinProfitValue;
                if(pair_pnl >= tp_threshold)
                {
                    string reason = "TAKE PROFIT (PnL=" + DoubleToString(pair_pnl, 2) + " >= " + DoubleToString(tp_threshold, 2) + ")";
                    SA_ClosePairTrade(reason);
                }
            }

            // Stop loss (both TPSL and Gate modes — safety net against drawdown)
            if(g_sa_pair.position_dir != 0 && StatArb_MinStopLossValue > 0)
            {
                double sl_threshold = is_pct ? (balance * StatArb_MinStopLossValue / 100.0) : StatArb_MinStopLossValue;
                if(pair_pnl <= -sl_threshold)
                {
                    string reason = "DOLLAR SL (PnL=" + DoubleToString(pair_pnl, 2) + " <= -" + DoubleToString(sl_threshold, 2) + ")";
                    SA_ClosePairTrade(reason);
                }
            }
        }
    }

    // ML hook: update z-velocity, MFE/MAE on new bar — placed AFTER entry/exit logic
    // so that prev_zscore is still the old value when SAML_BuildFeatures computes dz
    if(is_new_bar && StatArb_MLMode != SA_ML_OFF)
        SAML_OnNewBar(g_sa_pair);
}

//+------------------------------------------------------------------+
//| SECTION 14: PANEL                                                  |
//+------------------------------------------------------------------+

void SA_PanelCreateLabel(string name, int x, int y, string text,
                         color clr = clrWhite, int font_size = 9,
                         string font = "Consolas")
{
    string obj_name = SA_PANEL_PREFIX + name;
    if(ObjectFind(ChartID(), obj_name) < 0)
        ObjectCreate(ChartID(), obj_name, OBJ_LABEL, 0, 0, 0);

    ObjectSetInteger(ChartID(), obj_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_YDISTANCE, y);
    ObjectSetString(ChartID(), obj_name, OBJPROP_TEXT, text);
    ObjectSetString(ChartID(), obj_name, OBJPROP_FONT, font);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_FONTSIZE, font_size);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_COLOR, clr);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_HIDDEN, true);
}

void SA_PanelCreateRect(string name, int x, int y, int w, int h, color bg_clr)
{
    string obj_name = SA_PANEL_PREFIX + name;
    if(ObjectFind(ChartID(), obj_name) < 0)
        ObjectCreate(ChartID(), obj_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);

    ObjectSetInteger(ChartID(), obj_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_XDISTANCE, x);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_YDISTANCE, y);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_XSIZE, w);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_YSIZE, h);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_BGCOLOR, bg_clr);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_BORDER_COLOR, clrDimGray);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_SELECTABLE, false);
    ObjectSetInteger(ChartID(), obj_name, OBJPROP_HIDDEN, true);
}

color SA_ZScoreColor(double z)
{
    double az = MathAbs(z);
    if(az >= StatArb_StopZ) return clrRed;
    if(az >= StatArb_EntryZ) return clrLime;
    if(az >= StatArb_EntryZ * 0.75) return clrYellow;
    return clrGray;
}

string SA_TFString()
{
    int val = (int)StatArb_Timeframe;
    if(val <= 20) return EnumToString(SA_EnumToTF(val));
    // Composite
    ENUM_TIMEFRAMES e, h;
    SA_ResolveTimeframes(e, h);
    return StringSubstr(EnumToString(e), 7) + "+" + StringSubstr(EnumToString(h), 7);
}

void SA_PanelCreate()
{
    if(!StatArb_ShowPanel) return;
    int px = StatArb_PanelX;
    int py = StatArb_PanelY;
    int pw = 340;
    int ph = 420 + g_sa_panel_topn * 16;

    SA_PanelCreateRect("bg", px, py, pw, ph, C'25,25,35');
    SA_PanelCreateLabel("title", px + 10, py + 5, "STAT ARB", clrCyan, 11, "Consolas Bold");
}

void SA_PanelUpdate()
{
    if(!StatArb_ShowPanel) return;
    int px = StatArb_PanelX;
    int py = StatArb_PanelY;
    int row = 0;

    // Header
    string mode_str = (StatArb_TradeMode == SA_MODE_AUTO) ? "AUTO" : "ALERT";
    string hedge_str = (StatArb_HedgeMode == SA_HEDGE_OLS) ? "OLS" : "TLS";
    string calc_str = (StatArb_CalcMode == SA_CALC_BAR) ? "Bar" : "Tick";
    string header_txt = mode_str + " | " + hedge_str + " | " + SA_TFString() + " | " + calc_str;
    if(StatArb_MaxPairs > 1)
    {
        int act = 0;
        int mp = ArraySize(g_sa_pairs);
        for(int p = 0; p < mp; p++)
            if(g_sa_pairs[p].active) act++;
        header_txt += " | P:" + IntegerToString(act) + "/" + IntegerToString(mp);
    }
    SA_PanelCreateLabel("header", px + 10, py + 22, header_txt, clrSilver, 8);
    row = 40;

    // Separator
    SA_PanelCreateLabel("sep1", px + 10, py + row,
        "----------------------------------------", clrDimGray, 8);
    row += 14;

    if(g_sa_pair.active)
    {
        // Active pair info
        SA_PanelCreateLabel("pair_label", px + 10, py + row, "ACTIVE PAIR", clrCyan, 9);
        row += 16;
        SA_PanelCreateLabel("pair_syms", px + 10, py + row,
            g_sa_pair.symbolY + " / " + g_sa_pair.symbolX, clrWhite, 10);
        row += 16;
        SA_PanelCreateLabel("pair_corr", px + 10, py + row,
            "Corr: " + DoubleToString(g_sa_pair.correlation, 3)
            + "  ADF: " + DoubleToString(g_sa_pair.adf_statistic, 2)
            + "  p=" + DoubleToString(g_sa_pair.adf_pvalue, 3),
            clrSilver, 8);
        row += 14;
        SA_PanelCreateLabel("pair_beta", px + 10, py + row,
            "Beta: " + DoubleToString(g_sa_pair.hedge_ratio, 4)
            + "  Alpha: " + DoubleToString(g_sa_pair.intercept, 5),
            clrSilver, 8);
        row += 14;
        SA_PanelCreateLabel("pair_hl", px + 10, py + row,
            "Half-Life: " + DoubleToString(g_sa_pair.half_life, 1) + " bars",
            clrSilver, 8);
        row += 14;
        // Hurst exponent (if enabled)
        if(StatArb_UseHurst)
        {
            color h_clr = (g_sa_pair.hurst < 0.45) ? clrLime : (g_sa_pair.hurst < 0.5) ? clrYellow : clrOrangeRed;
            SA_PanelCreateLabel("pair_hurst", px + 10, py + row,
                "Hurst: " + DoubleToString(g_sa_pair.hurst, 3), h_clr, 8);
            row += 14;
        }
        else
            SA_PanelCreateLabel("pair_hurst", px + 10, py - 100, "", clrBlack, 8);
        // Test type (Johansen vs ADF)
        if(StatArb_CointTest == SA_COINT_JOHANSEN)
        {
            SA_PanelCreateLabel("pair_test", px + 10, py + row,
                "Test: Johansen", clrCyan, 8);
            row += 14;
        }
        else
            SA_PanelCreateLabel("pair_test", px + 10, py - 100, "", clrBlack, 8);
        // Kalman filter status
        if(StatArb_HedgeAdapt == SA_ADAPT_KALMAN && g_sa_pair.kalman_initialized)
        {
            SA_PanelCreateLabel("pair_kalman", px + 10, py + row,
                "KALMAN  B=" + DoubleToString(g_sa_pair.kalman_beta, 4)
                + "  A=" + DoubleToString(g_sa_pair.kalman_alpha, 5), clrAqua, 8);
            row += 14;
        }
        else
            SA_PanelCreateLabel("pair_kalman", px + 10, py - 100, "", clrBlack, 8);
        row += 4;

        // Z-Score section
        SA_PanelCreateLabel("zsep", px + 10, py + row,
            "----------------------------------------", clrDimGray, 8);
        row += 14;

        double z = g_sa_pair.current_zscore;
        color z_clr = SA_ZScoreColor(z);

        // Z-score bar visualization
        string z_bar = "";
        int total_chars = 20;
        int center = total_chars / 2;
        int fill_pos = center + (int)MathRound((double)z / StatArb_StopZ * center);
        fill_pos = (int)MathMax(0, (int)MathMin(total_chars - 1, fill_pos));
        for(int i = 0; i < total_chars; i++)
        {
            if(i == center) z_bar += "|";
            else if((z < 0 && i >= fill_pos && i < center) || (z > 0 && i > center && i <= fill_pos))
                z_bar += "#";
            else z_bar += ".";
        }

        SA_PanelCreateLabel("zscore_val", px + 10, py + row,
            "Z-Score: " + DoubleToString(z, 2) + "  [" + z_bar + "]", z_clr, 9);
        row += 16;
        if(StatArb_BoundaryMode == SA_BOUND_OU && g_sa_pair.ou_theta > 0)
        {
            SA_PanelCreateLabel("zscore_thresh", px + 10, py + row,
                "OU: E=" + DoubleToString(g_sa_pair.ou_entry_short, 4)
                + " X=" + DoubleToString(g_sa_pair.ou_exit_short, 4)
                + " S=" + DoubleToString(g_sa_pair.ou_stop_short, 4),
                clrAqua, 8);
        }
        else
        {
            SA_PanelCreateLabel("zscore_thresh", px + 10, py + row,
                "Entry: +" + DoubleToString(StatArb_EntryZ, 1)
                + " | Exit: +" + DoubleToString(StatArb_ExitZ, 1)
                + " | Stop: +" + DoubleToString(StatArb_StopZ, 1),
                clrDarkGray, 8);
        }
        row += 14;
        SA_PanelCreateLabel("zscore_spread", px + 10, py + row,
            "Spread: " + DoubleToString(g_sa_pair.current_spread, 5)
            + "  Ring: " + IntegerToString(g_sa_pair.ring_count) + "/" + IntegerToString(g_sa_pair.ring_size),
            clrDarkGray, 8);
        row += 14;
        // Regime indicator
        if(StatArb_RegimeMode != SA_REGIME_OFF)
        {
            string reg_type = (StatArb_RegimeMode == SA_REGIME_VR) ? "VR" : "H";
            string reg_state = g_sa_pair.regime_trending ? "TREND" : "MR";
            color reg_clr = g_sa_pair.regime_trending ? clrOrangeRed : clrLime;
            SA_PanelCreateLabel("regime", px + 10, py + row,
                "Regime: " + reg_type + "=" + DoubleToString(g_sa_pair.regime_score, 2)
                + " [" + reg_state + "]", reg_clr, 8);
            row += 14;
        }
        else
            SA_PanelCreateLabel("regime", px + 10, py - 100, "", clrBlack, 8);
        row += 4;

        // Position section
        SA_PanelCreateLabel("psep", px + 10, py + row,
            "----------------------------------------", clrDimGray, 8);
        row += 14;

        if(g_sa_pair.position_dir != 0)
        {
            color pos_clr = (g_sa_pair.position_dir > 0) ? clrLime : clrOrangeRed;
            SA_PanelCreateLabel("pos_dir", px + 10, py + row,
                "POSITION: " + SA_DirectionStr(g_sa_pair.position_dir), pos_clr, 9);
            row += 16;

            int dur_sec = (int)(TimeCurrent() - g_sa_pair.entry_time);
            int dur_h = dur_sec / 3600;
            int dur_m = (dur_sec % 3600) / 60;

            SA_PanelCreateLabel("pos_entry", px + 10, py + row,
                "Entry Z: " + DoubleToString(g_sa_pair.entry_zscore, 2)
                + "  |  " + IntegerToString(dur_h) + "h " + IntegerToString(dur_m) + "m ago",
                clrSilver, 8);
            row += 14;
            // Bars in trade + time exit progress
            if(StatArb_TimeExitHL > 0 && g_sa_pair.half_life > 0)
            {
                int max_bars = (int)(StatArb_TimeExitHL * g_sa_pair.half_life);
                double pct = (max_bars > 0) ? 100.0 * g_sa_pair.bars_in_trade / max_bars : 0;
                color te_clr = (pct > 80) ? clrOrangeRed : (pct > 50) ? clrYellow : clrSilver;
                SA_PanelCreateLabel("pos_bars", px + 10, py + row,
                    "Bars: " + IntegerToString(g_sa_pair.bars_in_trade)
                    + " / " + IntegerToString(max_bars)
                    + " (" + DoubleToString(pct, 0) + "% time exit)",
                    te_clr, 8);
                row += 14;
            }
            else
                SA_PanelCreateLabel("pos_bars", px + 10, py - 100, "", clrBlack, 8);

            string dirY = (g_sa_pair.position_dir > 0) ? "BUY " : "SELL";
            int x_d = (g_sa_pair.hedge_ratio >= 0) ? -g_sa_pair.position_dir : g_sa_pair.position_dir;
            string dirX = (x_d > 0) ? "BUY " : "SELL";

            if(StatArb_TradeMode == SA_MODE_ALERT)
            {
                // Alert mode: virtual position, no real tickets — show signal info instead of PnL
                SA_PanelCreateLabel("pos_legY", px + 10, py + row,
                    "Y: " + g_sa_pair.symbolY + " " + dirY + "  (virtual)", clrSilver, 8);
                row += 14;
                SA_PanelCreateLabel("pos_legX", px + 10, py + row,
                    "X: " + g_sa_pair.symbolX + " " + dirX + "  (virtual)", clrSilver, 8);
                row += 14;
                SA_PanelCreateLabel("pos_total", px + 10, py + row,
                    "ALERT MODE — no real positions", clrYellow, 9);
            }
            else
            {
                // Auto mode: real positions with PnL
                // Fix 36: Include swap and commission in panel PnL display
                double pnlY = 0, pnlX = 0;
                if(g_sa_pair.ticketY > 0 && PositionSelectByTicket(g_sa_pair.ticketY))
                    pnlY = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);
                if(g_sa_pair.ticketX > 0 && PositionSelectByTicket(g_sa_pair.ticketX))
                    pnlX = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);

                SA_PanelCreateLabel("pos_legY", px + 10, py + row,
                    "Y: " + g_sa_pair.symbolY + " " + dirY
                    + "  $" + DoubleToString(pnlY, 2),
                    (pnlY >= 0) ? clrLime : clrOrangeRed, 8);
                row += 14;
                SA_PanelCreateLabel("pos_legX", px + 10, py + row,
                    "X: " + g_sa_pair.symbolX + " " + dirX
                    + "  $" + DoubleToString(pnlX, 2),
                    (pnlX >= 0) ? clrLime : clrOrangeRed, 8);
                row += 14;

                double total_pnl = pnlY + pnlX;
                SA_PanelCreateLabel("pos_total", px + 10, py + row,
                    "Total PnL: $" + DoubleToString(total_pnl, 2),
                    (total_pnl >= 0) ? clrLime : clrOrangeRed, 9);
            }
            row += 18;
        }
        else
        {
            SA_PanelCreateLabel("pos_dir", px + 10, py + row,
                "FLAT - Waiting for Z > +" + DoubleToString(StatArb_EntryZ, 1), clrGray, 9);
            row += 18;
            // Clear position labels
            SA_PanelCreateLabel("pos_entry", px + 10, py - 100, "", clrBlack, 8);
            // Fix 67: Hide pos_bars when flat — without this, stale time exit progress
            // from the previous position lingers on screen after position closes.
            SA_PanelCreateLabel("pos_bars", px + 10, py - 100, "", clrBlack, 8);
            SA_PanelCreateLabel("pos_legY", px + 10, py - 100, "", clrBlack, 8);
            SA_PanelCreateLabel("pos_legX", px + 10, py - 100, "", clrBlack, 8);
            SA_PanelCreateLabel("pos_total", px + 10, py - 100, "", clrBlack, 8);
        }
    }
    else
    {
        SA_PanelCreateLabel("pair_label", px + 10, py + row, "SCANNING...", clrYellow, 10);
        row += 18;
        // Clear other labels
        SA_PanelCreateLabel("pair_syms", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("pair_corr", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("pair_beta", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("pair_hl", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("pair_hurst", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("pair_test", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("pair_kalman", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("zscore_val", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("zscore_thresh", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("zscore_spread", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("regime", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("pos_dir", px + 10, py - 100, "", clrBlack, 8);
        SA_PanelCreateLabel("pos_bars", px + 10, py - 100, "", clrBlack, 8);
    }

    // Top N candidates section
    SA_PanelCreateLabel("topsep", px + 10, py + row,
        "----------------------------------------", clrDimGray, 8);
    row += 14;
    SA_PanelCreateLabel("top_title", px + 10, py + row,
        "TOP " + IntegerToString(g_sa_panel_topn) + " CANDIDATES", clrCyan, 9);
    row += 16;

    for(int i = 0; i < g_sa_panel_topn; i++)
    {
        string label = "top_" + IntegerToString(i);
        if(i < g_sa_top_count)
        {
            string arrow = "";
            color row_clr = clrSilver;
            double tz = g_sa_top_pairs[i].current_zscore;
            if(tz >= StatArb_EntryZ) { arrow = " ^"; row_clr = clrYellow; }     // Sell spread opportunity
            else if(tz <= -StatArb_EntryZ) { arrow = " v"; row_clr = clrYellow; } // Buy spread opportunity
            if(g_sa_top_pairs[i].is_active) { row_clr = clrLime; arrow += " <"; }

            string sym_pair = g_sa_top_pairs[i].symbolY + "/" + g_sa_top_pairs[i].symbolX;
            // Pad/truncate to 16 chars
            while(StringLen(sym_pair) < 16) sym_pair += " ";
            if(StringLen(sym_pair) > 16) sym_pair = StringSubstr(sym_pair, 0, 16);

            SA_PanelCreateLabel(label, px + 10, py + row,
                IntegerToString(i + 1) + ". " + sym_pair
                + " r=" + DoubleToString(g_sa_top_pairs[i].correlation, 2)
                + " ADF=" + DoubleToString(g_sa_top_pairs[i].adf_stat, 1)
                + " Z=" + DoubleToString(tz, 2) + arrow,
                row_clr, 8);
        }
        else
        {
            SA_PanelCreateLabel(label, px + 10, py + row,
                IntegerToString(i + 1) + ". ---", clrDimGray, 8);
        }
        row += 14;
    }

    // Scan stats footer
    row += 4;
    SA_PanelCreateLabel("statsep", px + 10, py + row,
        "----------------------------------------", clrDimGray, 8);
    row += 14;

    int mins_since = (g_sa_pair.last_scan_time > 0)
        ? (int)((TimeCurrent() - g_sa_pair.last_scan_time) / 60) : -1;
    int mins_next = (StatArb_RescanInterval == 0) ? 0
        : (StatArb_RescanInterval > 0 && mins_since >= 0)
        ? (int)MathMax(0, StatArb_RescanInterval - mins_since) : -1;

    SA_PanelCreateLabel("scan_stats", px + 10, py + row,
        "Scanned: " + IntegerToString(g_sa_scan_total)
        + " | Corr: " + IntegerToString(g_sa_scan_corr_passed)
        + " | Coint: " + IntegerToString(g_sa_scan_cointegrated),
        clrDarkGray, 8);
    row += 14;

    string scan_time_str = "";
    if(mins_since >= 0) scan_time_str += "Last: " + IntegerToString(mins_since) + "m ago";
    if(mins_next >= 0) scan_time_str += " | Next: ~" + IntegerToString(mins_next) + "m";
    SA_PanelCreateLabel("scan_time", px + 10, py + row,
        scan_time_str, clrDarkGray, 8);
    row += 14;

    if(StatArb_TradeMode == SA_MODE_ALERT)
    {
        SA_PanelCreateLabel("alert_count", px + 10, py + row,
            "Alerts sent: " + IntegerToString(g_sa_alerts_sent), clrDarkGray, 8);
        row += 14;
    }
    else
    {
        SA_PanelCreateLabel("alert_count", px + 10, py + row,
            "Active trades: " + IntegerToString(g_sa_active_trades / 2),
            clrDarkGray, 8);
        row += 14;
    }

    // Resize background to fit
    int total_height = row + 10;
    SA_PanelCreateRect("bg", px, py, 340, total_height, C'25,25,35');

    ChartRedraw();
}

void SA_PanelDelete()
{
    int total = ObjectsTotal(ChartID());
    for(int i = total - 1; i >= 0; i--)
    {
        string name = ObjectName(ChartID(), i);
        if(StringFind(name, SA_PANEL_PREFIX) == 0)
            ObjectDelete(ChartID(), name);
    }
}

//+------------------------------------------------------------------+
//| SECTION 15: INIT / DEINIT / ONTICK                                |
//+------------------------------------------------------------------+

void StatArb_Init()
{
    if(!StatArb_Enabled) return;

    // Input validation
    if(StatArb_EntryZ <= 0)
    {
        Print("StatArb: ERROR - EntryZ must be > 0 (got ", StatArb_EntryZ, "). Disabling.");
        return;
    }
    if(StatArb_ExitZ < 0)
    {
        Print("StatArb: ERROR - ExitZ must be >= 0 (got ", StatArb_ExitZ, "). Disabling.");
        return;
    }
    if(StatArb_StopZ <= StatArb_EntryZ)
    {
        Print("StatArb: ERROR - StopZ (", StatArb_StopZ, ") must be > EntryZ (", StatArb_EntryZ, "). Disabling.");
        return;
    }
    // Fix 23: ExitZ must be strictly less than EntryZ.
    // If ExitZ >= EntryZ, every entry bar immediately satisfies the exit condition,
    // causing an open-close-open-close cycle on every bar.
    if(StatArb_ExitZ >= StatArb_EntryZ)
    {
        Print("StatArb: ERROR - ExitZ (", StatArb_ExitZ, ") must be < EntryZ (", StatArb_EntryZ, "). Disabling.");
        return;
    }
    if(StatArb_LookbackLength < 50)
    {
        Print("StatArb: ERROR - LookbackLength must be >= 50 (got ", StatArb_LookbackLength, "). Disabling.");
        return;
    }
    // Fix 42: ADF_Lags must be non-negative. A negative value causes out-of-bounds
    // array access in SA_ADF_Test (negative loop start, undersized row array).
    if(StatArb_ADF_Lags < 0)
    {
        Print("StatArb: ERROR - ADF_Lags must be >= 0 (got ", StatArb_ADF_Lags, "). Disabling.");
        return;
    }
    // Fix 66: JohansenLags must be non-negative. A negative value causes array
    // out-of-bounds in SA_JohansenTest (loop starts at negative index on y[t]/x[t]).
    if(StatArb_JohansenLags < 0)
    {
        Print("StatArb: ERROR - JohansenLags must be >= 0 (got ", StatArb_JohansenLags, "). Disabling.");
        return;
    }

    sa_trade.SetExpertMagicNumber(SA_PairMagic(0));
    sa_trade.SetDeviationInPoints(10);

    // Clamp PanelTopN to safe range (inputs cannot be modified at runtime)
    g_sa_panel_topn = (int)MathMax(0, (int)MathMin(20, StatArb_PanelTopN));

    SA_BuildCandidateList();

    if(g_sa_candidate_count < 2)
    {
        Print("StatArb: ERROR - Need at least 2 candidate symbols (got ", g_sa_candidate_count, "). Disabling.");
        return;
    }

    // Initialize pair array
    int max_pairs = (int)MathMax(1, StatArb_MaxPairs);
    ArrayResize(g_sa_pairs, max_pairs);
    g_sa_current_idx = 0;
    for(int p = 0; p < max_pairs; p++)
        g_sa_pairs[p].Reset();

    // Recover orphaned positions — close any SA-magic positions left over from
    // a previous session or a crash. Fix 34 supersedes Fix 30: Deinit now always
    // closes positions (for all reasons), so orphans here are genuine crash remnants.
    g_sa_deinit_reason = -1; // consume (no longer used for branching)
    {
        // Check all possible pair magic numbers
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong ticket = PositionGetTicket(i);
            if(ticket <= 0) continue;
            ulong pos_magic = (ulong)PositionGetInteger(POSITION_MAGIC);

            // Check if this magic belongs to any of our pair slots
            bool is_ours = false;
            for(int p = 0; p < max_pairs; p++)
            {
                if(pos_magic == SA_PairMagic(p)) { is_ours = true; break; }
            }
            if(!is_ours) continue;

            string sym = PositionGetString(POSITION_SYMBOL);
            Print("StatArb: Found orphaned position on ", sym,
                  " ticket=", ticket, " magic=", pos_magic, " — closing for safety.");
            // Fix 32: Check close result — failed closes leave unmanaged exposure.
            if(!sa_trade.PositionClose(ticket))
            {
                Print("StatArb: CRITICAL — Failed to close orphan ticket ", ticket,
                      " on ", sym, ": ", sa_trade.ResultRetcodeDescription());
                Alert("StatArb: Failed to close orphan on ", sym, " — manual intervention required!");
            }
        }
    }

    ArrayResize(g_sa_price_cache, 0);
    g_sa_price_cache_count = 0;
    ArrayResize(g_sa_top_pairs, 0);
    g_sa_top_count = 0;

    g_sa_last_scan_bar_time = 0;
    g_sa_last_alert_time = 0;
    g_sa_alerts_sent = 0;
    g_sa_active_trades = 0;
    g_sa_scan_total = 0;
    g_sa_scan_corr_passed = 0;
    g_sa_scan_cointegrated = 0;

    // Initialize ML module
    if(StatArb_MLMode != SA_ML_OFF)
        SAML_Init(max_pairs);

    g_sa_initialized = true;

    Print("StatArb: Initialized. Candidates=", g_sa_candidate_count,
          " Source=", EnumToString(StatArb_SymbolSource),
          " TF=", SA_TFString(),
          " Mode=", EnumToString(StatArb_TradeMode),
          " Hedge=", EnumToString(StatArb_HedgeMode),
          " Calc=", EnumToString(StatArb_CalcMode),
          " ML=", EnumToString(StatArb_MLMode),
          " MaxPairs=", max_pairs);

    if(StatArb_ShowPanel)
        SA_PanelCreate();
}

void StatArb_Deinit(int reason = REASON_REMOVE)
{
    if(!StatArb_Enabled) return;

    // Fix 34: Always close positions on Deinit, regardless of reason.
    // Close all active pairs' positions before destroying state.
    int max_pairs = ArraySize(g_sa_pairs);
    for(int p = 0; p < max_pairs; p++)
    {
        g_sa_current_idx = p;
        if(g_sa_pair.position_dir != 0)
        {
            SA_RefreshTickets();
            if(g_sa_pair.ticketY > 0 || g_sa_pair.ticketX > 0)
                SA_ClosePairTrade("EA DEINIT (reason=" + IntegerToString(reason) + ")");
        }
    }

    // Save ML state before cleanup
    if(StatArb_MLMode != SA_ML_OFF)
        SAML_Deinit();

    g_sa_deinit_reason = reason;

    // Cleanup
    for(int p = 0; p < max_pairs; p++)
        g_sa_pairs[p].Reset();
    ArrayResize(g_sa_pairs, 0);
    g_sa_current_idx = 0;

    for(int i = 0; i < g_sa_price_cache_count; i++)
        g_sa_price_cache[i].Reset();
    ArrayResize(g_sa_price_cache, 0);
    g_sa_price_cache_count = 0;

    ArrayResize(g_sa_top_pairs, 0);
    g_sa_top_count = 0;

    ArrayResize(g_sa_candidates, 0);
    g_sa_candidate_count = 0;

    g_sa_initialized = false;

    SA_PanelDelete();

    Print("StatArb: Deinitialized");
}

//+------------------------------------------------------------------+
//| Main OnTick entry point — called from ProcessSymbol               |
//+------------------------------------------------------------------+

void StatArb_OnTick(string symbol)
{
    if(!StatArb_Enabled || !g_sa_initialized) return;

    int max_pairs = ArraySize(g_sa_pairs);

    // Phase 1: Manage all active pairs whose Y symbol matches this tick
    bool any_managed = false;
    for(int p = 0; p < max_pairs; p++)
    {
        g_sa_current_idx = p;
        if(!g_sa_pair.active) continue;
        if(g_sa_pair.symbolY != symbol) continue;
        SA_ManageTrade(symbol);
        any_managed = true;
    }

    // Phase 2: Count active pairs
    int active_count = 0;
    for(int p = 0; p < max_pairs; p++)
    {
        if(g_sa_pairs[p].active) active_count++;
    }

    // Phase 3: Scan for new pairs if below max capacity
    if(active_count < max_pairs)
    {
        int free_slot = SA_FindFreeSlot(g_sa_pairs, max_pairs);
        if(free_slot >= 0)
        {
            ENUM_TIMEFRAMES scan_entry_tf, scan_hedge_tf;
            SA_ResolveTimeframes(scan_entry_tf, scan_hedge_tf);
            datetime scan_bar = iTime(symbol, scan_entry_tf, 0);
            if(scan_bar > 0 && scan_bar != g_sa_last_scan_bar_time)
            {
                g_sa_last_scan_bar_time = scan_bar;
                // Set current idx to free slot BEFORE scan so overlap check
                // correctly skips this (inactive) slot
                g_sa_current_idx = free_slot;

                SA_PairResult result;
                if(SA_ScanForBestPair(symbol, result))
                    SA_ActivatePair(result);
            }
        }
    }

    // Phase 4: Update panel (use first active pair for display, or idx 0)
    if(StatArb_ShowPanel && symbol == Symbol())
    {
        g_sa_current_idx = 0;
        // Find first active pair for primary display
        for(int p = 0; p < max_pairs; p++)
        {
            if(g_sa_pairs[p].active) { g_sa_current_idx = p; break; }
        }
        SA_PanelUpdate();
    }
}

//+------------------------------------------------------------------+
//| END OF FILE                                                        |
//+------------------------------------------------------------------+

#endif // SA_MQH
