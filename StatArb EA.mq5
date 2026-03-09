//+------------------------------------------------------------------+
//| StatArb EA.mq5 — Standalone Statistical Arbitrage Expert Advisor |
//| Pairs trading via cointegration (ADF/Johansen) + Z-score         |
//| ML-enhanced entry gating via CMLEngine (LightGBM/XGBoost/CB/DT) |
//+------------------------------------------------------------------+
#property copyright "StatArb EA"
#property version   "1.00"
#property description "Statistical Arbitrage EA with ML entry gating"
#property stacksize 65536

//+------------------------------------------------------------------+
//| Include order matters — dependencies flow top-down               |
//+------------------------------------------------------------------+
#include "SA_Common.mqh"         // Enums, normalization, CTrade, MagicNumber alias
#include "SA_Telegram.mqh"       // Telegram notification helpers
#include "StatArb.mqh"           // Core engine (internally includes CMLEngine + StatArb_Advanced + StatArb_ML)

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=== StatArb EA v1.00 initializing ===");
    Print("Magic base: ", SA_MagicBase, "  Offset: ", StatArb_MagicOffset);

    // Hide ATR indicator from tester chart (must be before any iATR calls)
    TesterHideIndicators(tester_hideIndicator);

    // Dodger Crimson chart template
    ChartSetInteger(0, CHART_SHOW_GRID, 0);
    ChartSetInteger(0, CHART_SHOW_PERIOD_SEP, 1);
    ChartSetInteger(0, CHART_COLOR_BACKGROUND, C'0,0,14');
    ChartSetInteger(0, CHART_COLOR_CANDLE_BULL, clrDodgerBlue);
    ChartSetInteger(0, CHART_COLOR_CANDLE_BEAR, clrCrimson);
    ChartSetInteger(0, CHART_COLOR_CHART_UP, clrDodgerBlue);
    ChartSetInteger(0, CHART_COLOR_CHART_DOWN, clrCrimson);
    ChartSetInteger(0, CHART_COLOR_FOREGROUND, clrWhite);
    ChartSetInteger(0, CHART_COLOR_BID, C'193,255,255');
    ChartSetInteger(0, CHART_COLOR_ASK, C'193,255,255');
    ChartSetInteger(0, CHART_COLOR_GRID, C'16,20,29');
    ChartRedraw(0);

    StatArb_Init();

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    StatArb_Deinit(reason);
    Print("=== StatArb EA deinitialized (reason: ", reason, ") ===");
}

//+------------------------------------------------------------------+
//| Expert tick handler                                               |
//+------------------------------------------------------------------+
void OnTick()
{
    StatArb_OnTick(Symbol());
}

//+------------------------------------------------------------------+
//| Chart event handler (for panel clicks)                            |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
    // Panel is display-only (OBJ_LABEL); no interactive click handling needed
}
