//+------------------------------------------------------------------+
//| SA_Common.mqh — Shared dependencies for standalone StatArb EA    |
//| Provides: enums, normalization, CTrade that were in main EA      |
//+------------------------------------------------------------------+
#ifndef SA_COMMON_MQH
#define SA_COMMON_MQH

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Enums (originally in Multisymbol EA.mq5 / LLM_Bridge.mqh)       |
//+------------------------------------------------------------------+
enum adr_atr_norml
{
    adr_norml     = 0  /*ADR norml*/,
    atr_norml     = 1  /*ATR norml*/
};

enum ENUM_LLM_OHLC_LOOKBACK
{
    LLM_OHLC_LB_BARS      = 0  /*Bars*/,
    LLM_OHLC_LB_HOURS     = 1  /*Hours*/,
    LLM_OHLC_LB_MIDNIGHT0 = 2  /*Today's midnight (00:00)*/,
    LLM_OHLC_LB_MIDNIGHT1 = 3  /*Yesterday's midnight (-1 day)*/,
    LLM_OHLC_LB_MIDNIGHT2 = 4  /*2 days ago midnight (-2 days)*/,
    LLM_OHLC_LB_MIDNIGHT3 = 5  /*3 days ago midnight (-3 days)*/,
    LLM_OHLC_LB_MIDNIGHT4 = 6  /*4 days ago midnight (-4 days)*/
};

//+------------------------------------------------------------------+
//| Normalization inputs                                              |
//+------------------------------------------------------------------+
input group "=============== Normalization ==============="
input adr_atr_norml SA_NormMode   = atr_norml;  // Spread normalization mode
input int           SA_ATR_Period = 450;         // ATR period
input int           SA_ADR_Days   = 28;          // ADR days period

//+------------------------------------------------------------------+
//| Magic number base (replaces main EA's MagicNumber)               |
//+------------------------------------------------------------------+
input group "=============== Trade Settings ==============="
input long SA_MagicBase = 900000;  // Magic number base
input bool tester_hideIndicator = true;  // Hide indicator in tester

// Alias so StatArb_Advanced.mqh SA_PairMagic() compiles unchanged
#define MagicNumber SA_MagicBase

//+------------------------------------------------------------------+
//| ATR cache for normalization (simplified from main EA)            |
//+------------------------------------------------------------------+
struct SA_ATRCache
{
    string   symbol;
    double   value;
    datetime calc_time;
    int      handle;
};

SA_ATRCache  g_sa_atr_cache[];
int          g_sa_atr_cache_count = 0;

int SA_GetOrCreateATRHandle(string symbol, ENUM_TIMEFRAMES tf, int period)
{
    for(int i = 0; i < g_sa_atr_cache_count; i++)
        if(g_sa_atr_cache[i].symbol == symbol)
            return g_sa_atr_cache[i].handle;

    int h = iATR(symbol, tf, period);
    if(h == INVALID_HANDLE)
    {
        Print("SA_Common: iATR handle failed for ", symbol, " — spread normalization will return 0");
        return INVALID_HANDLE;
    }
    if(g_sa_atr_cache_count < 200)
    {
        ArrayResize(g_sa_atr_cache, g_sa_atr_cache_count + 1, 32);
        g_sa_atr_cache[g_sa_atr_cache_count].symbol    = symbol;
        g_sa_atr_cache[g_sa_atr_cache_count].value     = 0.0;
        g_sa_atr_cache[g_sa_atr_cache_count].calc_time = 0;
        g_sa_atr_cache[g_sa_atr_cache_count].handle    = h;
        g_sa_atr_cache_count++;
    }
    return h;
}

double SA_GetATRValue(string symbol)
{
    datetime current_time = iTime(symbol, PERIOD_CURRENT, 0);

    for(int i = 0; i < g_sa_atr_cache_count; i++)
    {
        if(g_sa_atr_cache[i].symbol == symbol)
        {
            if(g_sa_atr_cache[i].calc_time == current_time && g_sa_atr_cache[i].value > 0)
                return g_sa_atr_cache[i].value;

            int handle = g_sa_atr_cache[i].handle;
            if(handle == INVALID_HANDLE)
                handle = SA_GetOrCreateATRHandle(symbol, PERIOD_CURRENT, SA_ATR_Period);

            if(handle != INVALID_HANDLE)
            {
                double buf[];
                ArraySetAsSeries(buf, true);
                if(CopyBuffer(handle, 0, 0, 1, buf) > 0 && buf[0] > 0)
                {
                    g_sa_atr_cache[i].value     = buf[0];
                    g_sa_atr_cache[i].calc_time = current_time;
                    return buf[0];
                }
            }
            return (g_sa_atr_cache[i].value > 0) ? g_sa_atr_cache[i].value : 0.0;
        }
    }

    // Not cached yet — create entry
    int handle = SA_GetOrCreateATRHandle(symbol, PERIOD_CURRENT, SA_ATR_Period);
    if(handle != INVALID_HANDLE)
    {
        double buf[];
        ArraySetAsSeries(buf, true);
        if(CopyBuffer(handle, 0, 0, 1, buf) > 0 && buf[0] > 0)
        {
            // Entry was already added by SA_GetOrCreateATRHandle, just update value
            for(int i = 0; i < g_sa_atr_cache_count; i++)
            {
                if(g_sa_atr_cache[i].symbol == symbol)
                {
                    g_sa_atr_cache[i].value     = buf[0];
                    g_sa_atr_cache[i].calc_time = iTime(symbol, PERIOD_CURRENT, 0);
                    return buf[0];
                }
            }
            return buf[0];
        }
    }
    return 0.0;
}

//+------------------------------------------------------------------+
//| ADR (Average Daily Range) with caching                           |
//+------------------------------------------------------------------+
struct SA_ADRCache
{
    string   symbol;
    double   value;
    datetime calc_time;
};

SA_ADRCache  g_sa_adr_cache[];
int          g_sa_adr_cache_count = 0;

double SA_GetADRValue(string symbol)
{
    datetime current_day = iTime(symbol, PERIOD_D1, 0);

    for(int i = 0; i < g_sa_adr_cache_count; i++)
    {
        if(g_sa_adr_cache[i].symbol == symbol)
        {
            if(g_sa_adr_cache[i].calc_time == current_day && g_sa_adr_cache[i].value > 0)
                return g_sa_adr_cache[i].value;

            // Recalculate
            double highs[], lows[];
            ArraySetAsSeries(highs, true);
            ArraySetAsSeries(lows, true);
            int copied = CopyHigh(symbol, PERIOD_D1, 1, SA_ADR_Days, highs);
            int copied2 = CopyLow(symbol, PERIOD_D1, 1, SA_ADR_Days, lows);
            if(copied >= SA_ADR_Days && copied2 >= SA_ADR_Days)
            {
                double sum = 0;
                for(int d = 0; d < SA_ADR_Days; d++)
                    sum += highs[d] - lows[d];
                g_sa_adr_cache[i].value     = sum / SA_ADR_Days;
                g_sa_adr_cache[i].calc_time = current_day;
                return g_sa_adr_cache[i].value;
            }
            return (g_sa_adr_cache[i].value > 0) ? g_sa_adr_cache[i].value : 0.0;
        }
    }

    // New entry
    if(g_sa_adr_cache_count < 200)
    {
        ArrayResize(g_sa_adr_cache, g_sa_adr_cache_count + 1, 32);
        g_sa_adr_cache[g_sa_adr_cache_count].symbol    = symbol;
        g_sa_adr_cache[g_sa_adr_cache_count].value     = 0.0;
        g_sa_adr_cache[g_sa_adr_cache_count].calc_time = 0;

        double highs[], lows[];
        ArraySetAsSeries(highs, true);
        ArraySetAsSeries(lows, true);
        int copied = CopyHigh(symbol, PERIOD_D1, 1, SA_ADR_Days, highs);
        int copied2 = CopyLow(symbol, PERIOD_D1, 1, SA_ADR_Days, lows);
        if(copied >= SA_ADR_Days && copied2 >= SA_ADR_Days)
        {
            double sum = 0;
            for(int d = 0; d < SA_ADR_Days; d++)
                sum += highs[d] - lows[d];
            g_sa_adr_cache[g_sa_adr_cache_count].value     = sum / SA_ADR_Days;
            g_sa_adr_cache[g_sa_adr_cache_count].calc_time = iTime(symbol, PERIOD_D1, 0);
            g_sa_adr_cache_count++;
            return sum / SA_ADR_Days;
        }
        g_sa_adr_cache_count++;
    }
    return 0.0;
}

//+------------------------------------------------------------------+
//| normalization() — same signature as main EA                      |
//| StatArb only uses adr_norml and atr_norml modes                  |
//+------------------------------------------------------------------+
double normalization(string symbol, double point1, double point2, adr_atr_norml norml)
{
    if(norml == adr_norml)
    {
        double adr = SA_GetADRValue(symbol);
        return (adr > 0) ? ((point1 - point2) / adr) * 100.0 : 0.0;
    }

    if(norml == atr_norml)
    {
        double atr = SA_GetATRValue(symbol);
        return (atr > 0) ? ((point1 - point2) / atr) : 0.0;
    }

    return 0.0;
}

// Alias for StatArb.mqh which uses adr_atr_norml_ as the global input name
#define adr_atr_norml_ SA_NormMode

#endif // SA_COMMON_MQH
