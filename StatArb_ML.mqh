//+------------------------------------------------------------------+
//| StatArb_ML.mqh — ML-Enhanced Statistical Arbitrage                |
//| Uses CMLEngine (LightGBM/XGBoost/CatBoost/DeepTabular) for       |
//| entry gating, exit optimization, and full RL decisions.           |
//+------------------------------------------------------------------+
//
// ARCHITECTURE NOTES (v2 — contextual bandit with CMLEngine):
//
// 1. TRAINING REGIME: All stored experiences are terminal (entry->close).
//    The system is effectively a contextual bandit (features -> reward).
//    Each action has its own independent CMLEngine model that learns
//    the expected reward for that action given the feature vector.
//
// 2. MODELS: One CMLEngine model per action, all using ML_TASK_REGRESSION
//    (Huber loss). This is cleaner than a shared multi-output NN because
//    each model's internal buffer only contains relevant experiences.
//    - ENTRY_GATE (mode 1): model_take, model_skip
//    - FULL_RL (mode 3):    model_long, model_short, model_flat
//
// 3. EXIT ML (modes 2-3): Exit Q-values compare model_flat.Predict()
//    vs model for the current position action. StopZ safety net always
//    enforced. For production use, ENTRY_GATE mode (mode 1) is the
//    most reliable.
//
// 4. COUNTERFACTUALS: Q[skip/flat] is grounded via counterfactual
//    experiences stored on losing trades (reward=0, action=skip) and
//    when FULL_RL agent chooses to stay flat.
//
// 5. FEATURES AT ENTRY: Features 11-13 (bars_in_trade, entry_z distance,
//    unrealized PnL) and 17 (position_state) are always 0 in stored
//    experiences because features are captured before position opens.
//    These features only provide signal during exit inference (modes 2-3).
//
// 6. ALERT MODE: Virtual position closes do not call SAML_OnTradeClose,
//    so models never train in ALERT mode. ML modes are effective only
//    in AUTO trade mode.
//
#ifndef SA_ML_MQH
#define SA_ML_MQH

//+------------------------------------------------------------------+
//| SECTION 1: ENUMS & INPUTS                                         |
//+------------------------------------------------------------------+

enum ENUM_SA_ML_MODE
{
    SA_ML_OFF         = 0,  // Off (Classic Z-Score)
    SA_ML_ENTRY_GATE  = 1,  // ML Entry Gate
    SA_ML_ENTRY_EXIT  = 2,  // ML Entry + Exit
    SA_ML_FULL_RL     = 3   // Full RL Agent
};

input group "=============== StatArb ML ==============="
input ENUM_SA_ML_MODE  StatArb_MLMode          = SA_ML_OFF;    // ML operating mode
input double           StatArb_MLEpsilon        = 0.20;         // Initial exploration rate
input double           StatArb_MLEpsilonDecay   = 0.995;        // Epsilon decay per trade
input double           StatArb_MLEpsilonMin     = 0.05;         // Minimum epsilon floor
input int              StatArb_MLMinTrades      = 20;           // Warmup: min trades before ML gates
input double           StatArb_MLConfThreshold  = 0.0;          // Q-value margin: Q[take]-Q[skip] > this
input bool             StatArb_MLSaveWeights    = true;         // Persist ML models across sessions

//+------------------------------------------------------------------+
//| SECTION 2: CONSTANTS                                              |
//+------------------------------------------------------------------+

#define SA_ML_FEATURE_COUNT   18
#define SA_ML_ENTRY_ACTIONS   2   // {skip, take}  — modes 1-2 entry
#define SA_ML_FULL_ACTIONS    3   // {stay_flat, go_long, go_short} — mode 3

//+------------------------------------------------------------------+
//| SECTION 3: DATA STRUCTURES                                        |
//+------------------------------------------------------------------+

// Per-pair ML tracking (parallel to g_sa_pairs[])
struct SA_MLPairState
{
    // Z-score velocity tracking
    double prev_zscore;
    double prev_dz;

    // Entry snapshot for experience storage
    double entry_features[SA_ML_FEATURE_COUNT];
    int    entry_action;         // Action taken at entry
    bool   has_entry;            // Whether ML was involved

    // Trade excursion tracking
    double mfe;                  // Max favorable excursion (PnL)
    double mae;                  // Max adverse excursion (PnL)

    // Rate-limit FULL_RL stay_flat counterfactual to once per bar per pair
    datetime last_cf_bar;

    void Reset()
    {
        prev_zscore = 0;
        prev_dz = 0;
        ArrayInitialize(entry_features, 0);
        entry_action = 0;
        has_entry = false;
        mfe = 0;
        mae = 0;
        last_cf_bar = 0;
    }
};

//+------------------------------------------------------------------+
//| SECTION 4: GLOBAL ML STATE                                        |
//+------------------------------------------------------------------+

// CMLEngine model pointers — one model per action
// ENTRY_GATE / ENTRY_EXIT modes (2 action models):
CMLModelBase     *g_saml_model_take = NULL;   // Q-value model for "take entry"
CMLModelBase     *g_saml_model_skip = NULL;   // Q-value model for "skip entry"

// FULL_RL mode (3 action models):
CMLModelBase     *g_saml_model_long  = NULL;  // Q-value model for "go long"
CMLModelBase     *g_saml_model_short = NULL;  // Q-value model for "go short"
CMLModelBase     *g_saml_model_flat  = NULL;  // Q-value model for "stay flat"

double                g_sa_ml_epsilon = 0.20;
int                   g_sa_ml_total_trades = 0;
int                   g_sa_ml_total_wins = 0;
bool                  g_sa_ml_initialized = false;
SA_MLPairState        g_sa_ml_pair_state[];    // Parallel to g_sa_pairs[]

// Per-instance file prefix: avoids file collision when multiple EA instances
// run in the same terminal (different charts / different magic numbers)
string                g_sa_ml_file_prefix;     // Set in SAML_Init

// Statistics tracking
int    g_saml_stats_take_count  = 0;
int    g_saml_stats_skip_count  = 0;
int    g_saml_stats_train_count = 0;

//+------------------------------------------------------------------+
//| SECTION 5: FEATURE BUILDER                                        |
//+------------------------------------------------------------------+

double SAML_Tanh(double x) { return MathTanh(x); }
double SAML_Sigmoid(double x) { return 1.0 / (1.0 + MathExp(-x)); }
double SAML_Clamp(double x, double lo, double hi) { return MathMax(lo, MathMin(hi, x)); }

void SAML_BuildFeatures(const SA_ActivePair &pair, const SA_MLPairState &ml, double &features[])
{
    if(ArraySize(features) < SA_ML_FEATURE_COUNT)
        ArrayResize(features, SA_ML_FEATURE_COUNT);

    // 0: Z-score
    features[0] = SAML_Tanh(pair.current_zscore / 3.0);

    // 1: Z-score velocity (dz)
    double dz = pair.current_zscore - ml.prev_zscore;
    features[1] = SAML_Tanh(dz / 1.5);

    // 2: Z-score acceleration (ddz)
    double ddz = dz - ml.prev_dz;
    features[2] = SAML_Tanh(ddz / 1.0);

    // 3: Spread vs ring mean (if ring has data)
    double ring_std = 0;
    double ring_mean = 0;
    if(pair.ring_count > 2)
    {
        ring_mean = pair.rolling_sum / pair.ring_count;
        // Sample variance (N-1) to match SA_UpdateZScore's calculation
        double var = (pair.rolling_sum_sq - (double)pair.ring_count * ring_mean * ring_mean) / (pair.ring_count - 1.0);
        ring_std = (var > 0) ? MathSqrt(var) : 1e-10;
    }
    features[3] = (ring_std > 1e-10) ? SAML_Tanh((pair.current_spread - ring_mean) / ring_std) : 0;

    // 4: Correlation (already [-1, 1])
    features[4] = SAML_Clamp(pair.correlation, -1.0, 1.0);

    // 5: Half-life (sigmoid normalized)
    features[5] = 2.0 * SAML_Sigmoid(pair.half_life / 100.0) - 1.0;

    // 6: ADF p-value (low p = strong cointegration -> positive)
    features[6] = SAML_Clamp(1.0 - 2.0 * pair.adf_pvalue, -1.0, 1.0);

    // 7: Hurst (< 0.5 = mean-reverting -> negative)
    features[7] = SAML_Clamp(2.0 * (pair.hurst - 0.5), -1.0, 1.0);

    // 8: OU theta (mean-reversion speed)
    features[8] = SAML_Tanh(pair.ou_theta / 5.0);

    // 9: OU distance (spread vs OU mean)
    features[9] = (pair.ou_sigma > 1e-6) ?
        SAML_Tanh((pair.current_spread - pair.ou_mu) / pair.ou_sigma) : 0;

    // 10: Regime score
    features[10] = SAML_Tanh((pair.regime_score - 1.0) * 2.0);

    // 11: Bars in trade (normalized by half-life)
    double hl = MathMax(pair.half_life, 1.0);
    features[11] = (pair.position_dir != 0) ?
        SAML_Tanh((double)pair.bars_in_trade / (2.0 * hl)) : 0;

    // 12: Entry z-score distance
    features[12] = (pair.position_dir != 0) ?
        SAML_Tanh((pair.current_zscore - pair.entry_zscore) / 2.0) : 0;

    // 13: Unrealized PnL (normalized by account balance)
    // Use actual position PnL from tickets, not balance diff (avoids multi-pair contamination)
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    if(pair.position_dir != 0 && balance > 0)
    {
        double pnl_est = 0;
        if(pair.ticketY > 0 && PositionSelectByTicket(pair.ticketY))
            pnl_est += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);
        if(pair.ticketX > 0 && PositionSelectByTicket(pair.ticketX))
            pnl_est += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);
        features[13] = SAML_Tanh(pnl_est / balance * 100.0);
    }
    else
        features[13] = 0;

    // 14: Hedge ratio
    features[14] = SAML_Tanh(pair.hedge_ratio);

    // 15: Kalman innovation (divergence between Kalman beta and OLS beta)
    features[15] = pair.kalman_initialized ?
        SAML_Tanh((pair.kalman_beta - pair.hedge_ratio) / 0.1) : 0;

    // 16: Spread volatility (realized vs OU-predicted)
    // Using OU sigma as normalizer instead of spread mean (which is near-zero for
    // cointegrated pairs, causing CV = std/mean to saturate at tanh(huge) = 1.0)
    features[16] = (pair.ou_sigma > 1e-6 && ring_std > 1e-10) ? SAML_Tanh(ring_std / pair.ou_sigma - 1.0) : 0;

    // 17: Position state (-1, 0, +1)
    features[17] = (double)pair.position_dir;

    // Sanitize: NaN/Inf from bad upstream data would corrupt training buffer
    for(int i = 0; i < SA_ML_FEATURE_COUNT; i++)
        if(!MathIsValidNumber(features[i])) features[i] = 0;
}

//+------------------------------------------------------------------+
//| SECTION 6: REWARD FUNCTION                                        |
//+------------------------------------------------------------------+

double SAML_ComputeReward(const SA_ActivePair &pair, const SA_MLPairState &ml, double pair_pnl)
{
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    if(balance <= 0) balance = 10000; // Fallback

    // Base: PnL as percentage of account
    double pnl_pct = pair_pnl / balance * 100.0;

    // Duration penalty: penalize trades held too long relative to half-life
    double duration_penalty = 0;
    double reward_hl = MathMax(pair.half_life, 1.0);
    double hl_ratio = (double)pair.bars_in_trade / reward_hl;
    if(hl_ratio > 2.0)
        duration_penalty = MathMax(-0.5, -0.1 * (hl_ratio - 2.0)); // Floor: PnL signal always dominates

    // Risk bonus: reward good risk/reward ratio
    // Zero-drawdown winning trades get max bonus (ideal trade profile)
    double risk_bonus = 0;
    if(pair_pnl > 0)
    {
        if(ml.mae > 0)
            risk_bonus = 0.1 * MathMin(pair_pnl / ml.mae, 3.0);
        else
            risk_bonus = 0.3; // Max bonus: won without any drawdown
    }

    // Scale factor 2.0: preserves gradient for typical stat-arb PnL range (0.1%-2%).
    // At 5.0 (old value), tanh(0.5%*5)=0.99 vs tanh(2%*5)=0.9999 -- nearly identical.
    // At 2.0: tanh(0.5%*2)=0.46 vs tanh(2%*2)=0.96 -- meaningful separation.
    double reward = SAML_Tanh(pnl_pct * 2.0) + duration_penalty + risk_bonus;
    return SAML_Clamp(reward, -2.0, 2.0);
}

//+------------------------------------------------------------------+
//| SECTION 7: MODEL MANAGEMENT                                       |
//+------------------------------------------------------------------+

// Create a single CMLEngine model for one action
CMLModelBase* SAML_CreateOneModel(string label)
{
    CMLModelBase *model = ML_CreateModel(ShadowQL_MLModel, ML_TASK_REGRESSION, SA_ML_FEATURE_COUNT);
    if(model == NULL)
        Print("StatArb ML: FAILED to create model for ", label);
    else
        Print("StatArb ML: Created ", ML_ModelName(ShadowQL_MLModel), " model for ", label,
              " (features=", SA_ML_FEATURE_COUNT, ")");
    return model;
}

void SAML_InitModels()
{
    if(StatArb_MLMode == SA_ML_ENTRY_GATE || StatArb_MLMode == SA_ML_ENTRY_EXIT)
    {
        g_saml_model_take = SAML_CreateOneModel("take");
        g_saml_model_skip = SAML_CreateOneModel("skip");
    }

    if(StatArb_MLMode == SA_ML_FULL_RL)
    {
        g_saml_model_long  = SAML_CreateOneModel("long");
        g_saml_model_short = SAML_CreateOneModel("short");
        g_saml_model_flat  = SAML_CreateOneModel("flat");
    }
}

void SAML_DeinitModels()
{
    if(g_saml_model_take != NULL) { delete g_saml_model_take; g_saml_model_take = NULL; }
    if(g_saml_model_skip != NULL) { delete g_saml_model_skip; g_saml_model_skip = NULL; }
    if(g_saml_model_long != NULL) { delete g_saml_model_long; g_saml_model_long = NULL; }
    if(g_saml_model_short != NULL) { delete g_saml_model_short; g_saml_model_short = NULL; }
    if(g_saml_model_flat != NULL) { delete g_saml_model_flat; g_saml_model_flat = NULL; }
}

//+------------------------------------------------------------------+
//| SECTION 8: MODEL PERSISTENCE                                      |
//+------------------------------------------------------------------+

void SAML_SaveModels()
{
    if(!StatArb_MLSaveWeights || StatArb_MLMode == SA_ML_OFF) return;

    if(g_saml_model_take != NULL) g_saml_model_take.SaveToFile(g_sa_ml_file_prefix + "_take.bin");
    if(g_saml_model_skip != NULL) g_saml_model_skip.SaveToFile(g_sa_ml_file_prefix + "_skip.bin");
    if(g_saml_model_long != NULL) g_saml_model_long.SaveToFile(g_sa_ml_file_prefix + "_long.bin");
    if(g_saml_model_short != NULL) g_saml_model_short.SaveToFile(g_sa_ml_file_prefix + "_short.bin");
    if(g_saml_model_flat != NULL) g_saml_model_flat.SaveToFile(g_sa_ml_file_prefix + "_flat.bin");

    Print("StatArb ML: Models saved (prefix=", g_sa_ml_file_prefix, ")");
}

void SAML_LoadModels()
{
    if(!StatArb_MLSaveWeights || StatArb_MLMode == SA_ML_OFF) return;

    int loaded = 0;
    if(g_saml_model_take != NULL && FileIsExist(g_sa_ml_file_prefix + "_take.bin", FILE_COMMON))
        if(g_saml_model_take.LoadFromFile(g_sa_ml_file_prefix + "_take.bin")) loaded++;
    if(g_saml_model_skip != NULL && FileIsExist(g_sa_ml_file_prefix + "_skip.bin", FILE_COMMON))
        if(g_saml_model_skip.LoadFromFile(g_sa_ml_file_prefix + "_skip.bin")) loaded++;
    if(g_saml_model_long != NULL && FileIsExist(g_sa_ml_file_prefix + "_long.bin", FILE_COMMON))
        if(g_saml_model_long.LoadFromFile(g_sa_ml_file_prefix + "_long.bin")) loaded++;
    if(g_saml_model_short != NULL && FileIsExist(g_sa_ml_file_prefix + "_short.bin", FILE_COMMON))
        if(g_saml_model_short.LoadFromFile(g_sa_ml_file_prefix + "_short.bin")) loaded++;
    if(g_saml_model_flat != NULL && FileIsExist(g_sa_ml_file_prefix + "_flat.bin", FILE_COMMON))
        if(g_saml_model_flat.LoadFromFile(g_sa_ml_file_prefix + "_flat.bin")) loaded++;

    if(loaded > 0)
        Print("StatArb ML: Loaded ", loaded, " model(s) from disk");
}

// Save/Load scalar state (epsilon, trade counts) separately from model weights
void SAML_SaveScalars()
{
    if(StatArb_MLMode == SA_ML_OFF) return;

    int h = FileOpen(g_sa_ml_file_prefix + "_scalars.bin", FILE_WRITE | FILE_BIN | FILE_COMMON);
    if(h == INVALID_HANDLE)
    {
        Print("StatArb ML: Failed to save scalar state file");
        return;
    }

    FileWriteInteger(h, 200); // version 2.00 (CMLEngine migration)
    FileWriteInteger(h, (int)StatArb_MLMode);
    FileWriteDouble(h, g_sa_ml_epsilon);
    FileWriteInteger(h, g_sa_ml_total_trades);
    FileWriteInteger(h, g_sa_ml_total_wins);
    FileWriteInteger(h, g_saml_stats_take_count);
    FileWriteInteger(h, g_saml_stats_skip_count);
    FileWriteInteger(h, g_saml_stats_train_count);

    FileClose(h);
    Print("StatArb ML: Saved scalar state (trades=", g_sa_ml_total_trades,
          " eps=", DoubleToString(g_sa_ml_epsilon, 3), ")");
}

void SAML_LoadScalars()
{
    if(StatArb_MLMode == SA_ML_OFF) return;
    if(!FileIsExist(g_sa_ml_file_prefix + "_scalars.bin", FILE_COMMON)) return;

    int h = FileOpen(g_sa_ml_file_prefix + "_scalars.bin", FILE_READ | FILE_BIN | FILE_COMMON);
    if(h == INVALID_HANDLE) return;

    int version = FileReadInteger(h);
    if(version != 200) { FileClose(h); return; }

    int saved_mode = FileReadInteger(h);
    if(saved_mode != (int)StatArb_MLMode)
    {
        Print("StatArb ML: Mode changed (", saved_mode, " -> ", (int)StatArb_MLMode,
              "), resetting scalar state for warmup");
        FileClose(h);
        return; // Keep defaults (epsilon from input, trades=0)
    }

    g_sa_ml_epsilon = FileReadDouble(h);
    g_sa_ml_total_trades = FileReadInteger(h);
    g_sa_ml_total_wins = FileReadInteger(h);
    g_saml_stats_take_count = FileReadInteger(h);
    g_saml_stats_skip_count = FileReadInteger(h);
    g_saml_stats_train_count = FileReadInteger(h);

    FileClose(h);

    // Validate loaded scalars against corruption
    if(!MathIsValidNumber(g_sa_ml_epsilon) || g_sa_ml_epsilon < 0 || g_sa_ml_epsilon > 1.0)
    {
        Print("StatArb ML: WARNING -- Corrupt epsilon in state file (",
              DoubleToString(g_sa_ml_epsilon, 6), "), resetting to input value");
        g_sa_ml_epsilon = StatArb_MLEpsilon;
    }
    if(g_sa_ml_total_trades < 0) g_sa_ml_total_trades = 0;
    if(g_sa_ml_total_wins < 0 || g_sa_ml_total_wins > g_sa_ml_total_trades)
        g_sa_ml_total_wins = 0;

    Print("StatArb ML: Loaded scalar state (trades=", g_sa_ml_total_trades,
          " eps=", DoubleToString(g_sa_ml_epsilon, 3), ")");
}

//+------------------------------------------------------------------+
//| SECTION 9: TRAINING HELPERS                                       |
//+------------------------------------------------------------------+

// Train the model for a specific action with reward as the label.
// CMLEngine handles internal ring buffer, cold start, and batch training.
void SAML_TrainAction(CMLModelBase *model, const double &features[], double reward, string action_name)
{
    if(model == NULL) return;
    model.AddSample(features, SA_ML_FEATURE_COUNT, reward);
    model.Train();
    g_saml_stats_train_count++;
}

// Add sample without triggering Train() — for counterfactual grounding.
// Avoids expensive per-bar training; sample will be used next time Train() fires.
void SAML_AddCounterfactual(CMLModelBase *model, const double &features[], double reward, string action_name)
{
    if(model == NULL) return;
    model.AddSample(features, SA_ML_FEATURE_COUNT, reward);
}

// Predict Q-value for a given action model
double SAML_PredictAction(CMLModelBase *model, const double &features[])
{
    if(model == NULL) return 0.0;
    return model.Predict(features, SA_ML_FEATURE_COUNT);
}

//+------------------------------------------------------------------+
//| SECTION 10: ENTRY GATE (Modes 1-2)                                |
//+------------------------------------------------------------------+

// Returns true if ML approves the entry, false to block
bool SAML_GateEntry(SA_ActivePair &pair, int direction)
{
    if(StatArb_MLMode == SA_ML_OFF || !g_sa_ml_initialized) return true;
    if(StatArb_MLMode == SA_ML_FULL_RL) return true;  // FULL_RL uses SAML_FullRLDecision, not gate

    int pair_idx = g_sa_current_idx;
    if(pair_idx < 0 || pair_idx >= ArraySize(g_sa_ml_pair_state)) return true;

    // Build features
    double features[];
    ArrayResize(features, SA_ML_FEATURE_COUNT);
    SAML_BuildFeatures(pair, g_sa_ml_pair_state[pair_idx], features);

    // Warmup: always allow (but snapshot features for learning)
    if(g_sa_ml_total_trades < StatArb_MLMinTrades)
    {
        ArrayCopy(g_sa_ml_pair_state[pair_idx].entry_features, features, 0, 0, SA_ML_FEATURE_COUNT);
        g_sa_ml_pair_state[pair_idx].entry_action = 1; // "take"
        g_sa_ml_pair_state[pair_idx].has_entry = true;
        g_sa_ml_pair_state[pair_idx].mfe = 0;
        g_sa_ml_pair_state[pair_idx].mae = 0;
        return true;
    }

    // Epsilon-greedy exploration
    double rand_val = (double)MathRand() / 32768.0;
    if(rand_val < g_sa_ml_epsilon)
    {
        // Random: 50% take, 50% skip
        bool take = (MathRand() % 2 == 0);
        if(take)
        {
            ArrayCopy(g_sa_ml_pair_state[pair_idx].entry_features, features, 0, 0, SA_ML_FEATURE_COUNT);
            g_sa_ml_pair_state[pair_idx].entry_action = 1;
            g_sa_ml_pair_state[pair_idx].has_entry = true;
            g_sa_ml_pair_state[pair_idx].mfe = 0;
            g_sa_ml_pair_state[pair_idx].mae = 0;
        }
        if(take) g_saml_stats_take_count++;
        else     g_saml_stats_skip_count++;
        Print("StatArb ML: Exploration ", (take ? "TAKE" : "SKIP"),
              " (eps=", DoubleToString(g_sa_ml_epsilon, 3), ")");
        return take;
    }

    // Predict Q-values from separate models
    double q_take = SAML_PredictAction(g_saml_model_take, features);
    double q_skip = SAML_PredictAction(g_saml_model_skip, features);

    // Compare Q[take] vs Q[skip] with confidence threshold
    bool take = (q_take - q_skip) > StatArb_MLConfThreshold;

    if(take)
    {
        ArrayCopy(g_sa_ml_pair_state[pair_idx].entry_features, features, 0, 0, SA_ML_FEATURE_COUNT);
        g_sa_ml_pair_state[pair_idx].entry_action = 1;
        g_sa_ml_pair_state[pair_idx].has_entry = true;
        g_sa_ml_pair_state[pair_idx].mfe = 0;
        g_sa_ml_pair_state[pair_idx].mae = 0;
        g_saml_stats_take_count++;
    }
    else
    {
        g_saml_stats_skip_count++;
    }

    Print("StatArb ML: Gate ", (take ? "APPROVED" : "BLOCKED"),
          " Q[skip]=", DoubleToString(q_skip, 4),
          " Q[take]=", DoubleToString(q_take, 4),
          " dir=", direction);
    return take;
}

//+------------------------------------------------------------------+
//| SECTION 11: EXIT ML (Modes 2-3)                                   |
//+------------------------------------------------------------------+

// Returns true if ML recommends closing the position
// classic_exit: the classic Z-score exit decision (preserved during warmup)
bool SAML_ShouldExitML(SA_ActivePair &pair, bool classic_exit = false)
{
    if(!g_sa_ml_initialized) return classic_exit;
    if(StatArb_MLMode == SA_ML_ENTRY_GATE || StatArb_MLMode == SA_ML_OFF) return classic_exit;
    int pair_idx = g_sa_current_idx;
    if(pair_idx < 0 || pair_idx >= ArraySize(g_sa_ml_pair_state)) return classic_exit;

    // Warmup: defer to classic exit logic (don't suppress z-score exits)
    if(g_sa_ml_total_trades < StatArb_MLMinTrades)
        return classic_exit;

    // Build features (includes position state, bars, PnL)
    double features[];
    ArrayResize(features, SA_ML_FEATURE_COUNT);
    SAML_BuildFeatures(pair, g_sa_ml_pair_state[pair_idx], features);

    // Epsilon-greedy (reduced exploration for exits)
    double rand_val = (double)MathRand() / 32768.0;
    if(rand_val < g_sa_ml_epsilon * 0.5)
        return (MathRand() % 4 == 0); // 25% random exit

    bool should_close;
    if(StatArb_MLMode == SA_ML_FULL_RL)
    {
        if(pair.position_dir == 0) return false; // No position to exit
        // Compare Q[flat] vs Q[action that matches current position]
        double q_flat = SAML_PredictAction(g_saml_model_flat, features);
        CMLModelBase *hold_model = (pair.position_dir > 0) ? g_saml_model_long : g_saml_model_short;
        double q_hold = SAML_PredictAction(hold_model, features);
        should_close = (q_flat > q_hold);
        if(should_close)
            Print("StatArb ML: RL Exit signal Q[flat]=", DoubleToString(q_flat, 4),
                  " Q[hold]=", DoubleToString(q_hold, 4));
    }
    else
    {
        // ENTRY_EXIT: compare Q[skip/flat] vs Q[take/hold]
        // Q[skip] represents "not being in a trade" value
        // Q[take] represents "being in a trade" value
        double q_skip = SAML_PredictAction(g_saml_model_skip, features);
        double q_take = SAML_PredictAction(g_saml_model_take, features);
        should_close = (q_skip > q_take);
        if(should_close)
            Print("StatArb ML: Exit signal Q[hold]=", DoubleToString(q_take, 4),
                  " Q[close]=", DoubleToString(q_skip, 4));
    }

    return should_close;
}

//+------------------------------------------------------------------+
//| SECTION 12: FULL RL DECISION (Mode 3)                             |
//+------------------------------------------------------------------+

// Returns: 0=stay_flat, 1=go_long, 2=go_short
int SAML_FullRLDecision(SA_ActivePair &pair)
{
    if(!g_sa_ml_initialized) return 0;
    int pair_idx = g_sa_current_idx;
    if(pair_idx < 0 || pair_idx >= ArraySize(g_sa_ml_pair_state)) return 0;

    // Build features
    double features[];
    ArrayResize(features, SA_ML_FEATURE_COUNT);
    SAML_BuildFeatures(pair, g_sa_ml_pair_state[pair_idx], features);

    // Warmup: fall back to classic entry logic for training data
    if(g_sa_ml_total_trades < StatArb_MLMinTrades)
    {
        // Use boundary functions to match actual entry logic (respects OU mode)
        int bm = (int)StatArb_BoundaryMode;
        int action = 0;
        if(SA_ShouldEnterLong(pair.current_zscore, pair.current_spread, pair, bm))
            action = 1;
        else if(SA_ShouldEnterShort(pair.current_zscore, pair.current_spread, pair, bm))
            action = 2;

        if(action != 0)
        {
            ArrayCopy(g_sa_ml_pair_state[pair_idx].entry_features, features, 0, 0, SA_ML_FEATURE_COUNT);
            g_sa_ml_pair_state[pair_idx].entry_action = action;
            g_sa_ml_pair_state[pair_idx].has_entry = true;
            g_sa_ml_pair_state[pair_idx].mfe = 0;
            g_sa_ml_pair_state[pair_idx].mae = 0;
        }
        return action;
    }

    // Epsilon-greedy
    double rand_val = (double)MathRand() / 32768.0;
    if(rand_val < g_sa_ml_epsilon)
    {
        int action = MathRand() % SA_ML_FULL_ACTIONS;
        if(action != 0)
        {
            ArrayCopy(g_sa_ml_pair_state[pair_idx].entry_features, features, 0, 0, SA_ML_FEATURE_COUNT);
            g_sa_ml_pair_state[pair_idx].entry_action = action;
            g_sa_ml_pair_state[pair_idx].has_entry = true;
            g_sa_ml_pair_state[pair_idx].mfe = 0;
            g_sa_ml_pair_state[pair_idx].mae = 0;
        }
        else
        {
            // Counterfactual: ground Q[flat] -- rate-limited to once per bar per pair
            datetime cur_bar = iTime(pair.symbolY, PERIOD_CURRENT, 0);
            if(cur_bar != g_sa_ml_pair_state[pair_idx].last_cf_bar)
            {
                g_sa_ml_pair_state[pair_idx].last_cf_bar = cur_bar;
                SAML_AddCounterfactual(g_saml_model_flat, features, 0.0, "flat_cf");
            }
        }
        Print("StatArb ML: RL Exploration action=", action,
              " (eps=", DoubleToString(g_sa_ml_epsilon, 3), ")");
        return action;
    }

    // Predict Q-values from all three models
    double q_flat  = SAML_PredictAction(g_saml_model_flat,  features);
    double q_long  = SAML_PredictAction(g_saml_model_long,  features);
    double q_short = SAML_PredictAction(g_saml_model_short, features);

    // Argmax
    int best_action = 0;
    double best_q = q_flat;
    if(q_long > best_q) { best_q = q_long; best_action = 1; }
    if(q_short > best_q) { best_q = q_short; best_action = 2; }

    if(best_action != 0)
    {
        ArrayCopy(g_sa_ml_pair_state[pair_idx].entry_features, features, 0, 0, SA_ML_FEATURE_COUNT);
        g_sa_ml_pair_state[pair_idx].entry_action = best_action;
        g_sa_ml_pair_state[pair_idx].has_entry = true;
        g_sa_ml_pair_state[pair_idx].mfe = 0;
        g_sa_ml_pair_state[pair_idx].mae = 0;
    }
    else
    {
        // Counterfactual: ground Q[flat] -- rate-limited to once per bar per pair
        datetime cur_bar = iTime(pair.symbolY, PERIOD_CURRENT, 0);
        if(cur_bar != g_sa_ml_pair_state[pair_idx].last_cf_bar)
        {
            g_sa_ml_pair_state[pair_idx].last_cf_bar = cur_bar;
            SAML_AddCounterfactual(g_saml_model_flat, features, 0.0, "flat_cf");
        }
    }

    if(best_action != 0)
        Print("StatArb ML: RL Q[flat]=", DoubleToString(q_flat, 4),
              " Q[long]=", DoubleToString(q_long, 4),
              " Q[short]=", DoubleToString(q_short, 4),
              " -> action=", best_action);
    return best_action;
}

//+------------------------------------------------------------------+
//| SECTION 13: EVENT HANDLERS                                        |
//+------------------------------------------------------------------+

// Reset ML pair state when a position closes without going through SA_ClosePairTrade
void SAML_ResetPairML()
{
    if(StatArb_MLMode == SA_ML_OFF || !g_sa_ml_initialized) return;
    int pair_idx = g_sa_current_idx;
    if(pair_idx < 0 || pair_idx >= ArraySize(g_sa_ml_pair_state)) return;
    g_sa_ml_pair_state[pair_idx].has_entry = false;
}

// Called on each new bar while pair is active
void SAML_OnNewBar(SA_ActivePair &pair)
{
    if(StatArb_MLMode == SA_ML_OFF) return;
    int pair_idx = g_sa_current_idx;
    if(pair_idx < 0 || pair_idx >= ArraySize(g_sa_ml_pair_state)) return;

    // Update velocity tracking
    double dz = pair.current_zscore - g_sa_ml_pair_state[pair_idx].prev_zscore;
    g_sa_ml_pair_state[pair_idx].prev_dz = dz;
    g_sa_ml_pair_state[pair_idx].prev_zscore = pair.current_zscore;

    // Update MFE/MAE if in position -- use actual pair PnL from position tickets
    if(pair.position_dir != 0 && g_sa_ml_pair_state[pair_idx].has_entry)
    {
        double pnl = 0;
        if(pair.ticketY > 0 && PositionSelectByTicket(pair.ticketY))
            pnl += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);
        if(pair.ticketX > 0 && PositionSelectByTicket(pair.ticketX))
            pnl += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP) + PositionGetDouble(POSITION_COMMISSION);
        if(pnl > g_sa_ml_pair_state[pair_idx].mfe) g_sa_ml_pair_state[pair_idx].mfe = pnl;
        if(pnl < -g_sa_ml_pair_state[pair_idx].mae) g_sa_ml_pair_state[pair_idx].mae = -pnl; // mae stored as positive
    }
}

// Called when a trade closes -- trains the relevant CMLEngine model(s)
void SAML_OnTradeClose(SA_ActivePair &pair, double pair_pnl)
{
    if(StatArb_MLMode == SA_ML_OFF) return;
    int pair_idx = g_sa_current_idx;
    if(pair_idx < 0 || pair_idx >= ArraySize(g_sa_ml_pair_state)) return;

    if(!g_sa_ml_pair_state[pair_idx].has_entry) return;

    // Final MFE/MAE update using closed PnL
    if(pair_pnl > g_sa_ml_pair_state[pair_idx].mfe)
        g_sa_ml_pair_state[pair_idx].mfe = pair_pnl;
    if(pair_pnl < 0 && (-pair_pnl) > g_sa_ml_pair_state[pair_idx].mae)
        g_sa_ml_pair_state[pair_idx].mae = -pair_pnl;

    // Compute reward
    double reward = SAML_ComputeReward(pair, g_sa_ml_pair_state[pair_idx], pair_pnl);

    // Train the model for the action that was taken
    int action = g_sa_ml_pair_state[pair_idx].entry_action;

    if(StatArb_MLMode == SA_ML_ENTRY_GATE || StatArb_MLMode == SA_ML_ENTRY_EXIT)
    {
        // Train model_take with the actual reward
        SAML_TrainAction(g_saml_model_take, g_sa_ml_pair_state[pair_idx].entry_features, reward, "take");

        // Counterfactual: train model_skip with 0 reward on losing trades
        // This teaches "skipping would have been better than this losing trade"
        if(pair_pnl < 0)
            SAML_AddCounterfactual(g_saml_model_skip, g_sa_ml_pair_state[pair_idx].entry_features, 0.0, "skip_cf");
    }
    else if(StatArb_MLMode == SA_ML_FULL_RL)
    {
        // Train the model corresponding to the action taken
        if(action == 1)
            SAML_TrainAction(g_saml_model_long, g_sa_ml_pair_state[pair_idx].entry_features, reward, "long");
        else if(action == 2)
            SAML_TrainAction(g_saml_model_short, g_sa_ml_pair_state[pair_idx].entry_features, reward, "short");

        // Counterfactual: train model_flat with 0 reward on losing trades
        if(pair_pnl < 0)
            SAML_TrainAction(g_saml_model_flat, g_sa_ml_pair_state[pair_idx].entry_features, 0.0, "flat_cf");
    }

    // Track stats
    g_sa_ml_total_trades++;
    if(pair_pnl > 0) g_sa_ml_total_wins++;

    // Decay epsilon
    g_sa_ml_epsilon = MathMax(g_sa_ml_epsilon * StatArb_MLEpsilonDecay, StatArb_MLEpsilonMin);

    Print("StatArb ML: Trade closed PnL=", DoubleToString(pair_pnl, 2),
          " reward=", DoubleToString(reward, 3),
          " action=", action,
          " trades=", g_sa_ml_total_trades,
          " wins=", g_sa_ml_total_wins,
          " eps=", DoubleToString(g_sa_ml_epsilon, 3));

    // Reset pair ML state
    g_sa_ml_pair_state[pair_idx].has_entry = false;
}

//+------------------------------------------------------------------+
//| SECTION 14: INIT / DEINIT                                         |
//+------------------------------------------------------------------+

void SAML_Init(int max_pairs)
{
    if(StatArb_MLMode == SA_ML_OFF) return;

    // Build per-instance file prefix
    g_sa_ml_file_prefix = "StatArb_ML_" + Symbol() + "_" + IntegerToString(StatArb_MagicOffset);

    // Validate parameters
    if(StatArb_MLEpsilonDecay > 1.0)
        Print("StatArb ML: WARNING -- EpsilonDecay=", DoubleToString(StatArb_MLEpsilonDecay, 4),
              " > 1.0 -- epsilon will GROW instead of decay!");
    if(StatArb_MLEpsilonMin > StatArb_MLEpsilon)
        Print("StatArb ML: WARNING -- EpsilonMin(", DoubleToString(StatArb_MLEpsilonMin, 3),
              ") > Epsilon(", DoubleToString(StatArb_MLEpsilon, 3),
              ") -- floor above starting value, epsilon will jump to floor immediately");

    // Create CMLEngine models (one per action)
    SAML_InitModels();

    // Allocate per-pair state
    ArrayResize(g_sa_ml_pair_state, max_pairs);
    for(int p = 0; p < max_pairs; p++)
        g_sa_ml_pair_state[p].Reset();

    // Set initial epsilon and stats
    g_sa_ml_epsilon = StatArb_MLEpsilon;
    g_sa_ml_total_trades = 0;
    g_sa_ml_total_wins = 0;
    g_saml_stats_take_count = 0;
    g_saml_stats_skip_count = 0;
    g_saml_stats_train_count = 0;

    // Load persisted state (overrides defaults if files exist)
    SAML_LoadScalars();
    SAML_LoadModels();

    // If user increased epsilon input (e.g., to reset exploration), respect the new value
    if(g_sa_ml_epsilon < StatArb_MLEpsilon)
    {
        Print("StatArb ML: Epsilon input (", DoubleToString(StatArb_MLEpsilon, 3),
              ") > saved (", DoubleToString(g_sa_ml_epsilon, 3), ") -- using input value");
        g_sa_ml_epsilon = StatArb_MLEpsilon;
    }

    g_sa_ml_initialized = true;
    Print("StatArb ML: Initialized mode=", EnumToString(StatArb_MLMode),
          " model=", ML_ModelName(ShadowQL_MLModel),
          " features=", SA_ML_FEATURE_COUNT,
          " buffer=", ML_BufferCapacity,
          " cold_start=", ML_ColdStartMin);
}

void SAML_Deinit()
{
    if(StatArb_MLMode == SA_ML_OFF || !g_sa_ml_initialized) return;
    SAML_SaveModels();
    SAML_SaveScalars();
    SAML_DeinitModels();
    g_sa_ml_initialized = false;
    Print("StatArb ML: Deinitialized — trades=", g_sa_ml_total_trades,
          " wins=", g_sa_ml_total_wins,
          " take=", g_saml_stats_take_count,
          " skip=", g_saml_stats_skip_count,
          " train_calls=", g_saml_stats_train_count);
}

#endif // SA_ML_MQH
