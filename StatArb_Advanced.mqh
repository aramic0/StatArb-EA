//+------------------------------------------------------------------+
//|                    StatArb_Advanced.mqh                           |
//|     Advanced algorithms for Statistical Arbitrage module          |
//|     Kalman, OU, Hurst, Variance Ratio, Johansen                  |
//+------------------------------------------------------------------+
#ifndef SA_ADVANCED_MQH
#define SA_ADVANCED_MQH

//+------------------------------------------------------------------+
//| Johansen test result struct                                       |
//+------------------------------------------------------------------+
struct SA_JohansenResult
{
    double trace_stat_r0;
    double maxeig_stat_r0;
    double trace_stat_r1;
    double maxeig_stat_r1;
    double eigenvec_1;
    double eigenvec_2;
    bool   cointegrated;
    double pvalue;
};

//+------------------------------------------------------------------+
//| Kalman Filter: 2x2 state-space model for hedge ratio              |
//| State: x_t = [beta_t, alpha_t]'                                   |
//| Observation: y_t = H_t * x_t + v_t, where H_t = [priceX, 1]      |
//| Transition: x_t = x_{t-1} + w_t (random walk)                    |
//+------------------------------------------------------------------+
void SA_KalmanUpdate(SA_ActivePair &pair, double priceY, double priceX,
                     double q, double r)
{
    if(!pair.kalman_initialized)
    {
        pair.kalman_beta  = pair.hedge_ratio;
        pair.kalman_alpha = pair.intercept;
        pair.kalman_P00   = 1.0;
        pair.kalman_P01   = 0.0;
        pair.kalman_P10   = 0.0;
        pair.kalman_P11   = 1.0;
        pair.kalman_initialized = true;
    }

    // Predict: P_pred = P + Q
    double P00 = pair.kalman_P00 + q;
    double P01 = pair.kalman_P01;
    double P10 = pair.kalman_P10;
    double P11 = pair.kalman_P11 + q;

    // Innovation: e = y - H*x
    double e = priceY - pair.kalman_beta * priceX - pair.kalman_alpha;

    // Innovation covariance: S = H*P*H' + R
    double S = priceX * priceX * P00 + priceX * (P01 + P10) + P11 + r;
    if(MathAbs(S) < 1e-20) return;

    // Kalman gain: K = P*H' / S
    double K0 = (P00 * priceX + P01) / S;
    double K1 = (P10 * priceX + P11) / S;

    // State update
    pair.kalman_beta  += K0 * e;
    pair.kalman_alpha += K1 * e;

    // Covariance update: P = (I - K*H)*P
    double I_KH_00 = 1.0 - K0 * priceX;
    double I_KH_01 = -K0;
    double I_KH_10 = -K1 * priceX;
    double I_KH_11 = 1.0 - K1;

    pair.kalman_P00 = I_KH_00 * P00 + I_KH_01 * P10;
    pair.kalman_P01 = I_KH_00 * P01 + I_KH_01 * P11;
    pair.kalman_P10 = I_KH_10 * P00 + I_KH_11 * P10;
    pair.kalman_P11 = I_KH_10 * P01 + I_KH_11 * P11;

    // Propagate to active hedge_ratio/intercept
    pair.hedge_ratio = pair.kalman_beta;
    pair.intercept   = pair.kalman_alpha;
}

//+------------------------------------------------------------------+
//| Estimate OU parameters from spread series                         |
//| dS = theta*(mu - S)*dt + sigma*dW                                |
//+------------------------------------------------------------------+
void SA_EstimateOU(const double &spread[], int n, double dt,
                   double &theta, double &mu, double &sigma)
{
    theta = 0; mu = 0; sigma = 0;
    if(n < 10) return;

    // AR(1) regression: S_t = a + b*S_{t-1}
    int m = n - 1;
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for(int i = 1; i < n; i++)
    {
        double x = spread[i - 1];
        double y = spread[i];
        sum_x  += x;
        sum_y  += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    double denom = (double)m * sum_xx - sum_x * sum_x;
    if(MathAbs(denom) < 1e-15) return;

    double b = ((double)m * sum_xy - sum_x * sum_y) / denom;
    double a = (sum_y - b * sum_x) / (double)m;

    if(b <= 0 || b >= 1.0) return;

    theta = -MathLog(b) / dt;
    mu = a / (1.0 - b);

    double sse = 0;
    for(int i = 1; i < n; i++)
    {
        double residual = spread[i] - a - b * spread[i - 1];
        sse += residual * residual;
    }
    double var_e = sse / (double)(m - 2);
    if(var_e <= 0) return;

    sigma = MathSqrt(var_e * (-2.0 * MathLog(b)) / (dt * (1.0 - b * b)));
}

//+------------------------------------------------------------------+
//| OU Optimal Entry/Exit boundaries (Bertram 2010 heuristic)         |
//+------------------------------------------------------------------+
void SA_OUOptimalBoundaries(double theta, double mu, double sigma,
                            double trans_cost, double stop_mult,
                            double &entry_long, double &entry_short,
                            double &exit_long, double &exit_short,
                            double &stop_long, double &stop_short)
{
    if(theta <= 0 || sigma <= 0)
    {
        entry_long = entry_short = exit_long = exit_short = 0;
        stop_long = stop_short = 0;
        return;
    }

    double ou_std = sigma / MathSqrt(2.0 * theta);
    double cost_adj = (ou_std > 1e-12) ? trans_cost / ou_std : 0;

    double entry_dist = ou_std * (1.2 + cost_adj * 0.5);
    double exit_dist  = ou_std * (0.1 + cost_adj * 0.3);

    if(entry_dist <= exit_dist)
        entry_dist = exit_dist + ou_std * 0.5;

    entry_long  = mu - entry_dist;
    entry_short = mu + entry_dist;
    exit_long   = mu - exit_dist;
    exit_short  = mu + exit_dist;
    stop_long   = mu - entry_dist * stop_mult;
    stop_short  = mu + entry_dist * stop_mult;
}

//+------------------------------------------------------------------+
//| OU boundary helpers: abstract z-score vs spread-based comparison   |
//+------------------------------------------------------------------+
bool SA_ShouldEnterLong(double z, double spread, const SA_ActivePair &pair,
                        int boundary_mode)
{
    if(boundary_mode == 1 && pair.ou_theta > 0) // SA_BOUND_OU
        return (spread <= pair.ou_entry_long && spread > pair.ou_stop_long);
    return (z <= -StatArb_EntryZ && z > -StatArb_StopZ);
}

bool SA_ShouldEnterShort(double z, double spread, const SA_ActivePair &pair,
                         int boundary_mode)
{
    if(boundary_mode == 1 && pair.ou_theta > 0)
        return (spread >= pair.ou_entry_short && spread < pair.ou_stop_short);
    return (z >= StatArb_EntryZ && z < StatArb_StopZ);
}

bool SA_ShouldExitLong(double z, double spread, const SA_ActivePair &pair,
                       int boundary_mode)
{
    if(boundary_mode == 1 && pair.ou_theta > 0)
        return (spread >= pair.ou_exit_long);
    return (z >= -StatArb_ExitZ);
}

bool SA_ShouldExitShort(double z, double spread, const SA_ActivePair &pair,
                        int boundary_mode)
{
    if(boundary_mode == 1 && pair.ou_theta > 0)
        return (spread <= pair.ou_exit_short);
    return (z <= StatArb_ExitZ);
}

bool SA_ShouldStopLong(double z, double spread, const SA_ActivePair &pair,
                       int boundary_mode)
{
    if(boundary_mode == 1 && pair.ou_theta > 0)
        return (spread <= pair.ou_stop_long);
    return (z <= -StatArb_StopZ);
}

bool SA_ShouldStopShort(double z, double spread, const SA_ActivePair &pair,
                        int boundary_mode)
{
    if(boundary_mode == 1 && pair.ou_theta > 0)
        return (spread >= pair.ou_stop_short);
    return (z >= StatArb_StopZ);
}

//+------------------------------------------------------------------+
//| Anis-Lloyd expected R/S for iid process of length n               |
//| E[R/S_n] = [Gamma((n-1)/2) / (sqrt(pi)*Gamma(n/2))]              |
//|          * sum_{i=1}^{n-1} sqrt((n-i)/i)                          |
//| Gamma ratio computed exactly via recurrence (no approximation).   |
//+------------------------------------------------------------------+
double SA_AnisLloydExpectedRS(int win)
{
    if(win < 3) return 1.0;

    double sum_term = 0;
    for(int i = 1; i < win; i++)
        sum_term += MathSqrt((double)(win - i) / i);

    // Exact Gamma((n-1)/2) / Gamma(n/2) via recurrence:
    //   f(2) = sqrt(pi),  f(n+1) = 2 / ((n-1) * f(n))
    // Then gamma_ratio = f(win) / sqrt(pi)
    double f = MathSqrt(M_PI); // f(2) = Gamma(0.5)/Gamma(1) = sqrt(pi)
    for(int k = 2; k < win; k++)
        f = 2.0 / ((double)(k - 1) * f);
    double gamma_ratio = f / MathSqrt(M_PI);

    return gamma_ratio * sum_term;
}

//+------------------------------------------------------------------+
//| Hurst Exponent via Modified R/S (Lo 1991 + Anis-Lloyd)            |
//| - Newey-West corrected variance (handles autocorrelated residuals)|
//| - Anis-Lloyd bias correction (removes iid R/S upward bias)       |
//+------------------------------------------------------------------+
double SA_HurstExponent(const double &series[], int n)
{
    if(n < 20) return 0.5;

    int sizes[];
    int size_count = 0;

    int sz = 10;
    while(sz <= n / 2)
    {
        ArrayResize(sizes, size_count + 1, 10);
        sizes[size_count++] = sz;
        sz = (int)(sz * 1.5);
        if(sz == sizes[size_count - 1]) sz++;
    }

    if(size_count < 3) return 0.5;

    double log_n[], log_rs[];
    ArrayResize(log_n, size_count);
    ArrayResize(log_rs, size_count);

    // Temp array for deviations (reused across blocks)
    double devs[];

    for(int si = 0; si < size_count; si++)
    {
        int win = sizes[si];
        int num_blocks = n / win;
        if(num_blocks < 1) { log_n[si] = 0; log_rs[si] = 0; continue; }

        ArrayResize(devs, win);
        // Newey-West bandwidth: q = floor(win^(1/3))
        int q = (int)MathFloor(MathPow((double)win, 1.0 / 3.0));
        if(q < 1) q = 1;

        double rs_sum = 0;
        int valid_blocks = 0;
        for(int b = 0; b < num_blocks; b++)
        {
            int bstart = b * win;
            double mean = 0;
            for(int i = 0; i < win; i++)
                mean += series[bstart + i];
            mean /= win;

            // Compute deviations, range, and gamma_0
            double cum = 0, max_cum = -1e30, min_cum = 1e30;
            double gamma0 = 0;
            for(int i = 0; i < win; i++)
            {
                devs[i] = series[bstart + i] - mean;
                cum += devs[i];
                if(cum > max_cum) max_cum = cum;
                if(cum < min_cum) min_cum = cum;
                gamma0 += devs[i] * devs[i];
            }
            gamma0 /= win;

            double R = max_cum - min_cum;

            // Lo (1991) Newey-West corrected variance
            // sigma^2_NW = gamma_0 + 2 * sum_{j=1}^{q} w_j * gamma_j
            // w_j = 1 - j/(q+1)  (Bartlett kernel)
            double sigma2 = gamma0;
            for(int j = 1; j <= q && j < win; j++)
            {
                double w_j = 1.0 - (double)j / ((double)q + 1.0);
                double gamma_j = 0;
                for(int t = 0; t < win - j; t++)
                    gamma_j += devs[t] * devs[t + j];
                gamma_j /= win;
                sigma2 += 2.0 * w_j * gamma_j;
            }

            double S = MathSqrt(MathMax(sigma2, 1e-30));
            if(S > 1e-15)
            {
                rs_sum += R / S;
                valid_blocks++;
            }
        }

        if(valid_blocks > 0)
        {
            double observed_rs = rs_sum / valid_blocks;
            // Anis-Lloyd correction: divide by expected R/S for iid process
            double expected_rs = SA_AnisLloydExpectedRS(win);
            double ratio = observed_rs / MathMax(expected_rs, 1e-10);

            log_n[si]  = MathLog((double)win);
            log_rs[si] = MathLog(MathMax(ratio, 1e-10));
        }
        else
        {
            log_n[si] = 0;
            log_rs[si] = 0;
        }
    }

    // Linear regression: log(R/S_NW / E[R/S]) = (H - 0.5) * log(n) + c
    // Slope gives (H - 0.5), so final H = slope + 0.5
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    int valid = 0;
    for(int i = 0; i < size_count; i++)
    {
        if(log_n[i] == 0 && log_rs[i] == 0) continue;
        sum_x  += log_n[i];
        sum_y  += log_rs[i];
        sum_xy += log_n[i] * log_rs[i];
        sum_xx += log_n[i] * log_n[i];
        valid++;
    }
    if(valid < 3) return 0.5;

    double denom = (double)valid * sum_xx - sum_x * sum_x;
    if(MathAbs(denom) < 1e-15) return 0.5;

    double slope = ((double)valid * sum_xy - sum_x * sum_y) / denom;
    double H = slope + 0.5;
    return MathMax(0.0, MathMin(1.0, H));
}

//+------------------------------------------------------------------+
//| Variance Ratio: VR(q) = Var(r_q) / (q * Var(r_1))                |
//| VR < 1 = mean-reverting, VR > 1 = trending                       |
//+------------------------------------------------------------------+
double SA_VarianceRatio(const double &spread[], int n, int q)
{
    if(n < q * 2 + 10) return 1.0;

    int m = n - 1;
    double r1[];
    ArrayResize(r1, m);
    double sum_r1 = 0;
    for(int i = 0; i < m; i++)
    {
        r1[i] = spread[i + 1] - spread[i];
        sum_r1 += r1[i];
    }
    double mean_r1 = sum_r1 / m;

    double var1 = 0;
    for(int i = 0; i < m; i++)
    {
        double d = r1[i] - mean_r1;
        var1 += d * d;
    }
    var1 /= (m - 1);

    int mq = n - q;
    double varq = 0;
    double mean_rq = mean_r1 * q;
    for(int i = 0; i < mq; i++)
    {
        double rq = spread[i + q] - spread[i];
        double d = rq - mean_rq;
        varq += d * d;
    }
    varq /= (mq - 1);

    if(var1 < 1e-15) return 1.0;
    return varq / (q * var1);
}

//+------------------------------------------------------------------+
//| Regime detection: returns true if currently mean-reverting         |
//+------------------------------------------------------------------+
bool SA_IsRegimeMeanReverting(SA_ActivePair &pair, int mode, int window,
                               double threshold)
{
    if(mode == 0) return true;

    int count = (int)MathMin(pair.ring_count, pair.ring_size);
    int use_n = (int)MathMin(count, window);
    if(use_n < 20) return true;

    double spreads[];
    ArrayResize(spreads, use_n);
    for(int i = 0; i < use_n; i++)
    {
        int idx = (pair.ring_head - use_n + i + pair.ring_size) % pair.ring_size;
        spreads[i] = pair.spread_ring[idx];
    }

    if(mode == 1) // Variance Ratio
    {
        double vr = SA_VarianceRatio(spreads, use_n, 2);
        pair.regime_score = vr;
        pair.regime_trending = (vr > threshold);
        return !pair.regime_trending;
    }
    else if(mode == 2) // Rolling Hurst
    {
        double h = SA_HurstExponent(spreads, use_n);
        pair.regime_score = h;
        pair.regime_trending = (h > threshold);
        return !pair.regime_trending;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Johansen Cointegration Test for 2 variables                       |
//| Critical values from Osterwald-Lenum (1992)                       |
//+------------------------------------------------------------------+
bool SA_JohansenTest(const double &y[], const double &x[], int n, int lags,
                     double crit_level, SA_JohansenResult &result)
{
    result.cointegrated = false;
    result.pvalue = 1.0;
    int T = n - lags - 1;
    if(T < 20) return false;

    // First differences
    int nd = n - 1;
    double dy[], dx[];
    ArrayResize(dy, nd);
    ArrayResize(dx, nd);
    for(int i = 0; i < nd; i++)
    {
        dy[i] = y[i + 1] - y[i];
        dx[i] = x[i + 1] - x[i];
    }

    // Accumulate moment matrices (simplified for lag=1 case)
    double S00_11 = 0, S00_12 = 0, S00_22 = 0;
    double S11_11 = 0, S11_12 = 0, S11_22 = 0;
    double S01_11 = 0, S01_12 = 0, S01_21 = 0, S01_22 = 0;

    for(int t = lags; t < nd; t++)
    {
        double r0_1 = dy[t];
        double r0_2 = dx[t];
        double r1_1 = y[t];
        double r1_2 = x[t];

        S00_11 += r0_1 * r0_1; S00_12 += r0_1 * r0_2; S00_22 += r0_2 * r0_2;
        S11_11 += r1_1 * r1_1; S11_12 += r1_1 * r1_2; S11_22 += r1_2 * r1_2;
        S01_11 += r0_1 * r1_1; S01_12 += r0_1 * r1_2;
        S01_21 += r0_2 * r1_1; S01_22 += r0_2 * r1_2;
    }

    double Tf = (double)T;
    S00_11 /= Tf; S00_12 /= Tf; S00_22 /= Tf;
    S11_11 /= Tf; S11_12 /= Tf; S11_22 /= Tf;
    S01_11 /= Tf; S01_12 /= Tf; S01_21 /= Tf; S01_22 /= Tf;

    // Invert S00 (2x2)
    double det00 = S00_11 * S00_22 - S00_12 * S00_12;
    if(MathAbs(det00) < 1e-20) return false;
    double iS00_11 =  S00_22 / det00;
    double iS00_12 = -S00_12 / det00;
    double iS00_22 =  S00_11 / det00;

    // M = S10 * inv(S00) * S01
    double S10_11 = S01_11, S10_12 = S01_21;
    double S10_21 = S01_12, S10_22 = S01_22;

    double t11 = S10_11 * iS00_11 + S10_12 * iS00_12;
    double t12 = S10_11 * iS00_12 + S10_12 * iS00_22;
    double t21 = S10_21 * iS00_11 + S10_22 * iS00_12;
    double t22 = S10_21 * iS00_12 + S10_22 * iS00_22;

    double M11 = t11 * S01_11 + t12 * S01_21;
    double M12 = t11 * S01_12 + t12 * S01_22;
    double M21 = t21 * S01_11 + t22 * S01_21;
    double M22 = t21 * S01_12 + t22 * S01_22;

    // Solve generalized eigenvalue: inv(S11) * M
    double det11 = S11_11 * S11_22 - S11_12 * S11_12;
    if(MathAbs(det11) < 1e-20) return false;
    double iS11_11 =  S11_22 / det11;
    double iS11_12 = -S11_12 / det11;
    double iS11_22 =  S11_11 / det11;

    double A11 = iS11_11 * M11 + iS11_12 * M21;
    double A12 = iS11_11 * M12 + iS11_12 * M22;
    double A21 = iS11_12 * M11 + iS11_22 * M21;
    double A22 = iS11_12 * M12 + iS11_22 * M22;

    // Eigenvalues of 2x2 A
    double trace_a = A11 + A22;
    double det = A11 * A22 - A12 * A21;
    double disc = trace_a * trace_a - 4.0 * det;
    if(disc < 0) disc = 0;
    double sqrt_disc = MathSqrt(disc);

    double lambda1 = (trace_a + sqrt_disc) / 2.0;
    double lambda2 = (trace_a - sqrt_disc) / 2.0;

    lambda1 = MathMax(0.0, MathMin(0.9999, lambda1));
    lambda2 = MathMax(0.0, MathMin(0.9999, lambda2));

    // Trace statistic
    result.trace_stat_r0 = -Tf * (MathLog(1.0 - lambda1) + MathLog(1.0 - lambda2));
    result.maxeig_stat_r0 = -Tf * MathLog(1.0 - lambda1);
    result.trace_stat_r1 = -Tf * MathLog(1.0 - lambda2);
    result.maxeig_stat_r1 = result.trace_stat_r1;

    // Eigenvector for lambda1
    double a_val = A11 - lambda1;
    double b_val = A12;
    if(MathAbs(b_val) > 1e-15 && MathAbs(a_val) > 1e-15)
    {
        result.eigenvec_1 = 1.0;
        result.eigenvec_2 = -a_val / b_val;
    }
    else if(MathAbs(a_val) > 1e-15)
    {
        result.eigenvec_1 = -b_val / a_val;
        result.eigenvec_2 = 1.0;
    }
    else
    {
        result.eigenvec_1 = 1.0;
        result.eigenvec_2 = 0.0;
    }

    // Normalize
    if(MathAbs(result.eigenvec_1) > 1e-15)
    {
        result.eigenvec_2 /= result.eigenvec_1;
        result.eigenvec_1 = 1.0;
    }

    // Critical values (trace, r=0, 2 variables, constant)
    double cv_10 = 13.43;
    double cv_05 = 15.41;
    double cv_01 = 20.04;

    if(result.trace_stat_r0 >= cv_01)
        result.pvalue = MathMax(0.001, 0.01 * cv_01 / result.trace_stat_r0);
    else if(result.trace_stat_r0 >= cv_05)
        result.pvalue = 0.01 + (cv_01 - result.trace_stat_r0) / (cv_01 - cv_05) * 0.04;
    else if(result.trace_stat_r0 >= cv_10)
        result.pvalue = 0.05 + (cv_05 - result.trace_stat_r0) / (cv_05 - cv_10) * 0.05;
    else
        result.pvalue = MathMin(1.0, 0.10 + (cv_10 - result.trace_stat_r0) * 0.10);

    result.cointegrated = (result.pvalue <= crit_level);
    return true;
}

//+------------------------------------------------------------------+
//| KPSS Stationarity Test (Kwiatkowski-Phillips-Schmidt-Shin)         |
//| H0: series IS stationary. Reject if stat > critical value.        |
//| Complement to ADF: ADF rejects non-stationarity, KPSS confirms it.|
//| Both must agree for robust cointegration evidence.                 |
//+------------------------------------------------------------------+
double SA_KPSS_Test(const double &series[], int n, double &pvalue)
{
    pvalue = 0.0; // H0: stationary → low p = reject stationarity
    if(n < 20) { pvalue = 1.0; return 0.0; }

    // Step 1: Detrend with constant (level stationarity)
    double sum_s = 0;
    for(int i = 0; i < n; i++) sum_s += series[i];
    double mean_s = sum_s / n;

    // Step 2: Cumulative partial sums of residuals
    double S[];
    ArrayResize(S, n);
    double cumsum = 0;
    for(int i = 0; i < n; i++)
    {
        cumsum += (series[i] - mean_s);
        S[i] = cumsum;
    }

    // Step 3: Numerator = (1/n^2) * sum(S_t^2)
    double sum_s2 = 0;
    for(int i = 0; i < n; i++)
        sum_s2 += S[i] * S[i];

    // Step 4: Newey-West long-run variance estimator
    // Bandwidth: q = int(4 * (n/100)^(2/9))  [Andrews 1991]
    int q = (int)(4.0 * MathPow((double)n / 100.0, 2.0 / 9.0));
    if(q < 1) q = 1;
    if(q >= n) q = n - 1;

    // Gamma_0
    double gamma0 = 0;
    for(int i = 0; i < n; i++)
    {
        double e = series[i] - mean_s;
        gamma0 += e * e;
    }
    gamma0 /= n;

    // Newey-West corrected variance
    double sigma2 = gamma0;
    for(int j = 1; j <= q; j++)
    {
        double w = 1.0 - (double)j / ((double)q + 1.0); // Bartlett kernel
        double gamma_j = 0;
        for(int t = j; t < n; t++)
            gamma_j += (series[t] - mean_s) * (series[t - j] - mean_s);
        gamma_j /= n;
        sigma2 += 2.0 * w * gamma_j;
    }

    if(sigma2 < 1e-20) { pvalue = 0.0; return 0.0; } // Perfectly constant

    // Step 5: KPSS statistic
    double kpss = sum_s2 / ((double)n * (double)n * sigma2);

    // Step 6: P-value interpolation from critical values (level stationarity)
    // cv_10=0.347, cv_05=0.463, cv_025=0.574, cv_01=0.739
    if(kpss >= 0.739)      pvalue = MathMin(1.0, 0.01 * kpss / 0.739); // > 1%
    else if(kpss >= 0.463) pvalue = 0.01 + (0.739 - kpss) / (0.739 - 0.463) * 0.04; // 1-5%
    else if(kpss >= 0.347) pvalue = 0.05 + (0.463 - kpss) / (0.463 - 0.347) * 0.05; // 5-10%
    else                   pvalue = MathMin(1.0, 0.10 + (0.347 - kpss) * 0.50); // >10% (stationary)

    return kpss;
}

//+------------------------------------------------------------------+
//| Cosine similarity between two hedge ratio vectors                  |
//| For pairs: vectors are [beta_old, 1] and [beta_new, 1]           |
//| Returns angle in degrees (0 = identical, 90 = orthogonal)         |
//+------------------------------------------------------------------+
double SA_HedgeRatioDrift(double beta_old, double beta_new)
{
    // Vectors: v1 = [beta_old, 1], v2 = [beta_new, 1]
    double dot = beta_old * beta_new + 1.0;
    double mag1 = MathSqrt(beta_old * beta_old + 1.0);
    double mag2 = MathSqrt(beta_new * beta_new + 1.0);
    if(mag1 < 1e-15 || mag2 < 1e-15) return 90.0;

    double cos_sim = dot / (mag1 * mag2);
    // Clamp for numerical safety
    cos_sim = MathMax(-1.0, MathMin(1.0, cos_sim));
    double angle_rad = MathArccos(cos_sim);
    return angle_rad * 180.0 / M_PI;
}

//+------------------------------------------------------------------+
//| In-Sample / Out-of-Sample split validation                         |
//| Estimates cointegration on IS portion, validates on OOS.           |
//| Returns OOS ADF p-value (low = stable relationship)               |
//+------------------------------------------------------------------+
double SA_OOS_Validate(const double &pricesY[], const double &pricesX[], int n,
                       double is_ratio, int adf_lags,
                       double &oos_beta, double &oos_adf_stat)
{
    oos_beta = 0;
    oos_adf_stat = 0;
    if(n < 60) return 1.0; // Need enough for both IS and OOS

    int is_n = (int)(n * is_ratio);
    int oos_n = n - is_n;
    if(is_n < 30 || oos_n < 20) return 1.0;

    // Step 1: OLS on IS data (oldest portion)
    double is_y[], is_x[];
    ArrayResize(is_y, is_n);
    ArrayResize(is_x, is_n);
    ArrayCopy(is_y, pricesY, 0, 0, is_n);
    ArrayCopy(is_x, pricesX, 0, 0, is_n);

    double beta = 0, alpha = 0;
    double is_resid[];
    SA_OLS(is_y, is_x, is_n, beta, alpha, is_resid);
    oos_beta = beta;

    if(MathAbs(beta) < 1e-10) return 1.0;

    // Step 2: Apply IS parameters to OOS data (newest portion)
    double oos_resid[];
    ArrayResize(oos_resid, oos_n);
    for(int i = 0; i < oos_n; i++)
        oos_resid[i] = pricesY[is_n + i] - alpha - beta * pricesX[is_n + i];

    // Step 3: ADF test on OOS residuals
    double oos_pvalue = 1.0;
    oos_adf_stat = SA_ADF_Test(oos_resid, oos_n, adf_lags, oos_pvalue);

    return oos_pvalue;
}

// NOTE: SA_CUSUMState struct + SA_CUSUM_Init/Update moved to StatArb.mqh
// (must be defined before SA_ActivePair which contains a SA_CUSUMState member)

//+------------------------------------------------------------------+
//| Multi-pair helpers                                                 |
//+------------------------------------------------------------------+
ulong SA_PairMagic(int pair_idx)
{
    return ((ulong)MagicNumber + (ulong)StatArb_MagicOffset + (ulong)pair_idx);
}

bool SA_IsSymbolInUse(const SA_ActivePair &pairs[], int pair_count, string symbol,
                      int skip_idx = -1)
{
    if(!StatArb_NoOverlap) return false;
    for(int p = 0; p < pair_count; p++)
    {
        if(p == skip_idx) continue;
        if(!pairs[p].active) continue;
        if(pairs[p].symbolY == symbol || pairs[p].symbolX == symbol)
            return true;
    }
    return false;
}

int SA_FindFreeSlot(const SA_ActivePair &pairs[], int max_slots)
{
    for(int p = 0; p < max_slots; p++)
    {
        if(!pairs[p].active) return p;
    }
    return -1;
}

#endif // SA_ADVANCED_MQH
