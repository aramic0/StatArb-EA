//+------------------------------------------------------------------+
//| CMLEngine.mqh — Pure MQL5 ML Engine                              |
//| LightGBM + XGBoost + CatBoost + Deep Tabular                    |
//| All 4 models train/predict natively in MT5, no Python/ONNX       |
//+------------------------------------------------------------------+
#ifndef CML_ENGINE_MQH
#define CML_ENGINE_MQH

//===================================================================
// ENUMS
//===================================================================
enum ENUM_ML_MODEL
{
   ML_LIGHTGBM,        // Leaf-wise histogram GBT
   ML_XGBOOST,         // Level-wise exact-split GBT
   ML_CATBOOST,        // Oblivious decision trees
   ML_DEEP_TABULAR     // MLP with PLE encoding
};

enum ENUM_ML_TASK
{
   ML_TASK_REGRESSION,     // Huber loss (direction model)
   ML_TASK_CLASSIFICATION, // Logloss (exit model)
   ML_TASK_QUANTILE        // Pinball loss (quantile regression)
};

//===================================================================
// DEFINES
//===================================================================
#define ML_NODE_WIDTH       3       // doubles per flat tree node
#define ML_LEAF_MARKER     -1.0     // leaf indicator in flat tree
#define ML_MAGIC_SAVE      0x4D4C4D44   // "MLMD"
#define ML_MAGIC_END       0x4D4C454E   // "MLEN"
#define ML_EPS             1e-8
#define ML_HUBER_DELTA     1.0
#define ML_MAX_TREE_NODES  512      // max nodes per tree
#define ML_MAX_CB_DEPTH    10       // max CatBoost oblivious depth
#define ML_ADAM_BETA1      0.9
#define ML_ADAM_BETA2      0.999
#define ML_NAN_SENTINEL   -1e308     // sentinel for missing values (replaces NaN→0.0)
#define ML_MAGIC_GBT_V2    0x47425432   // "GBT2" — extension block for GBT v2 features
#define ML_MAGIC_DT_V2     0x44545632   // "DTV2" — extension block for DeepTabular v2
#define ML_NODE_WIDTH_V2   4       // doubles per flat tree node (v2: +NaN direction)
#define ML_MAGIC_GBT_V3    0x47425433   // "GBT3" — NaN direction per node
#define ML_MAGIC_DART      0x44415254   // "DART" — tree weights extension
#define ML_NAN_GOES_RIGHT  0.0          // NaN direction flag: right child
#define ML_NAN_GOES_LEFT   1.0          // NaN direction flag: left child
#define ML_MAGIC_V4        0x47425434   // "GBT4" — V4 extensions (interaction/lossguide)

//===================================================================
// INPUT PARAMETERS
//===================================================================
input group "=============== ML Engine ==============="
input ENUM_ML_MODEL ShadowQL_MLModel   = ML_DEEP_TABULAR; // ML Model Type
input int    ML_MaxTrees        = 200;    // Max trees in ensemble
input int    ML_TreesPerRound   = 3;      // New trees per training round
input int    ML_MaxLeaves       = 31;     // Max leaves (LightGBM)
input int    ML_MaxDepth        = 6;      // Max depth (XGBoost/CatBoost)
input int    ML_NBins           = 63;     // Histogram bins
input double ML_LearningRate    = 0.1;    // Shrinkage
input double ML_L1Reg           = 0.0;    // L1 regularization (alpha)
input double ML_L2Reg           = 1.0;    // L2 regularization (lambda)
input double ML_Gamma           = 0.0;    // Min split gain
input double ML_Subsample       = 0.8;    // Row subsampling ratio
input double ML_ColsampleTree   = 0.8;    // Feature subsampling per tree
input int    ML_MinChildSamples = 20;     // Min samples per leaf
input int    ML_BufferCapacity  = 5000;   // Training ring buffer size
input int    ML_ColdStartMin    = 50;     // Min samples before first train
input bool   ML_SaveEnabled     = true;   // Save/load models
input int    ML_PLE_Bins        = 4;      // PLE bins per feature (Deep Tab)
input double ML_DT_LearningRate = 0.001;  // Deep Tabular learning rate
input int    ML_DT_Hidden1      = 128;    // Deep Tabular hidden layer 1
input int    ML_DT_Hidden2      = 64;     // Deep Tabular hidden layer 2
input double ML_DT_Dropout      = 0.2;    // Deep Tabular dropout rate
input int    ML_DT_BatchSize    = 64;     // Deep Tabular mini-batch size
input int    ML_DT_Epochs       = 5;      // Deep Tabular epochs per train

input group "=============== ML Engine V2 ==============="
input double ML_MinChildWeight    = 1.0;   // Min Hessian sum per leaf (GBTs)
input int    ML_EarlyStopRounds   = 0;    // Early stopping patience (0=disabled)
input double ML_GOSS_TopRate      = 0.2;   // GOSS top gradient fraction (LightGBM)
input double ML_GOSS_OtherRate    = 0.1;   // GOSS random sample fraction (LightGBM)
input double ML_ColsampleLevel    = 1.0;   // Column subsample per depth level (XGBoost)
input int    ML_DT_EnsembleK      = 8;     // TabM ensemble members (1=disabled)
input double ML_DT_WeightDecay    = 0.01;  // AdamW decoupled weight decay
input double ML_DT_GradClipNorm   = 1.0;   // Gradient norm clipping (0=disabled)
input int    ML_DT_LRSchedule     = 0;     // LR schedule: 0=None 1=Cosine 2=WarmupCosine

input group "=============== ML Engine V3 ==============="
input string ML_MonotoneConstraints  = "";     // "+1,0,-1,..." per feature (empty=off)
input bool   ML_DARTEnabled          = false;  // DART: dropout on trees during training
input double ML_DARTDropRate         = 0.1;    // DART: fraction of trees to drop per round
input double ML_ColsampleNode        = 1.0;    // XGBoost: column subsample per node (1=off)
input double ML_CB_RandomStrength    = 0.0;    // CatBoost: annealed split score noise (0=off)
input double ML_CB_BaggingTemp       = 0.0;    // CatBoost: Bayesian bootstrap temp (0=uniform)
input double ML_QuantileAlpha        = 0.5;    // Quantile level for ML_TASK_QUANTILE

input group "=============== ML Engine V4 ==============="
input string ML_InteractionConstraints = "";    // "[0,1,2],[3,4,5]" feature groups (empty=off)
input int    ML_CB_GrowPolicy          = 0;     // CatBoost: 0=Symmetric 1=Lossguide
input bool   ML_EnableEFB              = false;  // EFB feature bundling (LightGBM, experimental)
input int    ML_CB_OrderedPerms        = 1;      // Ordered boosting perms: 1=simplified (max 4)
input string ML_CategoricalFeatures    = "";     // Comma-sep categorical feature indices (empty=none)
input bool   ML_EnableLinearLeaves     = false;  // Linear regression in leaves (LightGBM)
input int    ML_LinearTreeMaxFeatures  = 30;     // Max features for linear leaf solve

//===================================================================
// HELPER FUNCTIONS
//===================================================================
double ML_Sign(double x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); }
double ML_Sigmoid(double x) { return 1.0 / (1.0 + MathExp(-MathMax(-500.0, MathMin(500.0, x)))); }

double ML_ThresholdL1(double g, double alpha)
{
   if(alpha <= 0.0) return g;
   return ML_Sign(g) * MathMax(MathAbs(g) - alpha, 0.0);
}

double ML_LeafValue(double grad_sum, double hess_sum, double l1, double l2)
{
   return -ML_ThresholdL1(grad_sum, l1) / (hess_sum + l2 + ML_EPS);
}

double ML_SplitGain(double g_l, double h_l, double g_r, double h_r, double l1, double l2, double gamma)
{
   double tl = ML_ThresholdL1(g_l, l1);
   double tr = ML_ThresholdL1(g_r, l1);
   double tp = ML_ThresholdL1(g_l + g_r, l1);
   return 0.5 * (tl*tl / (h_l + l2 + ML_EPS) +
                  tr*tr / (h_r + l2 + ML_EPS) -
                  tp*tp / (h_l + h_r + l2 + ML_EPS)) - gamma;
}

// Gradient/Hessian for Huber loss (regression)
double ML_HuberGrad(double pred, double label)
{
   double r = pred - label;
   if(MathAbs(r) <= ML_HUBER_DELTA) return r;
   return ML_HUBER_DELTA * ML_Sign(r);
}
double ML_HuberHess(double pred, double label)
{
   return (MathAbs(pred - label) <= ML_HUBER_DELTA) ? 1.0 : ML_EPS;
}

// Gradient/Hessian for logloss (classification)
double ML_LoglossGrad(double pred, double label)
{
   double p = ML_Sigmoid(pred);
   return p - label;
}
double ML_LoglossHess(double pred, double label)
{
   double p = ML_Sigmoid(pred);
   return MathMax(p * (1.0 - p), ML_EPS);
}

// Gradient/Hessian for quantile regression (smoothed pinball loss)
double ML_QuantileGrad(double pred, double label)
{
   double r = pred - label;
   double delta = 0.01 * MathMax(MathAbs(r), 0.001);
   if(r > delta)  return ML_QuantileAlpha;
   if(r < -delta) return ML_QuantileAlpha - 1.0;
   return (ML_QuantileAlpha - 0.5) + r / (2.0 * delta);
}
double ML_QuantileHess(double pred, double label)
{
   double r = pred - label;
   double delta = 0.01 * MathMax(MathAbs(r), 0.001);
   if(MathAbs(r) <= delta) return 1.0 / (2.0 * delta);
   return ML_EPS;
}

// xorshift64 PRNG — 64-bit quality, replaces 15-bit MathRand()
static ulong g_ml_rng_state = 0;

void ML_SeedRng(ulong seed)
{
   g_ml_rng_state = (seed != 0) ? seed : 6364136223846793005ULL;
}

ulong ML_Xorshift64()
{
   ulong x = g_ml_rng_state;
   x ^= x << 13;
   x ^= x >> 7;
   x ^= x << 17;
   g_ml_rng_state = x;
   return x;
}

// Uniform [0, 1) with full double precision (53 bits)
double ML_RandDouble()
{
   return (double)(ML_Xorshift64() >> 11) * (1.0 / 9007199254740992.0);
}

// Normal distribution via Box-Muller (needed for TabM r/s init)
double ML_RandNormal(double mu = 0.0, double sigma = 1.0)
{
   static bool has_spare = false;
   static double spare = 0.0;
   if(has_spare) { has_spare = false; return mu + sigma * spare; }
   double u, v, s;
   do {
      u = ML_RandDouble() * 2.0 - 1.0;
      v = ML_RandDouble() * 2.0 - 1.0;
      s = u * u + v * v;
   } while(s >= 1.0 || s == 0.0);
   double mul = MathSqrt(-2.0 * MathLog(s) / s);
   spare = v * mul;
   has_spare = true;
   return mu + sigma * u * mul;
}

// Random int in [0, max_val) without modulo bias for small ranges
int ML_RandInt(int max_val)
{
   if(max_val <= 1) return 0;
   return (int)(ML_Xorshift64() % (ulong)max_val);
}

void ML_Shuffle(int &arr[], int count)
{
   for(int i = count - 1; i > 0; i--)
   {
      int j = ML_RandInt(i + 1);
      int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
   }
}

// Merge sort for argsort: sorts idx[] by vals[idx[]]
void ML_MergeSortStep(const double &vals[], int &idx[], int &temp[], int left, int right)
{
   if(left >= right) return;
   int mid = (left + right) / 2;
   ML_MergeSortStep(vals, idx, temp, left, mid);
   ML_MergeSortStep(vals, idx, temp, mid + 1, right);
   int i = left, j = mid + 1, k = left;
   while(i <= mid && j <= right)
   {
      if(vals[idx[i]] <= vals[idx[j]])
         temp[k++] = idx[i++];
      else
         temp[k++] = idx[j++];
   }
   while(i <= mid) temp[k++] = idx[i++];
   while(j <= right) temp[k++] = idx[j++];
   for(int m = left; m <= right; m++) idx[m] = temp[m];
}

void ML_ArgSort(const double &vals[], int &idx[], int count)
{
   for(int i = 0; i < count; i++) idx[i] = i;
   int temp[];
   ArrayResize(temp, count);
   ML_MergeSortStep(vals, idx, temp, 0, count - 1);
}

// Flat tree prediction (shared by LightGBM and XGBoost)
double ML_PredictFlatTree(const double &trees[], int tree_start,
                          const double &features[], int n_features)
{
   int offs = tree_start;
   int safety = 0;
   while(trees[offs] >= 0.0 && safety < 200)
   {
      int feat = (int)trees[offs];
      if(feat >= 0 && feat < n_features && features[feat] < trees[offs + 1])
         offs += ML_NODE_WIDTH;   // left child
      else
         offs = tree_start + (int)trees[offs + 2]; // right child
      safety++;
   }
   return trees[offs + 1]; // leaf value
}

// Flat tree prediction V2: 4 doubles per node (feat, thresh, right_off, nan_dir)
// NaN-aware: when feature value is NaN sentinel, uses nan_direction to decide path
double ML_PredictFlatTreeV2(const double &trees[], int tree_start,
                            const double &features[], int n_features)
{
   int offs = tree_start;
   int safety = 0;
   while(trees[offs] >= 0.0 && safety < 200)
   {
      int feat = (int)trees[offs];
      double fval = (feat >= 0 && feat < n_features) ? features[feat] : 0.0;
      bool is_nan = (fval <= ML_NAN_SENTINEL + 1e300);
      if(is_nan)
      {
         // Use NaN direction: 1.0 = go left, 0.0 = go right
         if(trees[offs + 3] > 0.5)
            offs += ML_NODE_WIDTH_V2;   // left child
         else
            offs = tree_start + (int)trees[offs + 2]; // right child
      }
      else
      {
         if(fval < trees[offs + 1])
            offs += ML_NODE_WIDTH_V2;   // left child
         else
            offs = tree_start + (int)trees[offs + 2]; // right child
      }
      safety++;
   }
   return trees[offs + 1]; // leaf value
}

// Matrix/Vector file I/O helpers (for Deep Tabular)
void ML_SaveMatrix(int handle, matrix &m)
{
   int rows = (int)m.Rows(), cols = (int)m.Cols();
   FileWriteInteger(handle, rows);
   FileWriteInteger(handle, cols);
   for(int r = 0; r < rows; r++)
      for(int c = 0; c < cols; c++)
         FileWriteDouble(handle, m[r][c]);
}

bool ML_LoadMatrix(int handle, matrix &m)
{
   int rows = FileReadInteger(handle);
   int cols = FileReadInteger(handle);
   if(rows <= 0 || cols <= 0 || rows > 10000 || cols > 10000) return false;
   m.Resize(rows, cols);
   for(int r = 0; r < rows; r++)
      for(int c = 0; c < cols; c++)
         m[r][c] = FileReadDouble(handle);
   return true;
}

void ML_SaveVector(int handle, vector &v)
{
   int sz = (int)v.Size();
   FileWriteInteger(handle, sz);
   for(int i = 0; i < sz; i++)
      FileWriteDouble(handle, v[i]);
}

bool ML_LoadVector(int handle, vector &v)
{
   int sz = FileReadInteger(handle);
   if(sz <= 0 || sz > 100000) return false;
   v.Resize(sz);
   for(int i = 0; i < sz; i++)
      v[i] = FileReadDouble(handle);
   return true;
}

//===================================================================
// TRAINING BUFFER (Ring buffer for online learning)
//===================================================================
class CMLTrainingBuffer
{
private:
   double m_features[];     // [capacity * max_features]
   double m_labels[];       // [capacity]
   double m_weights[];      // [capacity]
   int    m_count;
   int    m_capacity;
   int    m_max_features;
   int    m_write_pos;

public:
   CMLTrainingBuffer() : m_count(0), m_capacity(0), m_max_features(0), m_write_pos(0) {}

   void Init(int capacity, int max_features)
   {
      m_capacity = capacity;
      m_max_features = max_features;
      ArrayResize(m_features, capacity * max_features);
      ArrayResize(m_labels, capacity);
      ArrayResize(m_weights, capacity);
      ArrayInitialize(m_features, 0.0);
      ArrayInitialize(m_labels, 0.0);
      ArrayInitialize(m_weights, 1.0);
      m_count = 0;
      m_write_pos = 0;
   }

   void AddSample(const double &features[], int feat_count, double label, double weight)
   {
      int offset = m_write_pos * m_max_features;
      int copy_count = MathMin(feat_count, m_max_features);
      ArrayCopy(m_features, features, offset, 0, copy_count);
      for(int i = copy_count; i < m_max_features; i++)
         m_features[offset + i] = 0.0;
      // Replace NaN with sentinel (preserves missingness for NaN-aware splits)
      for(int i = 0; i < m_max_features; i++)
         if(!MathIsValidNumber(m_features[offset + i]))
            m_features[offset + i] = ML_NAN_SENTINEL;
      m_labels[m_write_pos] = label;
      m_weights[m_write_pos] = weight;
      m_write_pos = (m_write_pos + 1) % m_capacity;
      if(m_count < m_capacity) m_count++;
   }

   double GetFeature(int sample_idx, int feat_idx)
   {
      return m_features[sample_idx * m_max_features + feat_idx];
   }

   void GetFeatures(int sample_idx, double &out[])
   {
      ArrayCopy(out, m_features, 0, sample_idx * m_max_features, m_max_features);
   }

   double GetLabel(int idx)  { return m_labels[idx]; }
   double GetWeight(int idx) { return m_weights[idx]; }
   int    Count()            { return m_count; }
   int    WritePos()         { return m_write_pos; }
   int    MaxFeatures()      { return m_max_features; }
   int    Capacity()         { return m_capacity; }

   void Reset()
   {
      m_count = 0;
      m_write_pos = 0;
   }
};

//===================================================================
// BIN MAPPER (Quantile binning for histogram-based models)
//===================================================================
class CMLBinMapper
{
private:
   double m_thresholds[];   // [n_features * n_thresholds]
   int    m_n_features;
   int    m_n_bins;          // number of bins (e.g. 63)
   int    m_n_thresholds;    // n_bins - 1 (e.g. 62)
   bool   m_computed;

public:
   CMLBinMapper() : m_n_features(0), m_n_bins(0), m_n_thresholds(0), m_computed(false) {}

   void Init(int n_features, int n_bins)
   {
      m_n_features = n_features;
      m_n_bins = n_bins;
      m_n_thresholds = n_bins - 1;
      ArrayResize(m_thresholds, n_features * m_n_thresholds);
      ArrayInitialize(m_thresholds, 0.0);
      m_computed = false;
   }

   void Compute(CMLTrainingBuffer &buffer)
   {
      int n = buffer.Count();
      if(n < 10) return;
      double vals[];
      ArrayResize(vals, n);

      for(int f = 0; f < m_n_features; f++)
      {
         for(int i = 0; i < n; i++)
            vals[i] = buffer.GetFeature(i, f);
         ArraySort(vals);

         int offset = f * m_n_thresholds;
         for(int t = 0; t < m_n_thresholds; t++)
         {
            double p = (double)(t + 1) / (double)m_n_bins;
            double pos = p * (n - 1);
            int lo = (int)MathFloor(pos);
            if(lo >= n - 1) lo = n - 2;
            double frac = pos - lo;
            m_thresholds[offset + t] = vals[lo] * (1.0 - frac) + vals[lo + 1] * frac;
         }
      }
      m_computed = true;
   }

   bool IsComputed() { return m_computed; }

   int GetBin(int feature, double value)
   {
      // NaN sentinel → dedicated NaN bin (one beyond normal bins)
      if(value <= ML_NAN_SENTINEL + 1e300) return m_n_bins;

      int lo = 0, hi = m_n_thresholds - 1;
      int offset = feature * m_n_thresholds;
      while(lo <= hi)
      {
         int mid = (lo + hi) / 2;
         if(value < m_thresholds[offset + mid])
            hi = mid - 1;
         else
            lo = mid + 1;
      }
      return lo; // bin in [0, n_bins-1]
   }

   int GetNaNBin() { return m_n_bins; }

   double GetBinThreshold(int feature, int bin)
   {
      if(bin >= m_n_thresholds) return DBL_MAX;
      return m_thresholds[feature * m_n_thresholds + bin];
   }

   int NBins() { return m_n_bins; }
   int NThresholds() { return m_n_thresholds; }

   bool Save(int handle)
   {
      FileWriteInteger(handle, m_n_features);
      FileWriteInteger(handle, m_n_bins);
      FileWriteInteger(handle, m_computed ? 1 : 0);
      if(m_computed)
         FileWriteArray(handle, m_thresholds, 0, m_n_features * m_n_thresholds);
      return true;
   }

   bool Load(int handle)
   {
      m_n_features = FileReadInteger(handle);
      m_n_bins = FileReadInteger(handle);
      m_n_thresholds = m_n_bins - 1;
      m_computed = (FileReadInteger(handle) != 0);
      ArrayResize(m_thresholds, m_n_features * m_n_thresholds);
      if(m_computed)
         FileReadArray(handle, m_thresholds, 0, m_n_features * m_n_thresholds);
      return true;
   }
};

//===================================================================
// TREE NODE (used during tree construction, not for inference)
//===================================================================
struct MLTreeNode
{
   bool   is_leaf;
   int    feature_idx;
   double threshold;
   double leaf_value;
   double grad_sum;
   double hess_sum;
   int    sample_count;
   int    left_child;
   int    right_child;
   double best_gain;
   int    best_feature;
   double best_threshold;
   int    best_bin;       // for histogram-based models
   int    hist_idx;       // index into histogram storage
   double nan_direction;  // 0.0 = NaN goes right (default), 1.0 = NaN goes left
   int    active_group;   // interaction constraint: -1=unconstrained, >=0=locked to group
};

//===================================================================
// CMLModelBase — Abstract base class
//===================================================================
class CMLModelBase
{
protected:
   CMLTrainingBuffer m_buffer;
   ENUM_ML_TASK      m_task;
   int               m_n_features;
   double            m_base_score;
   int               m_train_count;
   double            m_feature_importance[];
   double            m_best_val_loss;       // best validation loss seen
   int               m_no_improve_rounds;   // consecutive rounds without improvement
   bool              m_early_stopped;       // true if early stopping triggered
   int               m_monotone[];          // +1/0/-1 per feature for monotone constraints
   bool              m_has_monotone;        // true if any non-zero constraint exists
   int               m_interaction_groups[];   // group_id per feature (-1=unconstrained)
   int               m_n_interaction_groups;   // total distinct groups (0=disabled)
   bool              m_has_interaction;        // true when any group defined
   bool              m_is_categorical[];       // true for categorical features (stub)
   int               m_n_categorical;          // count of categorical features
   bool              m_has_categorical;        // true when any categorical feature exists

   void ComputeGradHess(double &preds[], double &grads[], double &hess[], int n)
   {
      for(int i = 0; i < n; i++)
      {
         double label = m_buffer.GetLabel(i);
         if(m_task == ML_TASK_QUANTILE)
         {
            grads[i] = ML_QuantileGrad(preds[i], label) * m_buffer.GetWeight(i);
            hess[i]  = ML_QuantileHess(preds[i], label) * m_buffer.GetWeight(i);
         }
         else if(m_task == ML_TASK_REGRESSION)
         {
            grads[i] = ML_HuberGrad(preds[i], label) * m_buffer.GetWeight(i);
            hess[i]  = ML_HuberHess(preds[i], label) * m_buffer.GetWeight(i);
         }
         else
         {
            grads[i] = ML_LoglossGrad(preds[i], label) * m_buffer.GetWeight(i);
            hess[i]  = ML_LoglossHess(preds[i], label) * m_buffer.GetWeight(i);
         }
      }
   }

   void ComputeBaseScore()
   {
      int n = m_buffer.Count();
      if(n == 0) { m_base_score = 0.0; return; }
      double sum = 0.0;
      for(int i = 0; i < n; i++) sum += m_buffer.GetLabel(i);
      double mean = sum / n;
      if(m_task == ML_TASK_CLASSIFICATION)
      {
         mean = MathMax(ML_EPS, MathMin(1.0 - ML_EPS, mean));
         m_base_score = MathLog(mean / (1.0 - mean)); // log-odds
      }
      else
         m_base_score = mean;
   }

   // Compute validation loss on samples [val_start, n)
   double ComputeValidationLoss(double &preds[], int val_start, int n)
   {
      double loss = 0.0;
      int count = n - val_start;
      if(count <= 0) return 1e30;
      for(int i = val_start; i < n; i++)
      {
         double label = m_buffer.GetLabel(i);
         if(m_task == ML_TASK_QUANTILE)
         {
            double r = preds[i] - label;
            loss += (r >= 0.0) ? ML_QuantileAlpha * r : (ML_QuantileAlpha - 1.0) * r;
         }
         else if(m_task == ML_TASK_REGRESSION)
         {
            double diff = preds[i] - label;
            loss += diff * diff;
         }
         else // classification — log loss
         {
            double p = 1.0 / (1.0 + MathExp(-preds[i]));
            p = MathMax(MathMin(p, 1.0 - 1e-15), 1e-15);
            loss -= label * MathLog(p) + (1.0 - label) * MathLog(1.0 - p);
         }
      }
      return loss / count;
   }

   // Check early stopping condition, returns true if should stop
   bool CheckEarlyStop(double &preds[], int n)
   {
      if(ML_EarlyStopRounds <= 0 || m_early_stopped) return m_early_stopped;
      int val_start = (int)(n * 0.8);
      if(val_start < 10 || n - val_start < 5) return false; // not enough data
      double val_loss = ComputeValidationLoss(preds, val_start, n);
      if(val_loss < m_best_val_loss - 1e-10)
      {
         m_best_val_loss = val_loss;
         m_no_improve_rounds = 0;
      }
      else
         m_no_improve_rounds++;
      if(m_no_improve_rounds >= ML_EarlyStopRounds)
      {
         m_early_stopped = true;
         Print("ML: Early stopping triggered after ", m_no_improve_rounds, " rounds without improvement");
         return true;
      }
      return false;
   }

public:
   CMLModelBase() : m_task(ML_TASK_REGRESSION), m_n_features(0), m_base_score(0.0), m_train_count(0),
                    m_best_val_loss(1e30), m_no_improve_rounds(0), m_early_stopped(false),
                    m_has_monotone(false), m_n_interaction_groups(0), m_has_interaction(false),
                    m_n_categorical(0), m_has_categorical(false) {}
   virtual ~CMLModelBase() {}

   void ParseMonotoneConstraints()
   {
      m_has_monotone = false;
      if(ML_MonotoneConstraints == "") return;
      ArrayResize(m_monotone, m_n_features);
      ArrayInitialize(m_monotone, 0);
      string parts[];
      int n = StringSplit(ML_MonotoneConstraints, ',', parts);
      for(int i = 0; i < n && i < m_n_features; i++)
      {
         StringTrimLeft(parts[i]);
         StringTrimRight(parts[i]);
         if(parts[i] == "+1" || parts[i] == "1") m_monotone[i] = +1;
         else if(parts[i] == "-1")               m_monotone[i] = -1;
         if(m_monotone[i] != 0) m_has_monotone = true;
      }
   }

   void ParseInteractionConstraints()
   {
      m_has_interaction = false;
      m_n_interaction_groups = 0;
      if(ML_InteractionConstraints == "") return;
      ArrayResize(m_interaction_groups, m_n_features);
      ArrayInitialize(m_interaction_groups, -1);  // -1 = unconstrained
      string groups[];
      int n_groups = StringSplit(ML_InteractionConstraints, ']', groups);
      int group_id = 0;
      for(int g = 0; g < n_groups; g++)
      {
         string tok = groups[g];
         int start = StringFind(tok, "[");
         if(start < 0) continue;
         tok = StringSubstr(tok, start + 1);
         string features[];
         int n_feats = StringSplit(tok, ',', features);
         if(n_feats == 0) continue;
         for(int fi = 0; fi < n_feats; fi++)
         {
            StringTrimLeft(features[fi]);
            StringTrimRight(features[fi]);
            if(features[fi] == "") continue;
            int feat_idx = (int)StringToInteger(features[fi]);
            if(feat_idx >= 0 && feat_idx < m_n_features)
               m_interaction_groups[feat_idx] = group_id;
         }
         group_id++;
         m_n_interaction_groups = group_id;
      }
      m_has_interaction = (m_n_interaction_groups > 0);
      if(m_has_interaction)
         Print("ML: ", m_n_interaction_groups, " interaction constraint groups parsed");
   }

   void ParseCategoricalFeatures()
   {
      m_has_categorical = false;
      m_n_categorical = 0;
      ArrayResize(m_is_categorical, m_n_features);
      ArrayInitialize(m_is_categorical, false);
      if(ML_CategoricalFeatures == "") return;
      string parts[];
      int n = StringSplit(ML_CategoricalFeatures, ',', parts);
      for(int i = 0; i < n; i++)
      {
         StringTrimLeft(parts[i]);
         StringTrimRight(parts[i]);
         if(parts[i] == "") continue;
         int fi = (int)StringToInteger(parts[i]);
         if(fi >= 0 && fi < m_n_features)
         {
            m_is_categorical[fi] = true;
            m_n_categorical++;
         }
      }
      m_has_categorical = (m_n_categorical > 0);
      if(m_has_categorical)
         Print("ML: ", m_n_categorical, " categorical features registered (stub: treated as numeric)");
   }

   void InitBase(ENUM_ML_TASK task, int n_features)
   {
      m_task = task;
      m_n_features = n_features;
      m_buffer.Init(ML_BufferCapacity, n_features);
      ArrayResize(m_feature_importance, n_features);
      ArrayInitialize(m_feature_importance, 0.0);
      ParseMonotoneConstraints();
      ParseInteractionConstraints();
      ParseCategoricalFeatures();
   }

   virtual double Predict(const double &features[], int count) { return m_base_score; }
   virtual void   AddSample(const double &features[], int count, double label, double weight = 1.0)
   {
      m_buffer.AddSample(features, count, label, weight);
   }
   virtual void   Train(bool force = false) {}
   virtual bool   SaveToFile(string filename) { return false; }
   virtual bool   LoadFromFile(string filename) { return false; }
   virtual void   GetFeatureImportance(double &importance[])
   {
      ArrayCopy(importance, m_feature_importance, 0, 0, m_n_features);
   }
   virtual void   ComputeSHAP(const double &features[], int count, double &phi[])
   {
      ArrayResize(phi, m_n_features + 1);
      ArrayInitialize(phi, 0.0);
      phi[m_n_features] = m_base_score;  // expected value
   }
   virtual void   Reset()
   {
      m_buffer.Reset();
      m_base_score = 0.0;
      m_train_count = 0;
      ArrayInitialize(m_feature_importance, 0.0);
      m_best_val_loss = 1e30;
      m_no_improve_rounds = 0;
      m_early_stopped = false;
   }

   bool IsTrained()     { return m_train_count > 0; }
   int  GetTrainCount() { return m_train_count; }
   int  GetSampleCount(){ return m_buffer.Count(); }
   ENUM_ML_TASK GetTask() { return m_task; }
};

//===================================================================
// TreeSHAP — Path-based SHAP attribution for flat V2 trees
// Used by LightGBM, XGBoost, and CatBoost lossguide trees
// (all share the same flat V2 node format: feat, thresh, right_off, nan_dir)
//
// Algorithm: Lundberg 2020 path-dependent polynomial-time algorithm.
// Uses 50/50 zero_frac assumption (interventional SHAP — no cover counts stored).
// Accurate feature ranking; absolute magnitudes are proportional to true SHAP values.
//===================================================================

// SHAPPath: fixed-size arrays (max depth 16) so struct copy works correctly in MQL5.
// Tracks the fraction of training-data paths through each feature as we recurse.
struct SHAPPath
{
   double feat[32];     // feature index at each path position (-1 = root sentinel)
   double zero[32];     // fraction of zero-paths (feature absent from this path)
   double one[32];      // fraction of one-paths (feature present on this path)
   double pweight[32];  // polynomial path weight at each position
   int    len;          // current path length (0 = empty)
};

//--------------------------------------------------------------------
// SHAP_Extend: extend the path by recording a new split edge.
// zf = zero-fraction, of = one-fraction, fi = feature index.
// Implements the "extend" recurrence from Lundberg 2020, Algorithm 2.
//--------------------------------------------------------------------
void SHAP_Extend(SHAPPath &p, double zf, double of, int fi)
{
   int d = p.len;
   if(d >= 31) return;   // safety cap — never exceed fixed array size

   p.feat[d]    = (double)fi;
   p.zero[d]    = zf;
   p.one[d]     = of;
   p.pweight[d] = (d == 0) ? 1.0 : 0.0;

   // Update weights for existing positions to account for the new feature
   for(int i = d - 1; i >= 0; i--)
   {
      p.pweight[i + 1] += of * p.pweight[i] * (double)(i + 1) / (double)(d + 1);
      p.pweight[i]     *= zf * (double)(d - i) / (double)(d + 1);
   }
   p.len++;
}

//--------------------------------------------------------------------
// SHAP_UnwoundWeight: compute the Shapley kernel weight for the feature
// at path position j.  Implements Lundberg (2020) Algorithm 2 unwind:
// reverses the SHAP_Extend recurrence for one_frac/zero_frac of feature j,
// accumulating the total permutation weight without that feature.
//--------------------------------------------------------------------
double SHAP_UnwoundWeight(const SHAPPath &p, int j)
{
   int D = p.len;
   if(D <= 1) return 0.0;

   double of  = p.one[j];
   double zf  = p.zero[j];
   double nxt = p.pweight[D - 1];
   double total = 0.0;

   for(int i = D - 2; i >= 0; i--)
   {
      if(of != 0.0)
      {
         double tmp = nxt * (double)D / ((double)(i + 1) * of);
         total += tmp;
         nxt = p.pweight[i] - tmp * zf * (double)(D - 1 - i) / (double)D;
      }
      else if(zf != 0.0)
      {
         total += (p.pweight[i] * (double)D) / (zf * (double)(D - 1 - i));
      }
      // else both zero — feature contributes nothing, skip
   }
   return total;
}

//--------------------------------------------------------------------
// SHAP_Recurse: core recursive tree traversal for a single flat V2 tree.
// off   = current node's offset in tree[]
// start = offset of the tree's root node (for right-child jumps)
// feats = sample feature values
// nf    = number of features
// phi[] = accumulator for SHAP attributions (size >= nf)
// m     = current path state (pass in by reference, modified and restored)
// pz/po = parent zero-frac / one-frac (use 1.0/1.0 for root call)
// pi    = parent feature index (use -1 for root call)
//--------------------------------------------------------------------
void SHAP_Recurse(const double &tree[], int off, int start,
                  const double &feats[], int nf, double &phi[],
                  SHAPPath &m, double pz, double po, int pi)
{
   SHAP_Extend(m, pz, po, pi);

   if(tree[off] < 0.0)
   {
      // ── LEAF NODE ──
      // Attribute the leaf value to each feature in the current path
      double val      = tree[off + 1];
      int    path_len = m.len;

      for(int j = 1; j < path_len; j++)   // skip j=0 (root sentinel, fi=-1)
      {
         double wj = SHAP_UnwoundWeight(m, j);
         int fi = (int)m.feat[j];
         if(fi >= 0 && fi < nf)
            phi[fi] += wj * (m.one[j] - m.zero[j]) * val;
      }

      m.len--;  // undo this node's SHAP_Extend call
      return;
   }

   // ── INTERNAL NODE — flat V2 layout: feat | thresh | right_off | nan_dir ──
   int    feat      = (int)tree[off];
   double thresh    = tree[off + 1];
   int    right_off = start + (int)tree[off + 2];
   double nan_dir   = tree[off + 3];
   int    left_off  = off + ML_NODE_WIDTH_V2;

   // Determine which child the sample falls into (hot) vs. counterfactual (cold)
   double fval      = (feat >= 0 && feat < nf) ? feats[feat] : 0.0;
   bool   is_nan    = (fval <= ML_NAN_SENTINEL + 1e300);
   bool   goes_left = is_nan ? (nan_dir > 0.5) : (fval < thresh);

   int hot_off  = goes_left ? left_off  : right_off;
   int cold_off = goes_left ? right_off : left_off;

   // 50/50 zero-fraction assumption (no cover counts stored in flat trees)
   double hz = 0.5, cz = 0.5;

   // Recurse into both subtrees — hot (sample actually goes here, one_frac=1)
   // and cold (counterfactual, one_frac=0)
   SHAP_Recurse(tree, hot_off,  start, feats, nf, phi, m, hz, 1.0, feat);
   SHAP_Recurse(tree, cold_off, start, feats, nf, phi, m, cz, 0.0, feat);

   m.len--;  // undo this node's SHAP_Extend call
}

//===================================================================
//===================================================================
// CLightGBM_Model — Leaf-wise histogram gradient boosted trees
//===================================================================
//===================================================================
class CLightGBM_Model : public CMLModelBase
{
private:
   // Ensemble storage
   double m_trees[];           // flat tree data (all trees concatenated)
   int    m_tree_offsets[];    // start offset per tree in m_trees
   int    m_tree_sizes[];      // node count per tree (in doubles)
   int    m_n_trees;
   int    m_trees_alloc;       // allocated size of m_trees
   double m_tree_weights[];    // per-tree weight (normally 1.0, DART rescales)

   // Histogram binning
   CMLBinMapper m_bins;

   // Working buffers (pre-allocated, reused)
   double m_preds[];
   double m_grads[];
   double m_hess[];
   int    m_selected_samples[];
   int    m_selected_features[];
   int    m_sample_leaf[];
   double m_hist_grad[];       // [max_leaves * n_sel_features * n_bins]
   double m_hist_hess[];
   int    m_hist_count[];
   MLTreeNode m_nodes[];

   // Incremental prediction tracking
   int  m_preds_buf_count;     // buffer count when preds last computed
   int  m_preds_buf_wpos;      // buffer write_pos when preds last computed

   // NaN direction tracking
   bool m_uses_v2_nodes;       // true if trees use V2 format (4 doubles/node with NaN dir)

   // Histogram indexing helpers
   int  m_n_sel_features;
   int  m_hist_stride_feat;    // n_bins + 1 (includes NaN bin)
   int  m_hist_stride_leaf;    // n_sel_features * m_hist_stride_feat

   // Helper: predict using the correct flat tree format (V1 or V2)
   double PredictTree(int tree_idx, const double &features[])
   {
      if(m_uses_v2_nodes)
         return ML_PredictFlatTreeV2(m_trees, m_tree_offsets[tree_idx], features, m_n_features);
      return ML_PredictFlatTree(m_trees, m_tree_offsets[tree_idx], features, m_n_features);
   }

   int HistIdx(int leaf, int feat_local, int bin)
   {
      return leaf * m_hist_stride_leaf + feat_local * m_hist_stride_feat + bin;
   }

   void ClearHistogram(int leaf)
   {
      int start = leaf * m_hist_stride_leaf;
      int count = m_hist_stride_leaf;
      ArrayFill(m_hist_grad, start, count, 0.0);
      ArrayFill(m_hist_hess, start, count, 0.0);
      ArrayFill(m_hist_count, start, count, 0);
   }

   void BuildLeafHistogram(int leaf_node_idx, int leaf_hist_idx)
   {
      ClearHistogram(leaf_hist_idx);
      int n = m_buffer.Count();
      for(int i = 0; i < n; i++)
      {
         if(m_sample_leaf[i] != leaf_node_idx) continue;
         for(int fl = 0; fl < m_n_sel_features; fl++)
         {
            int f = m_selected_features[fl];
            int bin = m_bins.GetBin(f, m_buffer.GetFeature(i, f));
            int idx = HistIdx(leaf_hist_idx, fl, bin);
            m_hist_grad[idx] += m_grads[i];
            m_hist_hess[idx] += m_hess[i];
            m_hist_count[idx]++;
         }
      }
   }

   void SubtractHistogram(int parent_hist, int smaller_hist, int larger_hist)
   {
      int count = m_hist_stride_leaf;
      int p_start = parent_hist * count;
      int s_start = smaller_hist * count;
      int l_start = larger_hist * count;
      for(int i = 0; i < count; i++)
      {
         m_hist_grad[l_start + i]  = m_hist_grad[p_start + i] - m_hist_grad[s_start + i];
         m_hist_hess[l_start + i]  = m_hist_hess[p_start + i] - m_hist_hess[s_start + i];
         m_hist_count[l_start + i] = m_hist_count[p_start + i] - m_hist_count[s_start + i];
      }
   }

   void FindBestSplit(int hist_idx, double total_grad, double total_hess,
                      int total_count, double &best_gain, int &best_feat_local,
                      int &best_bin, double &best_nan_dir, int active_group = -1)
   {
      best_gain = -1e30;
      best_feat_local = -1;
      best_bin = -1;
      best_nan_dir = ML_NAN_GOES_RIGHT;
      int nbins = m_bins.NBins();

      for(int fl = 0; fl < m_n_sel_features; fl++)
      {
         // Interaction constraint: skip features not in the active group
         if(m_has_interaction && active_group >= 0)
         {
            int feat_group = m_interaction_groups[m_selected_features[fl]];
            if(feat_group >= 0 && feat_group != active_group) continue;
         }
         double run_g = 0.0, run_h = 0.0;
         int run_c = 0;

         for(int b = 0; b < nbins - 1; b++) // can't split at last bin
         {
            int idx = HistIdx(hist_idx, fl, b);
            run_g += m_hist_grad[idx];
            run_h += m_hist_hess[idx];
            run_c += m_hist_count[idx];

            int right_c = total_count - run_c;
            if(run_c < ML_MinChildSamples || right_c < ML_MinChildSamples)
               continue;

            double right_h = total_hess - run_h;
            if(run_h < ML_MinChildWeight || right_h < ML_MinChildWeight)
               continue;

            double right_g = total_grad - run_g;
            double gain = ML_SplitGain(run_g, run_h, right_g, right_h,
                                        ML_L1Reg, ML_L2Reg, ML_Gamma);

            // Monotone constraint check
            if(m_has_monotone && gain > 0.0)
            {
               int feat_idx = m_selected_features[fl];
               int mc = m_monotone[feat_idx];
               if(mc != 0)
               {
                  double lv = ML_LeafValue(run_g, run_h, ML_L1Reg, ML_L2Reg);
                  double rv = ML_LeafValue(total_grad - run_g, total_hess - run_h, ML_L1Reg, ML_L2Reg);
                  if((mc == +1 && lv > rv + ML_EPS) || (mc == -1 && lv < rv - ML_EPS))
                     continue;
               }
            }

            if(gain > best_gain)
            {
               best_gain = gain;
               best_feat_local = fl;
               best_bin = b;
            }
         }
      }

      // Evaluate NaN direction for the winning feature
      if(best_feat_local >= 0)
      {
         int nan_bin = m_bins.GetNaNBin();
         int nan_idx = HistIdx(hist_idx, best_feat_local, nan_bin);
         double nan_g = m_hist_grad[nan_idx];
         double nan_h = m_hist_hess[nan_idx];
         int    nan_c = m_hist_count[nan_idx];
         if(nan_c > 0)
         {
            double left_g = 0.0, left_h = 0.0;
            for(int b = 0; b <= best_bin; b++)
            {
               int idx = HistIdx(hist_idx, best_feat_local, b);
               left_g += m_hist_grad[idx];
               left_h += m_hist_hess[idx];
            }
            double right_g = total_grad - left_g - nan_g;
            double right_h = total_hess - left_h - nan_h;
            double gain_left = ML_SplitGain(left_g + nan_g, left_h + nan_h,
                                             right_g, right_h,
                                             ML_L1Reg, ML_L2Reg, ML_Gamma);
            double gain_right = ML_SplitGain(left_g, left_h,
                                              right_g + nan_g, right_h + nan_h,
                                              ML_L1Reg, ML_L2Reg, ML_Gamma);
            best_nan_dir = (gain_left > gain_right) ? ML_NAN_GOES_LEFT : ML_NAN_GOES_RIGHT;
         }
      }
   }

   int SerializeTree(int node_idx, double &flat[], int &pos, int tree_start)
   {
      if(m_nodes[node_idx].is_leaf)
      {
         flat[pos++] = ML_LEAF_MARKER;
         flat[pos++] = m_nodes[node_idx].leaf_value;
         flat[pos++] = 0.0;  // padding for V2 leaf alignment
         flat[pos++] = 0.0;  // padding for V2 leaf alignment
         return pos;
      }
      flat[pos++] = (double)m_nodes[node_idx].feature_idx;
      flat[pos++] = m_nodes[node_idx].threshold;
      int right_offset_pos = pos;
      flat[pos++] = 0.0; // placeholder for right offset
      flat[pos++] = m_nodes[node_idx].nan_direction; // NaN direction

      // Left child (immediately after parent)
      pos = SerializeTree(m_nodes[node_idx].left_child, flat, pos, tree_start);

      // Right child
      flat[right_offset_pos] = (double)(pos - tree_start);
      pos = SerializeTree(m_nodes[node_idx].right_child, flat, pos, tree_start);
      return pos;
   }

   void AddTreeToEnsemble(int root_idx)
   {
      // Serialize tree to flat array (V2: 4 doubles per node)
      double flat[];
      ArrayResize(flat, ML_MAX_TREE_NODES * ML_NODE_WIDTH_V2);
      int pos = 0;
      int start = 0;
      SerializeTree(root_idx, flat, pos, start);
      int tree_size = pos;

      // Drop oldest tree if at capacity
      if(m_n_trees >= ML_MaxTrees)
      {
         // Subtract evicted tree contribution from all predictions
         int old_off = m_tree_offsets[0];
         int nn = m_buffer.Count();
         double evict_feat[];
         ArrayResize(evict_feat, m_n_features);
         for(int i = 0; i < nn; i++)
         {
            m_buffer.GetFeatures(i, evict_feat);
            double evict_pred = m_uses_v2_nodes
               ? ML_PredictFlatTreeV2(m_trees, old_off, evict_feat, m_n_features)
               : ML_PredictFlatTree(m_trees, old_off, evict_feat, m_n_features);
            m_preds[i] -= ML_LearningRate * m_tree_weights[0] * evict_pred;
         }

         int old_size = m_tree_sizes[0];
         // Shift tree data left
         ArrayCopy(m_trees, m_trees, 0, old_size, m_trees_alloc - old_size);
         // Update offsets and weights
         for(int t = 1; t < m_n_trees; t++)
         {
            m_tree_offsets[t - 1] = m_tree_offsets[t] - old_size;
            m_tree_sizes[t - 1] = m_tree_sizes[t];
         }
         ArrayCopy(m_tree_weights, m_tree_weights, 0, 1, m_n_trees);
         m_n_trees--;
         m_trees_alloc -= old_size;
      }

      // Ensure capacity
      if(m_trees_alloc + tree_size > ArraySize(m_trees))
         ArrayResize(m_trees, m_trees_alloc + tree_size + 1024);

      // Append new tree
      ArrayCopy(m_trees, flat, m_trees_alloc, 0, tree_size);
      m_tree_offsets[m_n_trees] = m_trees_alloc;
      m_tree_sizes[m_n_trees] = tree_size;
      m_tree_weights[m_n_trees] = 1.0;
      m_trees_alloc += tree_size;
      m_n_trees++;
   }

   // Accumulate feature importance from tree nodes
   void AccumulateImportance(int node_idx)
   {
      if(m_nodes[node_idx].is_leaf) return;
      int f = m_nodes[node_idx].feature_idx;
      if(f >= 0 && f < m_n_features)
         m_feature_importance[f] += m_nodes[node_idx].best_gain;
      AccumulateImportance(m_nodes[node_idx].left_child);
      AccumulateImportance(m_nodes[node_idx].right_child);
   }

   // GOSS: Gradient-based One-Side Sampling
   // Keeps top ML_GOSS_TopRate of samples by |gradient|, randomly samples ML_GOSS_OtherRate of rest
   void GOSS_Select(int n, int &selected[], int &n_selected)
   {
      // Fallback to all samples if GOSS rates invalid or too few samples
      if(ML_GOSS_TopRate <= 0 || ML_GOSS_OtherRate <= 0 ||
         ML_GOSS_TopRate + ML_GOSS_OtherRate >= 1.0 || n < 100)
      {
         n_selected = n;
         for(int i = 0; i < n; i++) selected[i] = i;
         return;
      }

      int n_top = (int)(n * ML_GOSS_TopRate);
      int n_other = (int)(n * ML_GOSS_OtherRate);
      if(n_top < 1) n_top = 1;
      if(n_other < 1) n_other = 1;

      // Sort indices by |gradient| descending
      double abs_grads[];
      int sort_idx[];
      ArrayResize(abs_grads, n);
      ArrayResize(sort_idx, n);
      for(int i = 0; i < n; i++)
      {
         abs_grads[i] = MathAbs(m_grads[i]);
         sort_idx[i] = i;
      }
      // Partial selection sort: only need the top n_top
      for(int i = 0; i < n - 1 && i < n_top; i++)
      {
         int max_idx = i;
         for(int j = i + 1; j < n; j++)
            if(abs_grads[sort_idx[j]] > abs_grads[sort_idx[max_idx]])
               max_idx = j;
         if(max_idx != i) { int tmp = sort_idx[i]; sort_idx[i] = sort_idx[max_idx]; sort_idx[max_idx] = tmp; }
      }

      // Take top n_top samples
      n_selected = 0;
      for(int i = 0; i < n_top; i++)
         selected[n_selected++] = sort_idx[i];

      // Random sample n_other from the rest [n_top, n)
      int rest_count = n - n_top;
      // Fisher-Yates partial shuffle on remaining indices
      for(int i = 0; i < n_other && i < rest_count; i++)
      {
         int j = n_top + ML_RandInt(rest_count - i);
         // swap sort_idx[n_top+i] and sort_idx[j]
         int tmp = sort_idx[n_top + i]; sort_idx[n_top + i] = sort_idx[j]; sort_idx[j] = tmp;
         selected[n_selected++] = sort_idx[n_top + i];
      }

      // Re-weight small-gradient samples: multiply their gradients/hessians by factor
      double weight_factor = (1.0 - ML_GOSS_TopRate) / ML_GOSS_OtherRate;
      for(int i = n_top; i < n_selected; i++)
      {
         int si = selected[i];
         m_grads[si] *= weight_factor;
         m_hess[si] *= weight_factor;
      }
   }

   // --- EFB: Exclusive Feature Bundling (stub) ---
   int  m_efb_bundles[];      // bundle_id per feature
   int  m_efb_offsets[];      // bin offset per feature within bundle
   int  m_n_efb_bundles;      // total bundles (= n_features for dense data)
   bool m_efb_computed;

   void ComputeEFBBundles()
   {
      if(!ML_EnableEFB || m_efb_computed) return;
      m_efb_computed = true;
      int n = m_buffer.Count();
      int nf = m_n_features;
      // Conflict detection: sample 200 rows to check mutual exclusivity
      int sample_n = MathMin(n, 200);
      int conflicts[];
      ArrayResize(conflicts, nf);
      ArrayInitialize(conflicts, 0);
      for(int i = 0; i < sample_n; i++)
      {
         int nonzero = 0;
         for(int a = 0; a < nf; a++)
         {
            double va = m_buffer.GetFeature(i, a);
            if(MathAbs(va) > ML_EPS && !(va <= ML_NAN_SENTINEL + 1e300))
               nonzero++;
         }
         // Dense features: most are nonzero → high conflict
         if(nonzero > nf / 2)
            for(int a = 0; a < nf; a++) conflicts[a]++;
      }
      // Greedy bundling: with dense features, each feature becomes its own bundle
      ArrayResize(m_efb_bundles, nf);
      ArrayResize(m_efb_offsets, nf);
      m_n_efb_bundles = 0;
      for(int f = 0; f < nf; f++)
      {
         m_efb_bundles[f] = m_n_efb_bundles;
         m_efb_offsets[f] = 0;
         m_n_efb_bundles++;
      }
      Print("ML EFB: ", m_n_efb_bundles, " bundles for ", nf, " features",
            (m_n_efb_bundles == nf) ? " (all singletons — dense data)" : "");
   }

   // --- Linear Trees (stub) ---
   bool m_linear_leaves_active;

public:
   CLightGBM_Model() : m_n_trees(0), m_trees_alloc(0), m_n_sel_features(0),
                        m_hist_stride_feat(0), m_hist_stride_leaf(0),
                        m_uses_v2_nodes(false), m_efb_computed(false),
                        m_n_efb_bundles(0), m_linear_leaves_active(false) {}
   ~CLightGBM_Model() {}

   void Init(ENUM_ML_TASK task, int n_features)
   {
      InitBase(task, n_features);
      m_n_trees = 0;
      m_trees_alloc = 0;
      m_bins.Init(n_features, ML_NBins);

      int max_samples = ML_BufferCapacity;
      ArrayResize(m_trees, 4096);
      ArrayResize(m_tree_offsets, ML_MaxTrees + 10);
      ArrayResize(m_tree_sizes, ML_MaxTrees + 10);
      ArrayResize(m_preds, max_samples);
      ArrayResize(m_grads, max_samples);
      ArrayResize(m_hess, max_samples);
      ArrayResize(m_selected_samples, max_samples);
      ArrayResize(m_selected_features, n_features);
      ArrayResize(m_sample_leaf, max_samples);
      ArrayResize(m_nodes, ML_MAX_TREE_NODES);
      ArrayResize(m_tree_weights, ML_MaxTrees + 10);
      ArrayInitialize(m_tree_weights, 1.0);

      m_preds_buf_count = 0;
      m_preds_buf_wpos = 0;

      // Histogram buffers allocated when bins are computed
      ML_SeedRng((ulong)GetTickCount());
   }

   virtual double Predict(const double &features[], int count)
   {
      double pred = m_base_score;
      if(m_uses_v2_nodes)
      {
         for(int t = 0; t < m_n_trees; t++)
            pred += ML_LearningRate * m_tree_weights[t] * ML_PredictFlatTreeV2(m_trees, m_tree_offsets[t],
                                                          features, m_n_features);
      }
      else
      {
         for(int t = 0; t < m_n_trees; t++)
            pred += ML_LearningRate * m_tree_weights[t] * ML_PredictFlatTree(m_trees, m_tree_offsets[t],
                                                          features, m_n_features);
      }
      return pred;
   }

   virtual void Train(bool force = false)
   {
      int n = m_buffer.Count();
      if(!force && n < ML_ColdStartMin) return;
      if(m_early_stopped) return;  // Already stopped

      // Compute bin edges on first training
      if(!m_bins.IsComputed())
      {
         m_bins.Compute(m_buffer);
         if(!m_bins.IsComputed()) return;
         // EFB stub: compute bundles after bins
         ComputeEFBBundles();
         // Linear tree gate
         m_linear_leaves_active = ML_EnableLinearLeaves &&
            (int)(m_n_features * ML_ColsampleTree) <= ML_LinearTreeMaxFeatures;
         if(m_linear_leaves_active)
            Print("ML LinearLeaves: active but OLS solver not yet implemented, using mean leaf values");
      }

      // Allocate histogram buffers if not yet done (first train or after load from file)
      if(m_hist_stride_feat == 0)
      {
         m_n_sel_features = (int)(m_n_features * ML_ColsampleTree);
         if(m_n_sel_features < 1) m_n_sel_features = 1;
         m_hist_stride_feat = m_bins.NBins() + 1;  // +1 for NaN bin
         m_hist_stride_leaf = m_n_sel_features * m_hist_stride_feat;
         int max_leaves = ML_MaxLeaves + 2;
         int hist_total = max_leaves * m_hist_stride_leaf;
         ArrayResize(m_hist_grad, hist_total);
         ArrayResize(m_hist_hess, hist_total);
         ArrayResize(m_hist_count, hist_total);
      }

      // Compute base score on first training
      if(m_train_count == 0) ComputeBaseScore();

      // Feature buffer used by incremental update, eviction, and per-round update
      double features[];
      ArrayResize(features, m_n_features);

      // Incremental prediction update — only recompute dirty (new/overwritten) samples
      if(m_n_trees == 0 || m_preds_buf_count == 0)
      {
         // Cold start or first training: recompute all from scratch
         for(int i = 0; i < n; i++)
         {
            m_preds[i] = m_base_score;
            m_buffer.GetFeatures(i, features);
            for(int t = 0; t < m_n_trees; t++)
               m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
         }
      }
      else
      {
         // Identify dirty sample range
         int dirty_start = -1, dirty_end = -1;
         bool wrap = false;
         if(n > m_preds_buf_count)
         {
            // Buffer grew: new samples at [old_count, n)
            dirty_start = m_preds_buf_count;
            dirty_end   = n;
         }
         else
         {
            // Buffer full, ring-wrapped: dirty = [old_wpos, new_wpos) with wraparound
            int new_wpos = m_buffer.WritePos();
            if(new_wpos != m_preds_buf_wpos)
            {
               dirty_start = m_preds_buf_wpos;
               dirty_end   = new_wpos;
               wrap = (dirty_end <= dirty_start);
            }
         }

         // Recompute only dirty samples
         if(dirty_start >= 0)
         {
            if(!wrap)
            {
               for(int i = dirty_start; i < dirty_end; i++)
               {
                  m_preds[i] = m_base_score;
                  m_buffer.GetFeatures(i, features);
                  for(int t = 0; t < m_n_trees; t++)
                     m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
               }
            }
            else
            {
               // Wraparound: [dirty_start, n) then [0, dirty_end)
               for(int i = dirty_start; i < n; i++)
               {
                  m_preds[i] = m_base_score;
                  m_buffer.GetFeatures(i, features);
                  for(int t = 0; t < m_n_trees; t++)
                     m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
               }
               for(int i = 0; i < dirty_end; i++)
               {
                  m_preds[i] = m_base_score;
                  m_buffer.GetFeatures(i, features);
                  for(int t = 0; t < m_n_trees; t++)
                     m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
               }
            }
         }
      }
      m_preds_buf_count = n;
      m_preds_buf_wpos  = m_buffer.WritePos();

      // Compute gradients/hessians
      ComputeGradHess(m_preds, m_grads, m_hess, n);

      // Row subsampling — use GOSS if rates are set, else uniform
      int n_sub;
      if(ML_GOSS_TopRate > 0 && ML_GOSS_OtherRate > 0 &&
         ML_GOSS_TopRate + ML_GOSS_OtherRate < 1.0 && n >= 100)
      {
         GOSS_Select(n, m_selected_samples, n_sub);
      }
      else
      {
         n_sub = (int)(n * ML_Subsample);
         if(n_sub < ML_MinChildSamples) n_sub = n;
         for(int i = 0; i < n; i++) m_selected_samples[i] = i;
         if(n_sub < n) ML_Shuffle(m_selected_samples, n);
      }

      // Column subsampling (per tree, applied below)
      int n_col = (int)(m_n_features * ML_ColsampleTree);
      if(n_col < 1) n_col = 1;
      m_n_sel_features = n_col;
      // Recalculate histogram strides
      m_hist_stride_leaf = m_n_sel_features * m_hist_stride_feat;

      // Build new trees
      for(int round = 0; round < ML_TreesPerRound; round++)
      {
         // Column selection for this tree
         int all_feats[];
         ArrayResize(all_feats, m_n_features);
         for(int i = 0; i < m_n_features; i++) all_feats[i] = i;
         ML_Shuffle(all_feats, m_n_features);
         ArrayCopy(m_selected_features, all_feats, 0, 0, n_col);

         // DART: drop random trees before building
         int dart_dropped[];
         int n_dart_dropped = 0;
         if(ML_DARTEnabled && m_n_trees > 0)
         {
            int n_drop = MathMax(1, (int)(m_n_trees * ML_DARTDropRate));
            if(n_drop > m_n_trees) n_drop = m_n_trees;

            int all_indices[];
            ArrayResize(all_indices, m_n_trees);
            for(int di = 0; di < m_n_trees; di++) all_indices[di] = di;
            ML_Shuffle(all_indices, m_n_trees);

            ArrayResize(dart_dropped, n_drop);
            n_dart_dropped = n_drop;
            for(int di = 0; di < n_drop; di++) dart_dropped[di] = all_indices[di];

            // Subtract dropped trees from predictions
            double feats_dart[];
            ArrayResize(feats_dart, m_n_features);
            for(int di = 0; di < n_drop; di++)
            {
               int dt = dart_dropped[di];
               double w = m_tree_weights[dt];
               for(int si = 0; si < n; si++)
               {
                  m_buffer.GetFeatures(si, feats_dart);
                  m_preds[si] -= ML_LearningRate * w * PredictTree(dt, feats_dart);
               }
            }

            // Recompute gradients on modified predictions
            ComputeGradHess(m_preds, m_grads, m_hess, n);
         }

         // Mark V2 nodes active (new trees always use V2 format)
         m_uses_v2_nodes = true;

         BuildTreeLeafWise(n_sub);

         // Update predictions with new tree
         int last_tree = m_n_trees - 1;
         for(int i = 0; i < n; i++)
         {
            m_buffer.GetFeatures(i, features);
            m_preds[i] += ML_LearningRate * PredictTree(last_tree, features);
         }

         // DART: restore dropped trees with rescaled weights
         if(n_dart_dropped > 0)
         {
            int new_tree = m_n_trees - 1;
            double scale_existing = (double)n_dart_dropped / (double)(n_dart_dropped + 1);
            double scale_new = 1.0 / (double)(n_dart_dropped + 1);

            double feats_dart[];
            ArrayResize(feats_dart, m_n_features);
            for(int di = 0; di < n_dart_dropped; di++)
            {
               int dt = dart_dropped[di];
               m_tree_weights[dt] *= scale_existing;
               for(int si = 0; si < n; si++)
               {
                  m_buffer.GetFeatures(si, feats_dart);
                  m_preds[si] += ML_LearningRate * m_tree_weights[dt] * PredictTree(dt, feats_dart);
               }
            }

            m_tree_weights[new_tree] = scale_new;
            // Adjust new tree prediction: was added with weight 1.0, need scale_new
            for(int si = 0; si < n; si++)
            {
               m_buffer.GetFeatures(si, feats_dart);
               double tp = PredictTree(new_tree, feats_dart);
               m_preds[si] -= ML_LearningRate * (1.0 - scale_new) * tp;
            }
         }

         // Recompute gradients for next tree in this round
         if(round < ML_TreesPerRound - 1)
            ComputeGradHess(m_preds, m_grads, m_hess, n);
      }

      // Check early stopping
      CheckEarlyStop(m_preds, n);

      m_train_count++;
   }

   void BuildTreeLeafWise(int n_samples)
   {
      // Init: all selected samples in root leaf
      ArrayInitialize(m_sample_leaf, -1);
      double root_g = 0.0, root_h = 0.0;
      int root_count = 0;
      for(int i = 0; i < n_samples; i++)
      {
         int si = m_selected_samples[i];
         m_sample_leaf[si] = 0;
         root_g += m_grads[si];
         root_h += m_hess[si];
         root_count++;
      }

      // Create root node
      int n_nodes = 0;
      ZeroMemory(m_nodes[0]);
      m_nodes[0].is_leaf = true;
      m_nodes[0].active_group = -1;
      m_nodes[0].grad_sum = root_g;
      m_nodes[0].hess_sum = root_h;
      m_nodes[0].sample_count = root_count;
      m_nodes[0].best_gain = -1e30;
      m_nodes[0].hist_idx = 0;
      n_nodes = 1;

      // Build root histogram and find best split
      BuildLeafHistogram(0, 0);
      double bg; int bf, bb; double nan_dir;
      FindBestSplit(0, root_g, root_h, root_count, bg, bf, bb, nan_dir, -1);
      m_nodes[0].best_gain = bg;
      m_nodes[0].best_feature = (bf >= 0) ? m_selected_features[bf] : -1;
      m_nodes[0].best_bin = bb;
      m_nodes[0].nan_direction = nan_dir;
      if(bf >= 0)
         m_nodes[0].best_threshold = m_bins.GetBinThreshold(m_selected_features[bf], bb);

      // Track which hist slots are in use (for reuse)
      int next_hist_slot = 1;
      int n_leaves = 1;
      int max_leaves = ML_MaxLeaves;

      while(n_leaves < max_leaves && n_nodes < ML_MAX_TREE_NODES - 2)
      {
         // Find leaf with best gain
         int best_leaf = -1;
         double best_leaf_gain = -1e30;
         for(int i = 0; i < n_nodes; i++)
         {
            if(!m_nodes[i].is_leaf) continue;
            if(m_nodes[i].sample_count < 2 * ML_MinChildSamples) continue;
            if(m_nodes[i].best_gain > best_leaf_gain && m_nodes[i].best_gain > 0)
            {
               best_leaf_gain = m_nodes[i].best_gain;
               best_leaf = i;
            }
         }
         if(best_leaf < 0) break;

         // Split this leaf
         int L = best_leaf;
         int left_idx = n_nodes++;
         int right_idx = n_nodes++;

         m_nodes[L].is_leaf = false;
         m_nodes[L].feature_idx = m_nodes[L].best_feature;
         m_nodes[L].threshold = m_nodes[L].best_threshold;
         m_nodes[L].left_child = left_idx;
         m_nodes[L].right_child = right_idx;

         // Count children from histogram
         int split_feat_local = -1;
         for(int fl = 0; fl < m_n_sel_features; fl++)
            if(m_selected_features[fl] == m_nodes[L].feature_idx)
            { split_feat_local = fl; break; }

         double left_g = 0.0, left_h = 0.0;
         int left_count = 0;
         int nbins = m_bins.NBins();
         int parent_hist = m_nodes[L].hist_idx;
         for(int b = 0; b <= m_nodes[L].best_bin; b++)
         {
            int idx = HistIdx(parent_hist, split_feat_local, b);
            left_g += m_hist_grad[idx];
            left_h += m_hist_hess[idx];
            left_count += m_hist_count[idx];
         }
         // Add NaN bin to the correct side based on nan_direction
         int nan_idx_ch = HistIdx(parent_hist, split_feat_local, m_bins.GetNaNBin());
         double nan_g_ch = m_hist_grad[nan_idx_ch];
         double nan_h_ch = m_hist_hess[nan_idx_ch];
         int    nan_c_ch = m_hist_count[nan_idx_ch];
         if(m_nodes[L].nan_direction > 0.5)  // NaN goes left
         {
            left_g += nan_g_ch;
            left_h += nan_h_ch;
            left_count += nan_c_ch;
         }
         double right_g = m_nodes[L].grad_sum - left_g;
         double right_h = m_nodes[L].hess_sum - left_h;
         int right_count = m_nodes[L].sample_count - left_count;

         // Compute child active_group for interaction constraints
         int child_group = m_nodes[L].active_group;
         if(m_has_interaction && child_group < 0 && m_nodes[L].feature_idx >= 0)
         {
            int fg = m_interaction_groups[m_nodes[L].feature_idx];
            if(fg >= 0) child_group = fg;
         }

         // Init child nodes
         ZeroMemory(m_nodes[left_idx]);
         m_nodes[left_idx].is_leaf = true;
         m_nodes[left_idx].grad_sum = left_g;
         m_nodes[left_idx].hess_sum = left_h;
         m_nodes[left_idx].sample_count = left_count;
         m_nodes[left_idx].best_gain = -1e30;
         m_nodes[left_idx].active_group = child_group;

         ZeroMemory(m_nodes[right_idx]);
         m_nodes[right_idx].is_leaf = true;
         m_nodes[right_idx].grad_sum = right_g;
         m_nodes[right_idx].hess_sum = right_h;
         m_nodes[right_idx].sample_count = right_count;
         m_nodes[right_idx].best_gain = -1e30;
         m_nodes[right_idx].active_group = child_group;

         // Determine smaller child
         bool left_smaller = (left_count <= right_count);
         int smaller_node = left_smaller ? left_idx : right_idx;
         int larger_node = left_smaller ? right_idx : left_idx;
         int smaller_hist = next_hist_slot++;
         int larger_hist = next_hist_slot++;
         m_nodes[smaller_node].hist_idx = smaller_hist;
         m_nodes[larger_node].hist_idx = larger_hist;

         // Ensure histogram buffer is large enough
         int needed = next_hist_slot * m_hist_stride_leaf;
         if(needed > ArraySize(m_hist_grad))
         {
            ArrayResize(m_hist_grad, needed + m_hist_stride_leaf);
            ArrayResize(m_hist_hess, needed + m_hist_stride_leaf);
            ArrayResize(m_hist_count, needed + m_hist_stride_leaf);
         }

         // Reassign samples and build smaller child histogram
         ClearHistogram(smaller_hist);
         int n_buf = m_buffer.Count();
         for(int i = 0; i < n_buf; i++)
         {
            if(m_sample_leaf[i] != L) continue;
            double fval = m_buffer.GetFeature(i, m_nodes[L].feature_idx);
            bool is_nan = (fval <= ML_NAN_SENTINEL + 1e300);
            bool go_left = is_nan ? (m_nodes[L].nan_direction > 0.5)
                                  : (fval < m_nodes[L].threshold);
            if(go_left)
               m_sample_leaf[i] = left_idx;
            else
               m_sample_leaf[i] = right_idx;

            // Build histogram for smaller child
            if(m_sample_leaf[i] == smaller_node)
            {
               for(int fl = 0; fl < m_n_sel_features; fl++)
               {
                  int f = m_selected_features[fl];
                  int bin = m_bins.GetBin(f, m_buffer.GetFeature(i, f));
                  int hidx = HistIdx(smaller_hist, fl, bin);
                  m_hist_grad[hidx] += m_grads[i];
                  m_hist_hess[hidx] += m_hess[i];
                  m_hist_count[hidx]++;
               }
            }
         }

         // Histogram subtraction for larger child
         SubtractHistogram(parent_hist, smaller_hist, larger_hist);

         // Find best splits for children
         if(left_count >= ML_MinChildSamples)
         {
            FindBestSplit(m_nodes[left_idx].hist_idx, left_g, left_h, left_count,
                          bg, bf, bb, nan_dir, m_nodes[left_idx].active_group);
            m_nodes[left_idx].best_gain = bg;
            m_nodes[left_idx].best_feature = (bf >= 0) ? m_selected_features[bf] : -1;
            m_nodes[left_idx].best_bin = bb;
            m_nodes[left_idx].nan_direction = nan_dir;
            if(bf >= 0)
               m_nodes[left_idx].best_threshold =
                  m_bins.GetBinThreshold(m_selected_features[bf], bb);
         }
         if(right_count >= ML_MinChildSamples)
         {
            FindBestSplit(m_nodes[right_idx].hist_idx, right_g, right_h, right_count,
                          bg, bf, bb, nan_dir, m_nodes[right_idx].active_group);
            m_nodes[right_idx].best_gain = bg;
            m_nodes[right_idx].best_feature = (bf >= 0) ? m_selected_features[bf] : -1;
            m_nodes[right_idx].best_bin = bb;
            m_nodes[right_idx].nan_direction = nan_dir;
            if(bf >= 0)
               m_nodes[right_idx].best_threshold =
                  m_bins.GetBinThreshold(m_selected_features[bf], bb);
         }

         n_leaves++;
      }

      // Compute leaf values
      for(int i = 0; i < n_nodes; i++)
         if(m_nodes[i].is_leaf)
            m_nodes[i].leaf_value = ML_LeafValue(m_nodes[i].grad_sum, m_nodes[i].hess_sum,
                                                  ML_L1Reg, ML_L2Reg);

      // Accumulate feature importance
      AccumulateImportance(0);

      // Serialize and add to ensemble
      AddTreeToEnsemble(0);
   }

   virtual bool SaveToFile(string filename)
   {
      int handle = FileOpen(filename, FILE_WRITE | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE) return false;

      FileWriteInteger(handle, ML_MAGIC_SAVE);
      FileWriteInteger(handle, (int)ML_LIGHTGBM);
      FileWriteInteger(handle, (int)m_task);
      FileWriteInteger(handle, m_n_features);
      FileWriteInteger(handle, m_n_trees);
      FileWriteDouble(handle, m_base_score);
      FileWriteInteger(handle, m_train_count);
      FileWriteInteger(handle, m_trees_alloc);

      // Tree data
      if(m_trees_alloc > 0)
         FileWriteArray(handle, m_trees, 0, m_trees_alloc);
      FileWriteArray(handle, m_tree_offsets, 0, m_n_trees);
      FileWriteArray(handle, m_tree_sizes, 0, m_n_trees);

      // Feature importance
      FileWriteArray(handle, m_feature_importance, 0, m_n_features);

      // Bin mapper
      m_bins.Save(handle);

      // GBT V2 extension block — early stopping state
      FileWriteInteger(handle, ML_MAGIC_GBT_V2);
      FileWriteDouble(handle, m_best_val_loss);
      FileWriteInteger(handle, m_no_improve_rounds);
      FileWriteInteger(handle, m_early_stopped ? 1 : 0);

      // DART weights
      FileWriteInteger(handle, ML_MAGIC_DART);
      FileWriteArray(handle, m_tree_weights, 0, m_n_trees);

      // GBT V3: NaN direction per node (V2 flat tree format)
      FileWriteInteger(handle, ML_MAGIC_GBT_V3);
      FileWriteInteger(handle, m_uses_v2_nodes ? 1 : 0);

      // GBT V4: reserved for future V4 extensions
      FileWriteInteger(handle, ML_MAGIC_V4);
      FileWriteInteger(handle, 0);  // reserved

      FileWriteInteger(handle, ML_MAGIC_END);
      FileClose(handle);
      return true;
   }

   virtual bool LoadFromFile(string filename)
   {
      int handle = FileOpen(filename, FILE_READ | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE) return false;

      int magic = FileReadInteger(handle);
      if(magic != ML_MAGIC_SAVE) { FileClose(handle); return false; }
      int model_type = FileReadInteger(handle);
      if(model_type != (int)ML_LIGHTGBM) { FileClose(handle); return false; }

      m_task = (ENUM_ML_TASK)FileReadInteger(handle);
      m_n_features = FileReadInteger(handle);
      m_n_trees = FileReadInteger(handle);
      m_base_score = FileReadDouble(handle);
      m_train_count = FileReadInteger(handle);
      m_trees_alloc = FileReadInteger(handle);

      ArrayResize(m_trees, MathMax(m_trees_alloc, 1));
      if(m_trees_alloc > 0)
         FileReadArray(handle, m_trees, 0, m_trees_alloc);
      ArrayResize(m_tree_offsets, ML_MaxTrees + 10);
      ArrayResize(m_tree_sizes, ML_MaxTrees + 10);
      FileReadArray(handle, m_tree_offsets, 0, m_n_trees);
      FileReadArray(handle, m_tree_sizes, 0, m_n_trees);

      ArrayResize(m_feature_importance, m_n_features);
      FileReadArray(handle, m_feature_importance, 0, m_n_features);

      m_bins.Load(handle);

      // Try reading GBT V2 extension block
      int next_magic = FileReadInteger(handle);
      if(next_magic == ML_MAGIC_GBT_V2)
      {
         m_best_val_loss = FileReadDouble(handle);
         m_no_improve_rounds = FileReadInteger(handle);
         m_early_stopped = (FileReadInteger(handle) != 0);
         next_magic = FileReadInteger(handle);
      }
      else
      {
         // Old format — use defaults
         m_best_val_loss = 1e30;
         m_no_improve_rounds = 0;
         m_early_stopped = false;
      }
      // Try reading DART weights
      if(next_magic == ML_MAGIC_DART)
      {
         ArrayResize(m_tree_weights, ML_MaxTrees + 10);
         FileReadArray(handle, m_tree_weights, 0, m_n_trees);
         next_magic = FileReadInteger(handle);
      }
      else
      {
         ArrayResize(m_tree_weights, ML_MaxTrees + 10);
         ArrayInitialize(m_tree_weights, 1.0);
      }
      // Try reading GBT V3 (NaN direction)
      if(next_magic == ML_MAGIC_GBT_V3)
      {
         m_uses_v2_nodes = (FileReadInteger(handle) != 0);
         next_magic = FileReadInteger(handle);
      }
      else
      {
         m_uses_v2_nodes = false;  // old file, trees use V1 format
      }
      // Try reading GBT V4 (reserved — LightGBM)
      if(next_magic == ML_MAGIC_V4)
      {
         int v4_reserved = FileReadInteger(handle);
         next_magic = FileReadInteger(handle);
      }

      // Always reset incremental preds (buffer is not saved)
      m_preds_buf_count = 0;
      m_preds_buf_wpos = 0;

      int end_magic = next_magic;
      FileClose(handle);
      return (end_magic == ML_MAGIC_END);
   }

   virtual void Reset()
   {
      CMLModelBase::Reset();
      m_n_trees = 0;
      m_trees_alloc = 0;
      m_uses_v2_nodes = false;
      ArrayInitialize(m_tree_weights, 1.0);
   }

   // Compute per-feature SHAP values for a single sample.
   // phi[] is sized (n_features + 1): phi[0..n-1] = feature attributions,
   // phi[n_features] = base score (expected model output = intercept).
   // Uses SHAP_Recurse over V2 flat trees; falls back to simple gain split
   // for V1 trees (no NaN direction, but same node layout minus nan_dir field).
   virtual void ComputeSHAP(const double &features[], int count, double &phi[])
   {
      ArrayResize(phi, m_n_features + 1);
      ArrayInitialize(phi, 0.0);
      phi[m_n_features] = m_base_score;

      if(m_n_trees == 0 || !m_uses_v2_nodes) return;  // V1 trees: skip (no SHAP support)

      SHAPPath path;
      double tree_phi[];
      ArrayResize(tree_phi, m_n_features);

      for(int t = 0; t < m_n_trees; t++)
      {
         ArrayInitialize(tree_phi, 0.0);
         ZeroMemory(path);
         path.len = 0;

         int root = m_tree_offsets[t];
         SHAP_Recurse(m_trees, root, root, features, m_n_features,
                      tree_phi, path, 1.0, 1.0, -1);

         double scale = ML_LearningRate * m_tree_weights[t];
         for(int f = 0; f < m_n_features; f++)
            phi[f] += scale * tree_phi[f];
      }
   }
};

//===================================================================
//===================================================================
// CXGBoost_Model — Level-wise exact-split gradient boosted trees
//===================================================================
//===================================================================
class CXGBoost_Model : public CMLModelBase
{
private:
   // Ensemble storage (same flat format as LightGBM)
   double m_trees[];
   int    m_tree_offsets[];
   int    m_tree_sizes[];
   int    m_n_trees;
   int    m_trees_alloc;
   double m_tree_weights[];    // per-tree weight (normally 1.0, DART rescales)

   // Working buffers
   double m_preds[];
   double m_grads[];
   double m_hess[];
   int    m_selected_features[];
   int    m_sample_leaf[];       // which leaf each sample belongs to during building
   MLTreeNode m_nodes[];
   double m_sort_vals[];         // for argsort
   int    m_sort_idx[];
   int    m_presorted[];         // [n_features * capacity] pre-sorted feature indices

   // Incremental prediction tracking
   int  m_preds_buf_count;
   int  m_preds_buf_wpos;

   // NaN direction tracking
   bool m_uses_v2_nodes;       // true if trees use V2 format (4 doubles/node with NaN dir)

   // Helper: predict using the correct flat tree format (V1 or V2)
   double PredictTree(int tree_idx, const double &features[])
   {
      if(m_uses_v2_nodes)
         return ML_PredictFlatTreeV2(m_trees, m_tree_offsets[tree_idx], features, m_n_features);
      return ML_PredictFlatTree(m_trees, m_tree_offsets[tree_idx], features, m_n_features);
   }

   int SerializeTree(int node_idx, double &flat[], int &pos, int tree_start)
   {
      if(m_nodes[node_idx].is_leaf)
      {
         flat[pos++] = ML_LEAF_MARKER;
         flat[pos++] = m_nodes[node_idx].leaf_value;
         flat[pos++] = 0.0;  // padding for V2 leaf alignment
         flat[pos++] = 0.0;  // padding for V2 leaf alignment
         return pos;
      }
      flat[pos++] = (double)m_nodes[node_idx].feature_idx;
      flat[pos++] = m_nodes[node_idx].threshold;
      int right_offset_pos = pos;
      flat[pos++] = 0.0; // placeholder for right offset
      flat[pos++] = m_nodes[node_idx].nan_direction; // NaN direction
      pos = SerializeTree(m_nodes[node_idx].left_child, flat, pos, tree_start);
      flat[right_offset_pos] = (double)(pos - tree_start);
      pos = SerializeTree(m_nodes[node_idx].right_child, flat, pos, tree_start);
      return pos;
   }

   void AddTreeToEnsemble(int root_idx)
   {
      double flat[];
      ArrayResize(flat, ML_MAX_TREE_NODES * ML_NODE_WIDTH_V2);
      int pos = 0;
      SerializeTree(root_idx, flat, pos, 0);
      int tree_size = pos;

      if(m_n_trees >= ML_MaxTrees)
      {
         // Subtract evicted tree from predictions
         int evict_off = m_tree_offsets[0];
         int n_buf = m_buffer.Count();
         double evict_feat[];
         ArrayResize(evict_feat, m_n_features);
         for(int ei = 0; ei < n_buf; ei++)
         {
            m_buffer.GetFeatures(ei, evict_feat);
            double evict_pred = m_uses_v2_nodes
               ? ML_PredictFlatTreeV2(m_trees, evict_off, evict_feat, m_n_features)
               : ML_PredictFlatTree(m_trees, evict_off, evict_feat, m_n_features);
            m_preds[ei] -= ML_LearningRate * m_tree_weights[0] * evict_pred;
         }

         int old_size = m_tree_sizes[0];
         ArrayCopy(m_trees, m_trees, 0, old_size, m_trees_alloc - old_size);
         for(int t = 1; t < m_n_trees; t++)
         {
            m_tree_offsets[t - 1] = m_tree_offsets[t] - old_size;
            m_tree_sizes[t - 1] = m_tree_sizes[t];
         }
         ArrayCopy(m_tree_weights, m_tree_weights, 0, 1, m_n_trees);
         m_n_trees--;
         m_trees_alloc -= old_size;
      }

      if(m_trees_alloc + tree_size > ArraySize(m_trees))
         ArrayResize(m_trees, m_trees_alloc + tree_size + 1024);

      ArrayCopy(m_trees, flat, m_trees_alloc, 0, tree_size);
      m_tree_offsets[m_n_trees] = m_trees_alloc;
      m_tree_sizes[m_n_trees] = tree_size;
      m_tree_weights[m_n_trees] = 1.0;
      m_trees_alloc += tree_size;
      m_n_trees++;
   }

   void AccumulateImportance(int node_idx)
   {
      if(m_nodes[node_idx].is_leaf) return;
      int f = m_nodes[node_idx].feature_idx;
      if(f >= 0 && f < m_n_features)
         m_feature_importance[f] += m_nodes[node_idx].best_gain;
      AccumulateImportance(m_nodes[node_idx].left_child);
      AccumulateImportance(m_nodes[node_idx].right_child);
   }

   // Find best exact split for a specific leaf node (uses pre-sorted indices)
   // Two-pass sparsity-aware: Pass 1 NaN defaults right, Pass 2 NaN defaults left
   void FindBestSplitExact(int leaf_node_idx, int n_buf,
                           double &best_gain, int &best_feature,
                           double &best_threshold, double &best_nan_dir,
                           int active_group = -1)
   {
      best_gain = -1e30;
      best_feature = -1;
      best_threshold = 0.0;
      best_nan_dir = ML_NAN_GOES_RIGHT;

      int leaf_count = m_nodes[leaf_node_idx].sample_count;
      if(leaf_count < 2 * ML_MinChildSamples) return;

      double total_g = m_nodes[leaf_node_idx].grad_sum;
      double total_h = m_nodes[leaf_node_idx].hess_sum;

      int n_sel = ArraySize(m_selected_features);
      int capacity = m_buffer.Capacity();

      for(int fl = 0; fl < n_sel; fl++)
      {
         int f = m_selected_features[fl];
         // Interaction constraint: skip features not in the active group
         if(m_has_interaction && active_group >= 0)
         {
            int feat_group = m_interaction_groups[f];
            if(feat_group >= 0 && feat_group != active_group) continue;
         }
         int off = f * capacity;

         // First, compute NaN stats for this feature in this leaf
         double nan_g = 0.0, nan_h = 0.0;
         int nan_c = 0;
         for(int k = 0; k < n_buf; k++)
         {
            int si = m_presorted[off + k];
            if(m_sample_leaf[si] != leaf_node_idx) continue;
            double val = m_buffer.GetFeature(si, f);
            if(val <= ML_NAN_SENTINEL + 1e300)
            {
               nan_g += m_grads[si];
               nan_h += m_hess[si];
               nan_c++;
            }
         }

         // Non-NaN total stats
         double valid_g = total_g - nan_g;
         double valid_h = total_h - nan_h;
         int valid_c = leaf_count - nan_c;

         // Two passes: pass=0 (NaN defaults right), pass=1 (NaN defaults left)
         for(int pass = 0; pass < 2; pass++)
         {
            // If no NaN samples, second pass is identical — skip
            if(pass == 1 && nan_c == 0) break;

            double cur_nan_dir = (pass == 0) ? ML_NAN_GOES_RIGHT : ML_NAN_GOES_LEFT;

            // Scan pre-sorted global indices, skip non-leaf and NaN samples
            double run_g = 0.0, run_h = 0.0;
            int    run_c = 0;
            double prev_val = -1e308;

            for(int k = 0; k < n_buf; k++)
            {
               int si = m_presorted[off + k];
               if(m_sample_leaf[si] != leaf_node_idx) continue;

               double val = m_buffer.GetFeature(si, f);
               // Skip NaN samples — they'll be added to the chosen side
               if(val <= ML_NAN_SENTINEL + 1e300) continue;

               run_g += m_grads[si];
               run_h += m_hess[si];
               run_c++;

               int right_valid = valid_c - run_c;
               if(right_valid == 0) break; // no more valid samples for right

               if(val == prev_val) { prev_val = val; continue; }

               // Compute effective left/right including NaN assignment
               double eff_left_g, eff_left_h, eff_right_g, eff_right_h;
               int eff_left_c, eff_right_c;
               if(pass == 1)  // NaN goes left
               {
                  eff_left_g = run_g + nan_g;
                  eff_left_h = run_h + nan_h;
                  eff_left_c = run_c + nan_c;
                  eff_right_g = valid_g - run_g;
                  eff_right_h = valid_h - run_h;
                  eff_right_c = right_valid;
               }
               else  // NaN goes right (pass == 0)
               {
                  eff_left_g = run_g;
                  eff_left_h = run_h;
                  eff_left_c = run_c;
                  eff_right_g = valid_g - run_g + nan_g;
                  eff_right_h = valid_h - run_h + nan_h;
                  eff_right_c = right_valid + nan_c;
               }

               if(eff_left_c < ML_MinChildSamples || eff_right_c < ML_MinChildSamples)
                  { prev_val = val; continue; }
               if(eff_left_h < ML_MinChildWeight || eff_right_h < ML_MinChildWeight)
                  { prev_val = val; continue; }

               double gain = ML_SplitGain(eff_left_g, eff_left_h,
                                           eff_right_g, eff_right_h,
                                           ML_L1Reg, ML_L2Reg, ML_Gamma);

               // Monotone constraint check
               if(m_has_monotone && gain > 0.0)
               {
                  int mc = m_monotone[f];
                  if(mc != 0)
                  {
                     double lv = ML_LeafValue(eff_left_g, eff_left_h, ML_L1Reg, ML_L2Reg);
                     double rv = ML_LeafValue(eff_right_g, eff_right_h, ML_L1Reg, ML_L2Reg);
                     if((mc == +1 && lv > rv + ML_EPS) || (mc == -1 && lv < rv - ML_EPS))
                        { prev_val = val; continue; }
                  }
               }

               if(gain > best_gain)
               {
                  best_gain = gain;
                  best_feature = f;
                  best_nan_dir = cur_nan_dir;
                  // Find next leaf sample's value for midpoint threshold
                  double next_val = val;
                  for(int kk = k + 1; kk < n_buf; kk++)
                  {
                     int si2 = m_presorted[off + kk];
                     if(m_sample_leaf[si2] != leaf_node_idx) continue;
                     double v2 = m_buffer.GetFeature(si2, f);
                     if(v2 <= ML_NAN_SENTINEL + 1e300) continue; // skip NaN
                     next_val = v2;
                     break;
                  }
                  best_threshold = 0.5 * (val + next_val);
               }
               prev_val = val;
            }
         }
      }
   }

public:
   CXGBoost_Model() : m_n_trees(0), m_trees_alloc(0), m_uses_v2_nodes(false) {}
   ~CXGBoost_Model() {}

   void Init(ENUM_ML_TASK task, int n_features)
   {
      InitBase(task, n_features);
      m_n_trees = 0;
      m_trees_alloc = 0;

      int max_samples = ML_BufferCapacity;
      ArrayResize(m_trees, 4096);
      ArrayResize(m_tree_offsets, ML_MaxTrees + 10);
      ArrayResize(m_tree_sizes, ML_MaxTrees + 10);
      ArrayResize(m_preds, max_samples);
      ArrayResize(m_grads, max_samples);
      ArrayResize(m_hess, max_samples);
      ArrayResize(m_selected_features, n_features);
      ArrayResize(m_sample_leaf, max_samples);
      ArrayResize(m_nodes, ML_MAX_TREE_NODES);
      ArrayResize(m_sort_vals, max_samples);
      ArrayResize(m_sort_idx, max_samples);
      ArrayResize(m_tree_weights, ML_MaxTrees + 10);
      ArrayInitialize(m_tree_weights, 1.0);
      m_preds_buf_count = 0;
      m_preds_buf_wpos = 0;

      ML_SeedRng((ulong)GetTickCount() + 1);
   }

   virtual double Predict(const double &features[], int count)
   {
      double pred = m_base_score;
      if(m_uses_v2_nodes)
      {
         for(int t = 0; t < m_n_trees; t++)
            pred += ML_LearningRate * m_tree_weights[t] * ML_PredictFlatTreeV2(m_trees, m_tree_offsets[t],
                                                          features, m_n_features);
      }
      else
      {
         for(int t = 0; t < m_n_trees; t++)
            pred += ML_LearningRate * m_tree_weights[t] * ML_PredictFlatTree(m_trees, m_tree_offsets[t],
                                                          features, m_n_features);
      }
      return pred;
   }

   virtual void Train(bool force = false)
   {
      int n = m_buffer.Count();
      if(!force && n < ML_ColdStartMin) return;
      if(m_early_stopped) return;  // Already stopped
      if(m_train_count == 0) ComputeBaseScore();

      // Recompute predictions (incremental where possible)
      double features[];
      ArrayResize(features, m_n_features);
      if(m_n_trees == 0 || m_preds_buf_count == 0)
      {
         // Full recompute — cold start or no trees yet
         for(int i = 0; i < n; i++)
         {
            m_preds[i] = m_base_score;
            m_buffer.GetFeatures(i, features);
            for(int t = 0; t < m_n_trees; t++)
               m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
         }
      }
      else
      {
         // Incremental — only recompute dirty (newly written) samples
         if(n > m_preds_buf_count)
         {
            // Buffer grew: dirty = [m_preds_buf_count, n)
            for(int i = m_preds_buf_count; i < n; i++)
            {
               m_preds[i] = m_base_score;
               m_buffer.GetFeatures(i, features);
               for(int t = 0; t < m_n_trees; t++)
                  m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
            }
         }
         else
         {
            // Buffer full, wrapping — dirty = overwritten region
            int new_wpos = m_buffer.WritePos();
            int old_wpos = m_preds_buf_wpos;
            if(new_wpos != old_wpos)
            {
               if(new_wpos >= old_wpos)
               {
                  // No wraparound: dirty = [old_wpos, new_wpos)
                  for(int i = old_wpos; i < new_wpos; i++)
                  {
                     m_preds[i] = m_base_score;
                     m_buffer.GetFeatures(i, features);
                     for(int t = 0; t < m_n_trees; t++)
                        m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
                  }
               }
               else
               {
                  // Wraparound: dirty = [old_wpos, n) + [0, new_wpos)
                  for(int i = old_wpos; i < n; i++)
                  {
                     m_preds[i] = m_base_score;
                     m_buffer.GetFeatures(i, features);
                     for(int t = 0; t < m_n_trees; t++)
                        m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
                  }
                  for(int i = 0; i < new_wpos; i++)
                  {
                     m_preds[i] = m_base_score;
                     m_buffer.GetFeatures(i, features);
                     for(int t = 0; t < m_n_trees; t++)
                        m_preds[i] += ML_LearningRate * m_tree_weights[t] * PredictTree(t, features);
                  }
               }
            }
         }
      }
      m_preds_buf_count = n;
      m_preds_buf_wpos = m_buffer.WritePos();

      ComputeGradHess(m_preds, m_grads, m_hess, n);

      // Pre-sort all features once (avoids re-sorting per leaf per feature)
      int capacity = m_buffer.Capacity();
      if(ArraySize(m_presorted) < m_n_features * capacity)
         ArrayResize(m_presorted, m_n_features * capacity);
      for(int f = 0; f < m_n_features; f++)
      {
         for(int i = 0; i < n; i++)
         {
            m_sort_vals[i] = m_buffer.GetFeature(i, f);
            m_sort_idx[i] = i;
         }
         ML_ArgSort(m_sort_vals, m_sort_idx, n);
         ArrayCopy(m_presorted, m_sort_idx, f * capacity, 0, n);
      }

      // Column subsampling
      int n_col = (int)(m_n_features * ML_ColsampleTree);
      if(n_col < 1) n_col = 1;
      int all_feats[];
      ArrayResize(all_feats, m_n_features);

      for(int round = 0; round < ML_TreesPerRound; round++)
      {
         for(int i = 0; i < m_n_features; i++) all_feats[i] = i;
         ML_Shuffle(all_feats, m_n_features);
         ArrayResize(m_selected_features, n_col);
         ArrayCopy(m_selected_features, all_feats, 0, 0, n_col);

         // Row subsampling
         int n_sub = (int)(n * ML_Subsample);
         if(n_sub < ML_MinChildSamples) n_sub = n;
         int sub_indices[];
         ArrayResize(sub_indices, n);
         for(int i = 0; i < n; i++) sub_indices[i] = i;
         if(n_sub < n) ML_Shuffle(sub_indices, n);

         // DART: drop random trees before building
         int dart_dropped[];
         int n_dart_dropped = 0;
         if(ML_DARTEnabled && m_n_trees > 0)
         {
            int n_drop = MathMax(1, (int)(m_n_trees * ML_DARTDropRate));
            if(n_drop > m_n_trees) n_drop = m_n_trees;

            int all_indices[];
            ArrayResize(all_indices, m_n_trees);
            for(int di = 0; di < m_n_trees; di++) all_indices[di] = di;
            ML_Shuffle(all_indices, m_n_trees);

            ArrayResize(dart_dropped, n_drop);
            n_dart_dropped = n_drop;
            for(int di = 0; di < n_drop; di++) dart_dropped[di] = all_indices[di];

            // Subtract dropped trees from predictions
            double feats_dart[];
            ArrayResize(feats_dart, m_n_features);
            for(int di = 0; di < n_drop; di++)
            {
               int dt = dart_dropped[di];
               double w = m_tree_weights[dt];
               for(int si = 0; si < n; si++)
               {
                  m_buffer.GetFeatures(si, feats_dart);
                  m_preds[si] -= ML_LearningRate * w * PredictTree(dt, feats_dart);
               }
            }

            // Recompute gradients on modified predictions
            ComputeGradHess(m_preds, m_grads, m_hess, n);
         }

         // Mark V2 nodes active (new trees always use V2 format)
         m_uses_v2_nodes = true;

         BuildTreeLevelWise(sub_indices, n_sub, n);

         // Update predictions
         int last_tree = m_n_trees - 1;
         for(int i = 0; i < n; i++)
         {
            m_buffer.GetFeatures(i, features);
            m_preds[i] += ML_LearningRate * PredictTree(last_tree, features);
         }

         // DART: restore dropped trees with rescaled weights
         if(n_dart_dropped > 0)
         {
            int new_tree = m_n_trees - 1;
            double scale_existing = (double)n_dart_dropped / (double)(n_dart_dropped + 1);
            double scale_new = 1.0 / (double)(n_dart_dropped + 1);

            double feats_dart[];
            ArrayResize(feats_dart, m_n_features);
            for(int di = 0; di < n_dart_dropped; di++)
            {
               int dt = dart_dropped[di];
               m_tree_weights[dt] *= scale_existing;
               for(int si = 0; si < n; si++)
               {
                  m_buffer.GetFeatures(si, feats_dart);
                  m_preds[si] += ML_LearningRate * m_tree_weights[dt] * PredictTree(dt, feats_dart);
               }
            }

            m_tree_weights[new_tree] = scale_new;
            // Adjust new tree prediction: was added with weight 1.0, need scale_new
            for(int si = 0; si < n; si++)
            {
               m_buffer.GetFeatures(si, feats_dart);
               double tp = PredictTree(new_tree, feats_dart);
               m_preds[si] -= ML_LearningRate * (1.0 - scale_new) * tp;
            }
         }

         if(round < ML_TreesPerRound - 1)
            ComputeGradHess(m_preds, m_grads, m_hess, n);
      }

      // Check early stopping
      CheckEarlyStop(m_preds, n);

      m_train_count++;
   }

   void BuildTreeLevelWise(int &sub_indices[], int n_sub, int n_buf)
   {
      // Init root with subsampled data
      ArrayInitialize(m_sample_leaf, -1);
      double root_g = 0.0, root_h = 0.0;
      int root_count = 0;
      for(int i = 0; i < n_sub; i++)
      {
         int si = sub_indices[i];
         m_sample_leaf[si] = 0;
         root_g += m_grads[si];
         root_h += m_hess[si];
         root_count++;
      }

      int n_nodes = 0;
      ZeroMemory(m_nodes[0]);
      m_nodes[0].is_leaf = true;
      m_nodes[0].active_group = -1;
      m_nodes[0].grad_sum = root_g;
      m_nodes[0].hess_sum = root_h;
      m_nodes[0].sample_count = root_count;
      n_nodes = 1;

      int max_depth = ML_MaxDepth;

      // Level-wise: process all leaves at each depth
      for(int depth = 0; depth < max_depth; depth++)
      {
         // Per-level column subsampling (ML_ColsampleLevel)
         if(ML_ColsampleLevel < 1.0)
         {
            int n_col_lvl = (int)(m_n_features * ML_ColsampleLevel);
            if(n_col_lvl < 1) n_col_lvl = 1;
            int lvl_feats[];
            ArrayResize(lvl_feats, m_n_features);
            for(int ii = 0; ii < m_n_features; ii++) lvl_feats[ii] = ii;
            ML_Shuffle(lvl_feats, m_n_features);
            ArrayResize(m_selected_features, n_col_lvl);
            ArrayCopy(m_selected_features, lvl_feats, 0, 0, n_col_lvl);
         }

         // Find all leaves at current depth (simple: scan nodes)
         int leaves_at_depth[];
         int n_leaves_d = 0;
         ArrayResize(leaves_at_depth, n_nodes);
         for(int i = 0; i < n_nodes; i++)
            if(m_nodes[i].is_leaf && m_nodes[i].sample_count >= 2 * ML_MinChildSamples)
               leaves_at_depth[n_leaves_d++] = i;

         if(n_leaves_d == 0) break;

         for(int li = 0; li < n_leaves_d; li++)
         {
            int L = leaves_at_depth[li];
            if(n_nodes >= ML_MAX_TREE_NODES - 2) break;

            // Per-node column subsampling (ML_ColsampleNode)
            if(ML_ColsampleNode < 1.0)
            {
               int n_col_node = (int)(m_n_features * ML_ColsampleNode);
               if(n_col_node < 1) n_col_node = 1;
               int node_feats[];
               ArrayResize(node_feats, m_n_features);
               for(int ii = 0; ii < m_n_features; ii++) node_feats[ii] = ii;
               ML_Shuffle(node_feats, m_n_features);
               ArrayResize(m_selected_features, n_col_node);
               ArrayCopy(m_selected_features, node_feats, 0, 0, n_col_node);
            }

            double bg; int bf; double bt; double nan_dir;
            FindBestSplitExact(L, n_buf, bg, bf, bt, nan_dir, m_nodes[L].active_group);

            if(bg <= 0 || bf < 0) continue; // no good split

            // Compute child active_group for interaction constraints
            int child_group = m_nodes[L].active_group;
            if(m_has_interaction && child_group < 0 && bf >= 0)
            {
               int fg = m_interaction_groups[bf];
               if(fg >= 0) child_group = fg;
            }

            // Split
            int left_idx = n_nodes++;
            int right_idx = n_nodes++;
            m_nodes[L].is_leaf = false;
            m_nodes[L].feature_idx = bf;
            m_nodes[L].threshold = bt;
            m_nodes[L].best_gain = bg;
            m_nodes[L].nan_direction = nan_dir;
            m_nodes[L].left_child = left_idx;
            m_nodes[L].right_child = right_idx;

            // Reassign samples (NaN-aware routing)
            double left_g = 0, left_h = 0, right_g = 0, right_h = 0;
            int left_c = 0, right_c = 0;

            for(int i = 0; i < n_buf; i++)
            {
               if(m_sample_leaf[i] != L) continue;
               double fval = m_buffer.GetFeature(i, bf);
               bool is_nan = (fval <= ML_NAN_SENTINEL + 1e300);
               bool go_left = is_nan ? (nan_dir > 0.5) : (fval < bt);
               if(go_left)
               {
                  m_sample_leaf[i] = left_idx;
                  left_g += m_grads[i];
                  left_h += m_hess[i];
                  left_c++;
               }
               else
               {
                  m_sample_leaf[i] = right_idx;
                  right_g += m_grads[i];
                  right_h += m_hess[i];
                  right_c++;
               }
            }

            ZeroMemory(m_nodes[left_idx]);
            m_nodes[left_idx].is_leaf = true;
            m_nodes[left_idx].grad_sum = left_g;
            m_nodes[left_idx].hess_sum = left_h;
            m_nodes[left_idx].sample_count = left_c;
            m_nodes[left_idx].active_group = child_group;

            ZeroMemory(m_nodes[right_idx]);
            m_nodes[right_idx].is_leaf = true;
            m_nodes[right_idx].grad_sum = right_g;
            m_nodes[right_idx].hess_sum = right_h;
            m_nodes[right_idx].sample_count = right_c;
            m_nodes[right_idx].active_group = child_group;
         }
      }

      // Compute leaf values
      for(int i = 0; i < n_nodes; i++)
         if(m_nodes[i].is_leaf)
            m_nodes[i].leaf_value = ML_LeafValue(m_nodes[i].grad_sum, m_nodes[i].hess_sum,
                                                  ML_L1Reg, ML_L2Reg);

      AccumulateImportance(0);
      AddTreeToEnsemble(0);
   }

   virtual bool SaveToFile(string filename)
   {
      int handle = FileOpen(filename, FILE_WRITE | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE) return false;

      FileWriteInteger(handle, ML_MAGIC_SAVE);
      FileWriteInteger(handle, (int)ML_XGBOOST);
      FileWriteInteger(handle, (int)m_task);
      FileWriteInteger(handle, m_n_features);
      FileWriteInteger(handle, m_n_trees);
      FileWriteDouble(handle, m_base_score);
      FileWriteInteger(handle, m_train_count);
      FileWriteInteger(handle, m_trees_alloc);

      if(m_trees_alloc > 0)
         FileWriteArray(handle, m_trees, 0, m_trees_alloc);
      FileWriteArray(handle, m_tree_offsets, 0, m_n_trees);
      FileWriteArray(handle, m_tree_sizes, 0, m_n_trees);
      FileWriteArray(handle, m_feature_importance, 0, m_n_features);

      // GBT V2 extension block — early stopping state
      FileWriteInteger(handle, ML_MAGIC_GBT_V2);
      FileWriteDouble(handle, m_best_val_loss);
      FileWriteInteger(handle, m_no_improve_rounds);
      FileWriteInteger(handle, m_early_stopped ? 1 : 0);

      // DART weights
      FileWriteInteger(handle, ML_MAGIC_DART);
      FileWriteArray(handle, m_tree_weights, 0, m_n_trees);

      // GBT V3: NaN direction per node (V2 flat tree format)
      FileWriteInteger(handle, ML_MAGIC_GBT_V3);
      FileWriteInteger(handle, m_uses_v2_nodes ? 1 : 0);

      // GBT V4: reserved for future V4 extensions
      FileWriteInteger(handle, ML_MAGIC_V4);
      FileWriteInteger(handle, 0);  // reserved

      FileWriteInteger(handle, ML_MAGIC_END);
      FileClose(handle);
      return true;
   }

   virtual bool LoadFromFile(string filename)
   {
      int handle = FileOpen(filename, FILE_READ | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE) return false;

      int magic = FileReadInteger(handle);
      if(magic != ML_MAGIC_SAVE) { FileClose(handle); return false; }
      int model_type = FileReadInteger(handle);
      if(model_type != (int)ML_XGBOOST) { FileClose(handle); return false; }

      m_task = (ENUM_ML_TASK)FileReadInteger(handle);
      m_n_features = FileReadInteger(handle);
      m_n_trees = FileReadInteger(handle);
      m_base_score = FileReadDouble(handle);
      m_train_count = FileReadInteger(handle);
      m_trees_alloc = FileReadInteger(handle);

      ArrayResize(m_trees, MathMax(m_trees_alloc, 1));
      if(m_trees_alloc > 0)
         FileReadArray(handle, m_trees, 0, m_trees_alloc);
      ArrayResize(m_tree_offsets, ML_MaxTrees + 10);
      ArrayResize(m_tree_sizes, ML_MaxTrees + 10);
      FileReadArray(handle, m_tree_offsets, 0, m_n_trees);
      FileReadArray(handle, m_tree_sizes, 0, m_n_trees);

      ArrayResize(m_feature_importance, m_n_features);
      FileReadArray(handle, m_feature_importance, 0, m_n_features);

      // Try reading GBT V2 extension block
      int next_magic = FileReadInteger(handle);
      if(next_magic == ML_MAGIC_GBT_V2)
      {
         m_best_val_loss = FileReadDouble(handle);
         m_no_improve_rounds = FileReadInteger(handle);
         m_early_stopped = (FileReadInteger(handle) != 0);
         next_magic = FileReadInteger(handle);
      }
      else
      {
         // Old format — use defaults
         m_best_val_loss = 1e30;
         m_no_improve_rounds = 0;
         m_early_stopped = false;
      }
      // Try reading DART weights
      if(next_magic == ML_MAGIC_DART)
      {
         ArrayResize(m_tree_weights, ML_MaxTrees + 10);
         FileReadArray(handle, m_tree_weights, 0, m_n_trees);
         next_magic = FileReadInteger(handle);
      }
      else
      {
         ArrayResize(m_tree_weights, ML_MaxTrees + 10);
         ArrayInitialize(m_tree_weights, 1.0);
      }
      // Try reading GBT V3 (NaN direction)
      if(next_magic == ML_MAGIC_GBT_V3)
      {
         m_uses_v2_nodes = (FileReadInteger(handle) != 0);
         next_magic = FileReadInteger(handle);
      }
      else
      {
         m_uses_v2_nodes = false;  // old file, trees use V1 format
      }
      // Try reading GBT V4 (reserved — XGBoost)
      if(next_magic == ML_MAGIC_V4)
      {
         int v4_reserved = FileReadInteger(handle);
         next_magic = FileReadInteger(handle);
      }

      // Always reset incremental preds (buffer is not saved)
      m_preds_buf_count = 0;
      m_preds_buf_wpos = 0;

      int end_magic = next_magic;
      FileClose(handle);
      return (end_magic == ML_MAGIC_END);
   }

   virtual void Reset()
   {
      CMLModelBase::Reset();
      m_n_trees = 0;
      m_trees_alloc = 0;
      m_uses_v2_nodes = false;
      ArrayInitialize(m_tree_weights, 1.0);
   }

   // Compute per-feature SHAP values for a single sample.
   // Identical to CLightGBM_Model::ComputeSHAP — both use flat V2 tree format.
   // phi[] sized (n_features + 1): phi[n_features] = base score intercept.
   virtual void ComputeSHAP(const double &features[], int count, double &phi[])
   {
      ArrayResize(phi, m_n_features + 1);
      ArrayInitialize(phi, 0.0);
      phi[m_n_features] = m_base_score;

      if(m_n_trees == 0 || !m_uses_v2_nodes) return;  // V1 trees: skip (no SHAP support)

      SHAPPath path;
      double tree_phi[];
      ArrayResize(tree_phi, m_n_features);

      for(int t = 0; t < m_n_trees; t++)
      {
         ArrayInitialize(tree_phi, 0.0);
         ZeroMemory(path);
         path.len = 0;

         int root = m_tree_offsets[t];
         SHAP_Recurse(m_trees, root, root, features, m_n_features,
                      tree_phi, path, 1.0, 1.0, -1);

         double scale = ML_LearningRate * m_tree_weights[t];
         for(int f = 0; f < m_n_features; f++)
            phi[f] += scale * tree_phi[f];
      }
   }
};

//===================================================================
//===================================================================
// CCatBoost_Model — Oblivious decision trees
//===================================================================
//===================================================================
class CCatBoost_Model : public CMLModelBase
{
private:
   // Per-tree: fixed depth, same split at each level for all nodes
   int    m_split_features[];     // [n_trees * depth]
   double m_split_thresholds[];   // [n_trees * depth]
   double m_leaf_values[];        // [n_trees * (1 << depth)]
   int    m_n_trees;
   int    m_depth;                // fixed depth for all trees
   int    m_leaves_per_tree;      // 1 << depth
   double m_tree_weights[];       // per-tree weight (normally 1.0, DART rescales)
   double m_nan_directions[];     // [n_trees * depth] NaN direction per split level
   bool   m_uses_nan_dirs;        // true once trained with NaN direction learning

   // Lossguide tree storage (flat V2 format, same as LightGBM/XGBoost)
   // Only used when ML_CB_GrowPolicy == 1
   double m_lg_trees[];           // flat tree data (all lossguide trees concatenated)
   int    m_lg_offsets[];         // start offset per tree in m_lg_trees
   int    m_lg_sizes[];           // node count per tree (in doubles)
   int    m_n_lg_trees;           // number of lossguide trees stored
   int    m_lg_alloc;             // allocated size of m_lg_trees used so far
   MLTreeNode m_lg_nodes[];       // building buffer (ML_MAX_TREE_NODES size)

   // Histogram working buffers for BuildLossguideTree
   double m_lg_hist_grad[];       // [max_leaves * n_sel_features * stride_feat]
   double m_lg_hist_hess[];
   int    m_lg_hist_count[];
   int    m_lg_sel_features[];    // column-sampled feature indices for lossguide
   int    m_lg_sample_leaf[];     // per-sample node index during tree building

   CMLBinMapper m_bins;

   // Working buffers
   double m_preds[];
   double m_grads[];
   double m_hess[];
   int    m_sample_leaf[];        // leaf assignment per sample (0..2^depth-1)
   double m_leaf_grad[];          // [2^depth] sum of gradients per leaf
   double m_leaf_hess[];          // [2^depth] sum of hessians per leaf
   int    m_leaf_count[];         // [2^depth] sample count per leaf

   // Histogram buffers for oblivious split finding
   double m_ob_hist_grad[];    // [max_leaves * n_features * n_bins]
   double m_ob_hist_hess[];    // same
   int    m_ob_hist_count[];   // same

   // Incremental prediction tracking
   int  m_preds_buf_count;
   int  m_preds_buf_wpos;
   double m_recompute_feats[];  // pre-allocated buffer for RecomputePredForSample

   // Find the single best (feature, threshold, nan_direction) for a level of the oblivious tree.
   // Histogram-based: O(n*features*leaves + features*bins*leaves) instead of
   // brute-force O(features*bins*leaves*n). ~200x faster at typical dimensions.
   // Requires m_leaf_count[], m_leaf_grad[], m_leaf_hess[] pre-populated by caller.
   // Two-pass NaN direction: evaluates gain with NaN going right (pass 0) and left (pass 1).
   void FindBestObliviousSplit(int n_samples, int current_depth, int n_current_leaves,
                                int &best_feature, double &best_threshold,
                                double &best_nan_dir, int active_group = -1)
   {
      double best_total_gain = -1e30;
      best_feature = -1;
      best_threshold = 0.0;
      best_nan_dir = ML_NAN_GOES_RIGHT;

      int n_sel = (int)(m_n_features * ML_ColsampleTree);
      if(n_sel < 1) n_sel = 1;
      int sel_feats[];
      ArrayResize(sel_feats, m_n_features);
      for(int i = 0; i < m_n_features; i++) sel_feats[i] = i;
      ML_Shuffle(sel_feats, m_n_features);

      int nbins = m_bins.NBins();
      int nan_bin = m_bins.GetNaNBin();  // index = nbins (one past valid bins)

      // Step 1: Build per-leaf histograms in a single pass over samples
      // Layout: hist[leaf * stride_leaf + feat_local * stride_feat + bin]
      // stride_feat = nbins+1 to include NaN bin at index nbins
      int stride_feat = nbins + 1;
      int stride_leaf = n_sel * stride_feat;
      int hist_total = n_current_leaves * stride_leaf;

      // Ensure histogram arrays are large enough
      if(ArraySize(m_ob_hist_grad) < hist_total)
      {
         ArrayResize(m_ob_hist_grad, hist_total);
         ArrayResize(m_ob_hist_hess, hist_total);
         ArrayResize(m_ob_hist_count, hist_total);
      }

      // Clear histograms
      ArrayInitialize(m_ob_hist_grad, 0.0);
      ArrayInitialize(m_ob_hist_hess, 0.0);
      ArrayInitialize(m_ob_hist_count, 0);

      // Single pass: accumulate gradients into histograms
      for(int i = 0; i < n_samples; i++)
      {
         int leaf = m_sample_leaf[i];
         if(leaf < 0 || leaf >= n_current_leaves) continue;
         for(int fl = 0; fl < n_sel; fl++)
         {
            int f = sel_feats[fl];
            int bin = m_bins.GetBin(f, m_buffer.GetFeature(i, f));
            int idx = leaf * stride_leaf + fl * stride_feat + bin;
            m_ob_hist_grad[idx] += m_grads[i];
            m_ob_hist_hess[idx] += m_hess[i];
            m_ob_hist_count[idx]++;
         }
      }

      // Step 2: Scan histograms with prefix sums to find best (feature, threshold, nan_dir)
      for(int fl = 0; fl < n_sel; fl++)
      {
         int f = sel_feats[fl];
         // Interaction constraint: skip features not in the active group
         if(m_has_interaction && active_group >= 0)
         {
            int feat_group = m_interaction_groups[f];
            if(feat_group >= 0 && feat_group != active_group) continue;
         }

         // Pre-compute per-leaf NaN stats for this feature
         double nan_g[];
         double nan_h[];
         int    nan_c[];
         ArrayResize(nan_g, n_current_leaves);
         ArrayResize(nan_h, n_current_leaves);
         ArrayResize(nan_c, n_current_leaves);
         bool has_any_nan = false;
         for(int leaf = 0; leaf < n_current_leaves; leaf++)
         {
            int nidx = leaf * stride_leaf + fl * stride_feat + nan_bin;
            nan_g[leaf] = m_ob_hist_grad[nidx];
            nan_h[leaf] = m_ob_hist_hess[nidx];
            nan_c[leaf] = m_ob_hist_count[nidx];
            if(nan_c[leaf] > 0) has_any_nan = true;
         }

         // Prefix-sum accumulators per leaf
         double psum_g[];
         double psum_h[];
         int    psum_c[];
         ArrayResize(psum_g, n_current_leaves);
         ArrayResize(psum_h, n_current_leaves);
         ArrayResize(psum_c, n_current_leaves);
         ArrayInitialize(psum_g, 0.0);
         ArrayInitialize(psum_h, 0.0);
         ArrayInitialize(psum_c, 0);

         for(int b = 0; b < nbins - 1; b++)
         {
            double thresh = m_bins.GetBinThreshold(f, b);

            // Accumulate prefix sums (non-NaN bins only)
            for(int leaf = 0; leaf < n_current_leaves; leaf++)
            {
               int idx = leaf * stride_leaf + fl * stride_feat + b;
               psum_g[leaf] += m_ob_hist_grad[idx];
               psum_h[leaf] += m_ob_hist_hess[idx];
               psum_c[leaf] += m_ob_hist_count[idx];
            }

            // Two-pass NaN direction evaluation
            int n_passes = has_any_nan ? 2 : 1;
            for(int pass = 0; pass < n_passes; pass++)
            {
               // pass 0: NaN goes right (default), pass 1: NaN goes left
               double candidate_nan_dir = (pass == 0) ? ML_NAN_GOES_RIGHT : ML_NAN_GOES_LEFT;
               double total_gain = 0.0;

               for(int leaf = 0; leaf < n_current_leaves; leaf++)
               {
                  if(m_leaf_count[leaf] < 2 * ML_MinChildSamples) continue;

                  // Base split: prefix sum = left, remainder = right
                  double lg = psum_g[leaf];
                  double lh = psum_h[leaf];
                  int    lc = psum_c[leaf];

                  // Add NaN stats to the chosen side
                  if(pass == 1)  // NaN goes left
                  {
                     lg += nan_g[leaf];
                     lh += nan_h[leaf];
                     lc += nan_c[leaf];
                  }

                  double rg = m_leaf_grad[leaf] - lg;
                  double rh = m_leaf_hess[leaf] - lh;
                  int    rc = m_leaf_count[leaf] - lc;

                  if(lc < ML_MinChildSamples || rc < ML_MinChildSamples) continue;
                  if(lh < ML_MinChildWeight || rh < ML_MinChildWeight) continue;

                  double gain = ML_SplitGain(lg, lh, rg, rh,
                                              ML_L1Reg, ML_L2Reg, ML_Gamma);
                  total_gain += gain;
               }

               // CatBoost random strength: annealed noise on split scores
               if(ML_CB_RandomStrength > 0.0 && m_train_count > 0)
                  total_gain += ML_RandNormal(0.0, ML_CB_RandomStrength / MathSqrt((double)(m_train_count + 1)));

               // Monotone constraint check — reject if ANY leaf violates
               if(m_has_monotone)
               {
                  int mc = m_monotone[f];
                  if(mc != 0)
                  {
                     bool violated = false;
                     for(int leaf = 0; leaf < n_current_leaves; leaf++)
                     {
                        double lg = psum_g[leaf];
                        double lh = psum_h[leaf];
                        if(pass == 1) { lg += nan_g[leaf]; lh += nan_h[leaf]; }
                        double rg = m_leaf_grad[leaf] - lg;
                        double rh = m_leaf_hess[leaf] - lh;
                        double lv = ML_LeafValue(lg, lh, ML_L1Reg, ML_L2Reg);
                        double rv = ML_LeafValue(rg, rh, ML_L1Reg, ML_L2Reg);
                        if((mc == +1 && lv > rv + ML_EPS) || (mc == -1 && lv < rv - ML_EPS))
                        { violated = true; break; }
                     }
                     if(violated) total_gain = -1e30;
                  }
               }

               if(total_gain > best_total_gain)
               {
                  best_total_gain = total_gain;
                  best_feature = f;
                  best_threshold = thresh;
                  best_nan_dir = candidate_nan_dir;
               }
            }
         }
      }
   }

   // NaN-aware leaf index for a sample in a given tree.
   // If m_uses_nan_dirs, NaN sentinel values use learned direction; otherwise features[feat] >= thresh.
   int ComputeLeafIndex(int sample_idx, int tree_idx)
   {
      int leaf_idx = 0;
      int split_off = tree_idx * m_depth;
      for(int d = 0; d < m_depth; d++)
      {
         int feat = m_split_features[split_off + d];
         if(feat < 0 || feat >= m_n_features) continue;
         double fval = m_buffer.GetFeature(sample_idx, feat);
         bool go_right;
         if(m_uses_nan_dirs && fval <= ML_NAN_SENTINEL + 1e300)
            go_right = (m_nan_directions[split_off + d] < 0.5);  // ML_NAN_GOES_RIGHT = 0
         else
            go_right = (fval >= m_split_thresholds[split_off + d]);
         if(go_right)
            leaf_idx |= (1 << d);
      }
      return leaf_idx;
   }

   // NaN-aware leaf index from feature array (for Predict, no buffer access).
   int ComputeLeafIndexFromFeatures(int tree_idx, const double &features[])
   {
      int leaf_idx = 0;
      int split_off = tree_idx * m_depth;
      for(int d = 0; d < m_depth; d++)
      {
         int feat = m_split_features[split_off + d];
         if(feat < 0 || feat >= m_n_features) continue;
         double fval = features[feat];
         bool go_right;
         if(m_uses_nan_dirs && fval <= ML_NAN_SENTINEL + 1e300)
            go_right = (m_nan_directions[split_off + d] < 0.5);
         else
            go_right = (fval >= m_split_thresholds[split_off + d]);
         if(go_right)
            leaf_idx |= (1 << d);
      }
      return leaf_idx;
   }

   // Ordered boosting: recompute gradients using leave-prefix-out predictions.
   // CatBoost's core innovation — prevents target leakage by ensuring each
   // sample's gradient is computed using a model that hasn't fully "seen" it.
   //
   // Simplified single-permutation approach:
   //   1. Generate random permutation of buffer indices
   //   2. For sample at permutation position k, blend its current model
   //      prediction toward m_base_score proportionally to 1/k
   //   3. Recompute gradient from the blended prediction
   //
   // The blending alpha = (k-1)/k means:
   //   - k=1 (first in perm): uses m_base_score only (no self-influence)
   //   - k=n (last in perm):  uses ~full model prediction (minimal correction)
   // This approximates the effect of prefix models without maintaining them.
   void ComputeOrderedGradients(int n)
   {
      // Full ordered boosting stub: multi-permutation support
      if(ML_CB_OrderedPerms > 1)
      {
         int s = MathMin(ML_CB_OrderedPerms, 4);
         static bool warned = false;
         if(!warned)
         {
            Print("ML CB Ordered: ", s, " permutations requested (stub: using simplified single-perm)");
            warned = true;
         }
         // TODO: maintain s separate prediction arrays, average gradient estimates
         // Fall through to simplified single-permutation below
      }
      // Generate random permutation of sample indices
      int perm[];
      ArrayResize(perm, n);
      for(int i = 0; i < n; i++) perm[i] = i;
      ML_Shuffle(perm, n);

      for(int k = 0; k < n; k++)
      {
         int i = perm[k];

         // Blend current prediction toward base_score based on position
         // alpha = (k-1)/k: early samples get more correction (less self-influence)
         double alpha = (k > 0) ? (double)(k - 1) / (double)k : 0.0;
         double blended_pred = alpha * m_preds[i] + (1.0 - alpha) * m_base_score;

         // Recompute gradient and hessian from blended prediction
         double label = m_buffer.GetLabel(i);
         if(m_task == ML_TASK_REGRESSION)
         {
            m_grads[i] = blended_pred - label;
            m_hess[i]  = 1.0;
         }
         else
         {
            double p = 1.0 / (1.0 + MathExp(-blended_pred));
            m_grads[i] = p - label;
            m_hess[i]  = p * (1.0 - p);
         }
      }
   }


   // ---------------------------------------------------------------
   // Lossguide tree helpers (ML_CB_GrowPolicy == 1)
   // Leaf-wise algorithm using CatBoost's m_bins / m_grads / m_hess.
   // ---------------------------------------------------------------

   // Histogram index: [leaf * stride_leaf + feat_local * stride_feat + bin]
   int LG_HistIdx(int leaf, int feat_local, int bin, int stride_leaf, int stride_feat)
   {
      return leaf * stride_leaf + feat_local * stride_feat + bin;
   }

   // Find best split for a lossguide leaf using m_lg_hist_* histograms.
   void LG_FindBestSplit(int hist_leaf_idx, double total_grad, double total_hess,
                         int total_count, int n_sel, int stride_leaf, int stride_feat,
                         double &best_gain, int &best_feat_local, int &best_bin,
                         double &best_nan_dir, int active_group = -1)
   {
      best_gain       = -1e30;
      best_feat_local = -1;
      best_bin        = -1;
      best_nan_dir    = ML_NAN_GOES_RIGHT;
      int nbins = m_bins.NBins();

      for(int fl = 0; fl < n_sel; fl++)
      {
         if(m_has_interaction && active_group >= 0)
         {
            int feat_group = m_interaction_groups[m_lg_sel_features[fl]];
            if(feat_group >= 0 && feat_group != active_group) continue;
         }
         double run_g = 0.0, run_h = 0.0;
         int    run_c = 0;

         for(int b = 0; b < nbins - 1; b++)
         {
            int idx = LG_HistIdx(hist_leaf_idx, fl, b, stride_leaf, stride_feat);
            run_g += m_lg_hist_grad[idx];
            run_h += m_lg_hist_hess[idx];
            run_c += m_lg_hist_count[idx];

            int right_c = total_count - run_c;
            if(run_c < ML_MinChildSamples || right_c < ML_MinChildSamples) continue;

            double right_h = total_hess - run_h;
            if(run_h < ML_MinChildWeight || right_h < ML_MinChildWeight) continue;

            double right_g = total_grad - run_g;
            double gain = ML_SplitGain(run_g, run_h, right_g, right_h,
                                        ML_L1Reg, ML_L2Reg, ML_Gamma);

            if(m_has_monotone && gain > 0.0)
            {
               int mc = m_monotone[m_lg_sel_features[fl]];
               if(mc != 0)
               {
                  double lv = ML_LeafValue(run_g, run_h, ML_L1Reg, ML_L2Reg);
                  double rv = ML_LeafValue(right_g, right_h, ML_L1Reg, ML_L2Reg);
                  if((mc == +1 && lv > rv + ML_EPS) || (mc == -1 && lv < rv - ML_EPS))
                     continue;
               }
            }

            if(gain > best_gain)
            {
               best_gain       = gain;
               best_feat_local = fl;
               best_bin        = b;
            }
         }
      }

      // Evaluate NaN direction for the winning split
      if(best_feat_local >= 0)
      {
         int nan_bin = m_bins.GetNaNBin();
         int nan_idx = LG_HistIdx(hist_leaf_idx, best_feat_local, nan_bin, stride_leaf, stride_feat);
         double nan_g = m_lg_hist_grad[nan_idx];
         double nan_h = m_lg_hist_hess[nan_idx];
         int    nan_c = m_lg_hist_count[nan_idx];
         if(nan_c > 0)
         {
            double left_g = 0.0, left_h = 0.0;
            for(int b = 0; b <= best_bin; b++)
            {
               int idx = LG_HistIdx(hist_leaf_idx, best_feat_local, b, stride_leaf, stride_feat);
               left_g += m_lg_hist_grad[idx];
               left_h += m_lg_hist_hess[idx];
            }
            double right_g = total_grad - left_g - nan_g;
            double right_h = total_hess - left_h - nan_h;
            double gain_left  = ML_SplitGain(left_g + nan_g, left_h + nan_h,
                                              right_g, right_h, ML_L1Reg, ML_L2Reg, ML_Gamma);
            double gain_right = ML_SplitGain(left_g, left_h,
                                              right_g + nan_g, right_h + nan_h,
                                              ML_L1Reg, ML_L2Reg, ML_Gamma);
            best_nan_dir = (gain_left > gain_right) ? ML_NAN_GOES_LEFT : ML_NAN_GOES_RIGHT;
         }
      }
   }

   // Recursive DFS serialization of m_lg_nodes[] subtree into V2 flat format.
   int LG_SerializeTree(int node_idx, double &flat[], int &pos, int tree_start)
   {
      if(m_lg_nodes[node_idx].is_leaf)
      {
         flat[pos++] = ML_LEAF_MARKER;
         flat[pos++] = m_lg_nodes[node_idx].leaf_value;
         flat[pos++] = 0.0;  // V2 leaf padding
         flat[pos++] = 0.0;  // V2 leaf padding
         return pos;
      }
      flat[pos++] = (double)m_lg_nodes[node_idx].feature_idx;
      flat[pos++] = m_lg_nodes[node_idx].threshold;
      int right_offset_pos = pos;
      flat[pos++] = 0.0;  // right child offset placeholder
      flat[pos++] = m_lg_nodes[node_idx].nan_direction;

      pos = LG_SerializeTree(m_lg_nodes[node_idx].left_child, flat, pos, tree_start);
      flat[right_offset_pos] = (double)(pos - tree_start);
      pos = LG_SerializeTree(m_lg_nodes[node_idx].right_child, flat, pos, tree_start);
      return pos;
   }

   // Serialize lossguide tree and append to m_lg_trees[].  Evicts oldest when at capacity.
   void LG_AddTreeToEnsemble(int root_idx)
   {
      double flat[];
      ArrayResize(flat, ML_MAX_TREE_NODES * ML_NODE_WIDTH_V2);
      int pos = 0;
      LG_SerializeTree(root_idx, flat, pos, 0);
      int tree_size = pos;

      // Evict oldest if at capacity
      if(m_n_lg_trees >= ML_MaxTrees)
      {
         int old_off  = m_lg_offsets[0];
         int old_size = m_lg_sizes[0];
         int nn = m_buffer.Count();
         double ef[];
         ArrayResize(ef, m_n_features);
         for(int i = 0; i < nn; i++)
         {
            m_buffer.GetFeatures(i, ef);
            double ep = ML_PredictFlatTreeV2(m_lg_trees, old_off, ef, m_n_features);
            m_preds[i] -= ML_LearningRate * m_tree_weights[0] * ep;
         }
         ArrayCopy(m_lg_trees, m_lg_trees, 0, old_size, m_lg_alloc - old_size);
         for(int t = 1; t < m_n_lg_trees; t++)
         {
            m_lg_offsets[t - 1] = m_lg_offsets[t] - old_size;
            m_lg_sizes[t - 1]   = m_lg_sizes[t];
         }
         ArrayCopy(m_tree_weights, m_tree_weights, 0, 1, m_n_lg_trees - 1);
         m_n_lg_trees--;
         m_lg_alloc -= old_size;
      }

      if(m_lg_alloc + tree_size > ArraySize(m_lg_trees))
         ArrayResize(m_lg_trees, m_lg_alloc + tree_size + 1024);

      ArrayCopy(m_lg_trees, flat, m_lg_alloc, 0, tree_size);
      m_lg_offsets[m_n_lg_trees]   = m_lg_alloc;
      m_lg_sizes[m_n_lg_trees]     = tree_size;
      m_tree_weights[m_n_lg_trees] = 1.0;
      m_lg_alloc += tree_size;
      m_n_lg_trees++;
   }

   // Leaf-wise tree building using CatBoost infrastructure.
   // Called instead of BuildObliviousTree when ML_CB_GrowPolicy == 1.
   void BuildLossguideTree(int n_sub)
   {
      int n_buf = m_buffer.Count();
      int n_use = MathMin(n_sub, n_buf);

      // Column subsampling for this tree
      int n_sel = (int)(m_n_features * ML_ColsampleTree);
      if(n_sel < 1) n_sel = 1;
      ArrayResize(m_lg_sel_features, n_sel);
      int all_feats[];
      ArrayResize(all_feats, m_n_features);
      for(int i = 0; i < m_n_features; i++) all_feats[i] = i;
      ML_Shuffle(all_feats, m_n_features);
      ArrayCopy(m_lg_sel_features, all_feats, 0, 0, n_sel);

      // Histogram strides (+1 for NaN bin)
      int stride_feat = m_bins.NBins() + 1;
      int stride_leaf = n_sel * stride_feat;

      // Ensure histogram buffers are large enough
      // Each split produces 2 child histograms; max splits = ML_MaxLeaves - 1
      // Plus root = 1 + 2*(ML_MaxLeaves-1) = 2*ML_MaxLeaves - 1
      int hist_needed = (2 * ML_MaxLeaves + 1) * stride_leaf;
      if(ArraySize(m_lg_hist_grad) < hist_needed)
      {
         ArrayResize(m_lg_hist_grad,  hist_needed);
         ArrayResize(m_lg_hist_hess,  hist_needed);
         ArrayResize(m_lg_hist_count, hist_needed);
      }
      if(ArraySize(m_lg_sample_leaf) < n_buf)
         ArrayResize(m_lg_sample_leaf, n_buf);

      // --- Root node ---
      ArrayInitialize(m_lg_sample_leaf, -1);
      double root_g = 0.0, root_h = 0.0;
      int root_count = 0;
      for(int i = 0; i < n_use; i++)
      {
         m_lg_sample_leaf[i] = 0;
         root_g += m_grads[i];
         root_h += m_hess[i];
         root_count++;
      }

      int n_nodes = 0;
      ZeroMemory(m_lg_nodes[0]);
      m_lg_nodes[0].is_leaf      = true;
      m_lg_nodes[0].active_group = -1;
      m_lg_nodes[0].grad_sum     = root_g;
      m_lg_nodes[0].hess_sum     = root_h;
      m_lg_nodes[0].sample_count = root_count;
      m_lg_nodes[0].best_gain    = -1e30;
      m_lg_nodes[0].hist_idx     = 0;
      n_nodes = 1;

      // Build root histogram (one pass)
      ArrayFill(m_lg_hist_grad,  0, stride_leaf, 0.0);
      ArrayFill(m_lg_hist_hess,  0, stride_leaf, 0.0);
      ArrayFill(m_lg_hist_count, 0, stride_leaf, 0);
      for(int i = 0; i < n_use; i++)
      {
         for(int fl = 0; fl < n_sel; fl++)
         {
            int f   = m_lg_sel_features[fl];
            int bin = m_bins.GetBin(f, m_buffer.GetFeature(i, f));
            int idx = LG_HistIdx(0, fl, bin, stride_leaf, stride_feat);
            m_lg_hist_grad[idx]  += m_grads[i];
            m_lg_hist_hess[idx]  += m_hess[i];
            m_lg_hist_count[idx]++;
         }
      }

      // Find best split for root
      double bg; int bf, bb; double nan_dir;
      LG_FindBestSplit(0, root_g, root_h, root_count, n_sel, stride_leaf, stride_feat,
                       bg, bf, bb, nan_dir, -1);
      m_lg_nodes[0].best_gain     = bg;
      m_lg_nodes[0].best_feature  = (bf >= 0) ? m_lg_sel_features[bf] : -1;
      m_lg_nodes[0].best_bin      = bb;
      m_lg_nodes[0].nan_direction = nan_dir;
      if(bf >= 0)
         m_lg_nodes[0].best_threshold = m_bins.GetBinThreshold(m_lg_sel_features[bf], bb);

      int next_hist_slot = 1;
      int n_leaves       = 1;

      // --- Priority loop ---
      while(n_leaves < ML_MaxLeaves && n_nodes < ML_MAX_TREE_NODES - 2)
      {
         int best_leaf = -1;
         double best_leaf_gain = -1e30;
         for(int i = 0; i < n_nodes; i++)
         {
            if(!m_lg_nodes[i].is_leaf) continue;
            if(m_lg_nodes[i].sample_count < 2 * ML_MinChildSamples) continue;
            if(m_lg_nodes[i].best_gain > best_leaf_gain && m_lg_nodes[i].best_gain > 0.0)
            { best_leaf_gain = m_lg_nodes[i].best_gain; best_leaf = i; }
         }
         if(best_leaf < 0) break;

         int L         = best_leaf;
         int left_idx  = n_nodes++;
         int right_idx = n_nodes++;

         m_lg_nodes[L].is_leaf     = false;
         m_lg_nodes[L].feature_idx = m_lg_nodes[L].best_feature;
         m_lg_nodes[L].threshold   = m_lg_nodes[L].best_threshold;
         m_lg_nodes[L].left_child  = left_idx;
         m_lg_nodes[L].right_child = right_idx;

         // Find local index of split feature
         int split_feat_local = -1;
         for(int fl = 0; fl < n_sel; fl++)
            if(m_lg_sel_features[fl] == m_lg_nodes[L].feature_idx)
            { split_feat_local = fl; break; }

         // Left child stats from prefix sum
         double left_g  = 0.0, left_h  = 0.0;
         int left_count = 0;
         int parent_hist = m_lg_nodes[L].hist_idx;
         for(int b = 0; b <= m_lg_nodes[L].best_bin; b++)
         {
            int idx = LG_HistIdx(parent_hist, split_feat_local, b, stride_leaf, stride_feat);
            left_g     += m_lg_hist_grad[idx];
            left_h     += m_lg_hist_hess[idx];
            left_count += m_lg_hist_count[idx];
         }
         // Add NaN bin to the correct side
         int nan_hist_idx = LG_HistIdx(parent_hist, split_feat_local,
                                       m_bins.GetNaNBin(), stride_leaf, stride_feat);
         if(m_lg_nodes[L].nan_direction > 0.5)
         {
            left_g     += m_lg_hist_grad[nan_hist_idx];
            left_h     += m_lg_hist_hess[nan_hist_idx];
            left_count += m_lg_hist_count[nan_hist_idx];
         }
         double right_g  = m_lg_nodes[L].grad_sum    - left_g;
         double right_h  = m_lg_nodes[L].hess_sum    - left_h;
         int right_count = m_lg_nodes[L].sample_count - left_count;

         // Propagate interaction group
         int child_group = m_lg_nodes[L].active_group;
         if(m_has_interaction && child_group < 0 && m_lg_nodes[L].feature_idx >= 0)
         {
            int fg = m_interaction_groups[m_lg_nodes[L].feature_idx];
            if(fg >= 0) child_group = fg;
         }

         ZeroMemory(m_lg_nodes[left_idx]);
         m_lg_nodes[left_idx].is_leaf       = true;
         m_lg_nodes[left_idx].grad_sum      = left_g;
         m_lg_nodes[left_idx].hess_sum      = left_h;
         m_lg_nodes[left_idx].sample_count  = left_count;
         m_lg_nodes[left_idx].best_gain     = -1e30;
         m_lg_nodes[left_idx].active_group  = child_group;

         ZeroMemory(m_lg_nodes[right_idx]);
         m_lg_nodes[right_idx].is_leaf      = true;
         m_lg_nodes[right_idx].grad_sum     = right_g;
         m_lg_nodes[right_idx].hess_sum     = right_h;
         m_lg_nodes[right_idx].sample_count = right_count;
         m_lg_nodes[right_idx].best_gain    = -1e30;
         m_lg_nodes[right_idx].active_group = child_group;

         bool left_smaller = (left_count <= right_count);
         int smaller_node  = left_smaller ? left_idx  : right_idx;
         int larger_node   = left_smaller ? right_idx : left_idx;
         int smaller_hist  = next_hist_slot++;
         int larger_hist   = next_hist_slot++;
         m_lg_nodes[smaller_node].hist_idx = smaller_hist;
         m_lg_nodes[larger_node].hist_idx  = larger_hist;

         // Grow histogram buffer if needed
         int needed = next_hist_slot * stride_leaf;
         if(needed > ArraySize(m_lg_hist_grad))
         {
            ArrayResize(m_lg_hist_grad,  needed + stride_leaf);
            ArrayResize(m_lg_hist_hess,  needed + stride_leaf);
            ArrayResize(m_lg_hist_count, needed + stride_leaf);
         }

         // Clear smaller-child histogram
         int sm_start = smaller_hist * stride_leaf;
         ArrayFill(m_lg_hist_grad,  sm_start, stride_leaf, 0.0);
         ArrayFill(m_lg_hist_hess,  sm_start, stride_leaf, 0.0);
         ArrayFill(m_lg_hist_count, sm_start, stride_leaf, 0);

         // One-pass: reassign samples + build smaller-child histogram
         for(int i = 0; i < n_use; i++)
         {
            if(m_lg_sample_leaf[i] != L) continue;
            double fval  = m_buffer.GetFeature(i, m_lg_nodes[L].feature_idx);
            bool is_nan  = (fval <= ML_NAN_SENTINEL + 1e300);
            bool go_left = is_nan ? (m_lg_nodes[L].nan_direction > 0.5)
                                  : (fval < m_lg_nodes[L].threshold);
            m_lg_sample_leaf[i] = go_left ? left_idx : right_idx;

            if(m_lg_sample_leaf[i] == smaller_node)
            {
               for(int fl = 0; fl < n_sel; fl++)
               {
                  int f    = m_lg_sel_features[fl];
                  int bin  = m_bins.GetBin(f, m_buffer.GetFeature(i, f));
                  int hidx = LG_HistIdx(smaller_hist, fl, bin, stride_leaf, stride_feat);
                  m_lg_hist_grad[hidx]  += m_grads[i];
                  m_lg_hist_hess[hidx]  += m_hess[i];
                  m_lg_hist_count[hidx]++;
               }
            }
         }

         // Histogram subtraction for larger child
         int lg_start = larger_hist  * stride_leaf;
         int pa_start = parent_hist  * stride_leaf;
         for(int i = 0; i < stride_leaf; i++)
         {
            m_lg_hist_grad[lg_start + i]  = m_lg_hist_grad[pa_start + i]  - m_lg_hist_grad[sm_start + i];
            m_lg_hist_hess[lg_start + i]  = m_lg_hist_hess[pa_start + i]  - m_lg_hist_hess[sm_start + i];
            m_lg_hist_count[lg_start + i] = m_lg_hist_count[pa_start + i] - m_lg_hist_count[sm_start + i];
         }

         // FindBestSplit for each child
         if(left_count >= ML_MinChildSamples)
         {
            LG_FindBestSplit(m_lg_nodes[left_idx].hist_idx, left_g, left_h, left_count,
                             n_sel, stride_leaf, stride_feat, bg, bf, bb, nan_dir,
                             m_lg_nodes[left_idx].active_group);
            m_lg_nodes[left_idx].best_gain     = bg;
            m_lg_nodes[left_idx].best_feature  = (bf >= 0) ? m_lg_sel_features[bf] : -1;
            m_lg_nodes[left_idx].best_bin      = bb;
            m_lg_nodes[left_idx].nan_direction = nan_dir;
            if(bf >= 0)
               m_lg_nodes[left_idx].best_threshold =
                  m_bins.GetBinThreshold(m_lg_sel_features[bf], bb);
         }
         if(right_count >= ML_MinChildSamples)
         {
            LG_FindBestSplit(m_lg_nodes[right_idx].hist_idx, right_g, right_h, right_count,
                             n_sel, stride_leaf, stride_feat, bg, bf, bb, nan_dir,
                             m_lg_nodes[right_idx].active_group);
            m_lg_nodes[right_idx].best_gain     = bg;
            m_lg_nodes[right_idx].best_feature  = (bf >= 0) ? m_lg_sel_features[bf] : -1;
            m_lg_nodes[right_idx].best_bin      = bb;
            m_lg_nodes[right_idx].nan_direction = nan_dir;
            if(bf >= 0)
               m_lg_nodes[right_idx].best_threshold =
                  m_bins.GetBinThreshold(m_lg_sel_features[bf], bb);
         }

         n_leaves++;
      }

      // Compute leaf values
      for(int i = 0; i < n_nodes; i++)
         if(m_lg_nodes[i].is_leaf)
            m_lg_nodes[i].leaf_value = ML_LeafValue(m_lg_nodes[i].grad_sum,
                                                     m_lg_nodes[i].hess_sum,
                                                     ML_L1Reg, ML_L2Reg);

      // Gain-based feature importance
      for(int i = 0; i < n_nodes; i++)
         if(!m_lg_nodes[i].is_leaf)
         {
            int f = m_lg_nodes[i].feature_idx;
            if(f >= 0 && f < m_n_features)
               m_feature_importance[f] += m_lg_nodes[i].best_gain;
         }

      LG_AddTreeToEnsemble(0);
   }

public:
   CCatBoost_Model() : m_n_trees(0), m_depth(0), m_leaves_per_tree(0), m_uses_nan_dirs(false),
                       m_n_lg_trees(0), m_lg_alloc(0) {}
   ~CCatBoost_Model() {}

   void Init(ENUM_ML_TASK task, int n_features)
   {
      InitBase(task, n_features);
      m_n_trees = 0;
      m_depth = MathMin(ML_MaxDepth, ML_MAX_CB_DEPTH);
      m_leaves_per_tree = 1 << m_depth;

      m_bins.Init(n_features, ML_NBins);

      int max_trees = ML_MaxTrees;
      ArrayResize(m_split_features, max_trees * m_depth);
      ArrayResize(m_split_thresholds, max_trees * m_depth);
      ArrayResize(m_leaf_values, max_trees * m_leaves_per_tree);

      int max_samples = ML_BufferCapacity;
      ArrayResize(m_preds, max_samples);
      ArrayResize(m_grads, max_samples);
      ArrayResize(m_hess, max_samples);
      ArrayResize(m_sample_leaf, max_samples);
      ArrayResize(m_leaf_grad, m_leaves_per_tree);
      ArrayResize(m_leaf_hess, m_leaves_per_tree);
      ArrayResize(m_leaf_count, m_leaves_per_tree);

      // Histogram buffers for oblivious split finding
      int hist_size = m_leaves_per_tree * n_features * (ML_NBins + 1);  // +1 for NaN bin
      ArrayResize(m_ob_hist_grad, hist_size);
      ArrayResize(m_ob_hist_hess, hist_size);
      ArrayResize(m_ob_hist_count, hist_size);

      ArrayResize(m_tree_weights, ML_MaxTrees + 10);
      ArrayInitialize(m_tree_weights, 1.0);

      ArrayResize(m_nan_directions, ML_MaxTrees * m_depth);
      ArrayInitialize(m_nan_directions, ML_NAN_GOES_RIGHT);
      m_uses_nan_dirs = false;

      ArrayResize(m_recompute_feats, n_features);

      m_preds_buf_count = 0;
      m_preds_buf_wpos = 0;

      // Lossguide storage (only used when ML_CB_GrowPolicy == 1)
      if(ML_CB_GrowPolicy == 1)
      {
         ArrayResize(m_lg_trees, 4096);
         ArrayResize(m_lg_offsets, ML_MaxTrees + 10);
         ArrayResize(m_lg_sizes,   ML_MaxTrees + 10);
         ArrayResize(m_lg_nodes,   ML_MAX_TREE_NODES);
         m_n_lg_trees = 0;
         m_lg_alloc   = 0;
      }

      ML_SeedRng((ulong)GetTickCount() + 2);
   }

   virtual double Predict(const double &features[], int count)
   {
      double pred = m_base_score;
      if(ML_CB_GrowPolicy == 1)
      {
         // Lossguide: flat V2 trees
         for(int t = 0; t < m_n_lg_trees; t++)
            pred += ML_LearningRate * m_tree_weights[t]
                    * ML_PredictFlatTreeV2(m_lg_trees, m_lg_offsets[t], features, m_n_features);
      }
      else
      {
         // Symmetric (oblivious) trees
         for(int t = 0; t < m_n_trees; t++)
         {
            int leaf_idx = ComputeLeafIndexFromFeatures(t, features);
            pred += ML_LearningRate * m_tree_weights[t] * m_leaf_values[t * m_leaves_per_tree + leaf_idx];
         }
      }
      return pred;
   }

   // Recompute m_preds[i] from scratch for buffer sample i (dispatches on grow policy)
   void RecomputePredForSample(int i)
   {
      m_preds[i] = m_base_score;
      if(ML_CB_GrowPolicy == 1)
      {
         // Lossguide: use flat V2 trees (m_recompute_feats pre-allocated in Init)
         m_buffer.GetFeatures(i, m_recompute_feats);
         for(int t = 0; t < m_n_lg_trees; t++)
            m_preds[i] += ML_LearningRate * m_tree_weights[t] * ML_PredictFlatTreeV2(m_lg_trees, m_lg_offsets[t], m_recompute_feats, m_n_features);
      }
      else
      {
         // Symmetric: use oblivious tree evaluation
         for(int t = 0; t < m_n_trees; t++)
         {
            int leaf_idx = ComputeLeafIndex(i, t);
            m_preds[i] += ML_LearningRate * m_tree_weights[t] * m_leaf_values[t * m_leaves_per_tree + leaf_idx];
         }
      }
   }

   virtual void Train(bool force = false)
   {
      int n = m_buffer.Count();
      if(!force && n < ML_ColdStartMin) return;
      if(m_early_stopped) return;  // Already stopped

      if(!m_bins.IsComputed())
      {
         m_bins.Compute(m_buffer);
         if(!m_bins.IsComputed()) return;
      }

      if(m_train_count == 0) ComputeBaseScore();

      // Recompute predictions (incremental where possible)
      int total_trees = (ML_CB_GrowPolicy == 1) ? m_n_lg_trees : m_n_trees;
      if(total_trees == 0 || m_preds_buf_count == 0)
      {
         // Full recompute — cold start or no trees yet
         for(int i = 0; i < n; i++)
            RecomputePredForSample(i);
      }
      else
      {
         // Incremental — only recompute dirty (newly written) samples
         if(n > m_preds_buf_count)
         {
            // Buffer grew: dirty = [m_preds_buf_count, n)
            for(int i = m_preds_buf_count; i < n; i++)
               RecomputePredForSample(i);
         }
         else
         {
            // Buffer full, wrapping — dirty = overwritten region
            int new_wpos = m_buffer.WritePos();
            int old_wpos = m_preds_buf_wpos;
            if(new_wpos != old_wpos)
            {
               if(new_wpos >= old_wpos)
               {
                  for(int i = old_wpos; i < new_wpos; i++)
                     RecomputePredForSample(i);
               }
               else
               {
                  for(int i = old_wpos; i < n; i++)
                     RecomputePredForSample(i);
                  for(int i = 0; i < new_wpos; i++)
                     RecomputePredForSample(i);
               }
            }
         }
      }
      m_preds_buf_count = n;
      m_preds_buf_wpos = m_buffer.WritePos();

      ComputeGradHess(m_preds, m_grads, m_hess, n);

      // Ordered boosting gradient correction (CatBoost's core anti-leakage innovation)
      // Skip on first training call — no model predictions to correct against yet
      if(total_trees > 0)
         ComputeOrderedGradients(n);

      // Bayesian bootstrap: soft-weight all samples with Exp(1) random weights
      if(ML_CB_BaggingTemp > 0.0)
      {
         double total_w = 0.0;
         for(int i = 0; i < n; i++)
         {
            double u = MathMax(ML_RandDouble(), ML_EPS);
            double ew = MathPow(-MathLog(u), ML_CB_BaggingTemp);
            m_grads[i] *= ew;
            m_hess[i]  *= ew;
            total_w += ew;
         }
         // Normalize to preserve gradient scale
         double scale = (double)n / total_w;
         for(int i = 0; i < n; i++)
         {
            m_grads[i] *= scale;
            m_hess[i]  *= scale;
         }
      }

      // Row subsampling
      int n_sub = (int)(n * ML_Subsample);
      if(n_sub < ML_MinChildSamples) n_sub = n;

      double cat_l2 = ML_L2Reg;

      for(int round = 0; round < ML_TreesPerRound; round++)
      {
         // Drop oldest tree if at capacity (symmetric path only;
         // lossguide eviction is handled inside LG_AddTreeToEnsemble)
         if(ML_CB_GrowPolicy == 0 && m_n_trees >= ML_MaxTrees)
         {
            // Subtract evicted tree (tree 0) contribution from all predictions
            for(int ei = 0; ei < n; ei++)
            {
               int evict_leaf = ComputeLeafIndex(ei, 0);
               m_preds[ei] -= ML_LearningRate * m_tree_weights[0] * m_leaf_values[evict_leaf];
            }

            // Shift all tree data left by 1 tree
            int splits_per = m_depth;
            int leaves_per = m_leaves_per_tree;
            ArrayCopy(m_split_features, m_split_features, 0, splits_per,
                      (m_n_trees - 1) * splits_per);
            ArrayCopy(m_split_thresholds, m_split_thresholds, 0, splits_per,
                      (m_n_trees - 1) * splits_per);
            ArrayCopy(m_nan_directions, m_nan_directions, 0, splits_per,
                      (m_n_trees - 1) * splits_per);
            ArrayCopy(m_leaf_values, m_leaf_values, 0, leaves_per,
                      (m_n_trees - 1) * leaves_per);
            ArrayCopy(m_tree_weights, m_tree_weights, 0, 1, m_n_trees);
            m_n_trees--;
         }

         // DART: drop random trees before building
         int dart_dropped[];
         int n_dart_dropped = 0;
         int n_trees_for_dart = (ML_CB_GrowPolicy == 1) ? m_n_lg_trees : m_n_trees;
         if(ML_DARTEnabled && n_trees_for_dart > 0)
         {
            int n_drop = MathMax(1, (int)(n_trees_for_dart * ML_DARTDropRate));
            if(n_drop > n_trees_for_dart) n_drop = n_trees_for_dart;

            int all_indices[];
            ArrayResize(all_indices, n_trees_for_dart);
            for(int di = 0; di < n_trees_for_dart; di++) all_indices[di] = di;
            ML_Shuffle(all_indices, n_trees_for_dart);

            ArrayResize(dart_dropped, n_drop);
            n_dart_dropped = n_drop;
            for(int di = 0; di < n_drop; di++) dart_dropped[di] = all_indices[di];

            if(ML_CB_GrowPolicy == 1)
            {
               // Lossguide: subtract dropped trees using flat V2 format
               double feats_dart[];
               ArrayResize(feats_dart, m_n_features);
               for(int di = 0; di < n_drop; di++)
               {
                  int dt = dart_dropped[di];
                  double w = m_tree_weights[dt];
                  for(int si = 0; si < n; si++)
                  {
                     m_buffer.GetFeatures(si, feats_dart);
                     m_preds[si] -= ML_LearningRate * w
                                    * ML_PredictFlatTreeV2(m_lg_trees, m_lg_offsets[dt],
                                                           feats_dart, m_n_features);
                  }
               }
            }
            else
            {
               // Symmetric: subtract dropped trees using oblivious tree eval
               for(int di = 0; di < n_drop; di++)
               {
                  int dt = dart_dropped[di];
                  double w = m_tree_weights[dt];
                  int dt_leaf_off = dt * m_leaves_per_tree;
                  for(int si = 0; si < n; si++)
                  {
                     int leaf_idx = ComputeLeafIndex(si, dt);
                     m_preds[si] -= ML_LearningRate * w * m_leaf_values[dt_leaf_off + leaf_idx];
                  }
               }
            }

            // Recompute gradients on modified predictions
            ComputeGradHess(m_preds, m_grads, m_hess, n);
         }

         if(ML_CB_GrowPolicy == 1)
         {
            // Lossguide: leaf-wise trees stored in flat V2 format
            BuildLossguideTree(n_sub);

            // Update predictions with new lossguide tree
            int new_lg = m_n_lg_trees - 1;
            double feat_lg[];
            ArrayResize(feat_lg, m_n_features);
            for(int i = 0; i < n; i++)
            {
               m_buffer.GetFeatures(i, feat_lg);
               m_preds[i] += ML_LearningRate *
                             ML_PredictFlatTreeV2(m_lg_trees, m_lg_offsets[new_lg],
                                                  feat_lg, m_n_features);
            }

            // DART: restore dropped lossguide trees with rescaled weights
            if(n_dart_dropped > 0)
            {
               double scale_existing = (double)n_dart_dropped / (double)(n_dart_dropped + 1);
               double scale_new = 1.0 / (double)(n_dart_dropped + 1);
               double fd[];
               ArrayResize(fd, m_n_features);
               for(int di = 0; di < n_dart_dropped; di++)
               {
                  int dt = dart_dropped[di];
                  m_tree_weights[dt] *= scale_existing;
                  for(int si = 0; si < n; si++)
                  {
                     m_buffer.GetFeatures(si, fd);
                     m_preds[si] += ML_LearningRate * m_tree_weights[dt]
                                    * ML_PredictFlatTreeV2(m_lg_trees, m_lg_offsets[dt], fd, m_n_features);
                  }
               }
               m_tree_weights[new_lg] = scale_new;
               for(int si = 0; si < n; si++)
               {
                  m_buffer.GetFeatures(si, fd);
                  double tp = ML_PredictFlatTreeV2(m_lg_trees, m_lg_offsets[new_lg], fd, m_n_features);
                  m_preds[si] -= ML_LearningRate * (1.0 - scale_new) * tp;
               }
            }
         }
         else
         {
            // Symmetric (oblivious) trees
            m_uses_nan_dirs = true;
            BuildObliviousTree(n, n_sub, cat_l2);

            // Update predictions with new tree
            int t = m_n_trees - 1;
            int leaf_off = t * m_leaves_per_tree;
            for(int i = 0; i < n; i++)
            {
               int leaf_idx = ComputeLeafIndex(i, t);
               m_preds[i] += ML_LearningRate * m_leaf_values[leaf_off + leaf_idx];
            }

            // DART: restore dropped trees with rescaled weights
            if(n_dart_dropped > 0)
            {
               int new_tree = m_n_trees - 1;
               double scale_existing = (double)n_dart_dropped / (double)(n_dart_dropped + 1);
               double scale_new = 1.0 / (double)(n_dart_dropped + 1);

               for(int di = 0; di < n_dart_dropped; di++)
               {
                  int dt = dart_dropped[di];
                  m_tree_weights[dt] *= scale_existing;
                  int dt_leaf_off = dt * m_leaves_per_tree;
                  for(int si = 0; si < n; si++)
                  {
                     int leaf_idx = ComputeLeafIndex(si, dt);
                     m_preds[si] += ML_LearningRate * m_tree_weights[dt] * m_leaf_values[dt_leaf_off + leaf_idx];
                  }
               }

               m_tree_weights[new_tree] = scale_new;
               // Adjust new tree prediction: was added with weight 1.0, need scale_new
               int nt_leaf_off = new_tree * m_leaves_per_tree;
               for(int si = 0; si < n; si++)
               {
                  int leaf_idx = ComputeLeafIndex(si, new_tree);
                  m_preds[si] -= ML_LearningRate * (1.0 - scale_new) * m_leaf_values[nt_leaf_off + leaf_idx];
               }
            }
         }

         if(round < ML_TreesPerRound - 1)
         {
            ComputeGradHess(m_preds, m_grads, m_hess, n);
            ComputeOrderedGradients(n);
         }
      }

      // Check early stopping
      CheckEarlyStop(m_preds, n);

      m_train_count++;
   }

   void BuildObliviousTree(int n_buf, int n_sub, double l2)
   {
      int tree_idx = m_n_trees;

      // C4 fix: Apply row subsampling — use random n_sub rows for split finding
      int n_use = MathMin(n_sub, n_buf);

      // Init: all samples in leaf 0
      ArrayInitialize(m_sample_leaf, 0);
      int n_current_leaves = 1;

      int split_off = tree_idx * m_depth;

      for(int d = 0; d < m_depth; d++)
      {
         // C5 fix: Compute leaf counts + grad/hess sums BEFORE split search
         // (histogram split needs per-leaf totals for right-side subtraction)
         ArrayInitialize(m_leaf_count, 0);
         ArrayInitialize(m_leaf_grad, 0.0);
         ArrayInitialize(m_leaf_hess, 0.0);
         for(int i = 0; i < n_use; i++)
         {
            int leaf = m_sample_leaf[i];
            if(leaf >= 0 && leaf < m_leaves_per_tree)
            {
               m_leaf_count[leaf]++;
               m_leaf_grad[leaf] += m_grads[i];
               m_leaf_hess[leaf] += m_hess[i];
            }
         }

         // Find best (feature, threshold, nan_direction) for this level
         // Interaction constraint: after level 0, lock to the feature group chosen
         int tree_active_group = -1;
         if(m_has_interaction && d > 0)
         {
            // Scan ALL previous levels to find the first constrained group
            for(int dd = 0; dd < d; dd++)
            {
               int prev_feat = m_split_features[split_off + dd];
               if(prev_feat >= 0)
               {
                  int fg = m_interaction_groups[prev_feat];
                  if(fg >= 0) { tree_active_group = fg; break; }
               }
            }
         }
         int bf;
         double bt;
         double nan_dir;
         FindBestObliviousSplit(n_use, d, n_current_leaves, bf, bt, nan_dir, tree_active_group);

         m_split_features[split_off + d] = bf;
         m_split_thresholds[split_off + d] = bt;
         m_nan_directions[split_off + d] = nan_dir;

         if(bf < 0)
         {
            // No good split found; fill remaining levels with dummy
            for(int dd = d; dd < m_depth; dd++)
            {
               m_split_features[split_off + dd] = 0;
               m_split_thresholds[split_off + dd] = -1e30; // everything goes right
               m_nan_directions[split_off + dd] = ML_NAN_GOES_RIGHT;
            }
            break;
         }

         // Update sample-to-leaf: each leaf splits into two (NaN-aware)
         for(int i = 0; i < n_buf; i++)
         {
            double fval = m_buffer.GetFeature(i, bf);
            bool is_nan = (fval <= ML_NAN_SENTINEL + 1e300);
            bool go_right = is_nan ? (nan_dir < 0.5) : (fval >= bt);
            if(go_right)
               m_sample_leaf[i] |= (1 << d);
         }
         n_current_leaves *= 2;
      }

      // Compute leaf values: accumulate gradients per leaf
      ArrayInitialize(m_leaf_grad, 0.0);
      ArrayInitialize(m_leaf_hess, 0.0);
      ArrayInitialize(m_leaf_count, 0);

      for(int i = 0; i < n_buf; i++)
      {
         int leaf = m_sample_leaf[i];
         if(leaf >= 0 && leaf < m_leaves_per_tree)
         {
            m_leaf_grad[leaf] += m_grads[i];
            m_leaf_hess[leaf] += m_hess[i];
            m_leaf_count[leaf]++;
         }
      }

      int leaf_off = tree_idx * m_leaves_per_tree;
      for(int leaf = 0; leaf < m_leaves_per_tree; leaf++)
         m_leaf_values[leaf_off + leaf] = ML_LeafValue(m_leaf_grad[leaf],
                                                        m_leaf_hess[leaf],
                                                        ML_L1Reg, l2);

      // Feature importance
      for(int d = 0; d < m_depth; d++)
      {
         int f = m_split_features[split_off + d];
         if(f >= 0 && f < m_n_features)
            m_feature_importance[f] += 1.0; // count-based for oblivious trees
      }

      m_n_trees++;
   }

   virtual bool SaveToFile(string filename)
   {
      int handle = FileOpen(filename, FILE_WRITE | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE) return false;

      FileWriteInteger(handle, ML_MAGIC_SAVE);
      FileWriteInteger(handle, (int)ML_CATBOOST);
      FileWriteInteger(handle, (int)m_task);
      FileWriteInteger(handle, m_n_features);
      FileWriteInteger(handle, m_n_trees);
      FileWriteInteger(handle, m_depth);
      FileWriteDouble(handle, m_base_score);
      FileWriteInteger(handle, m_train_count);

      int splits_total = m_n_trees * m_depth;
      int leaves_total = m_n_trees * m_leaves_per_tree;
      FileWriteArray(handle, m_split_features, 0, splits_total);
      FileWriteArray(handle, m_split_thresholds, 0, splits_total);
      FileWriteArray(handle, m_leaf_values, 0, leaves_total);

      FileWriteArray(handle, m_feature_importance, 0, m_n_features);
      m_bins.Save(handle);

      // GBT V2 extension block — early stopping state
      FileWriteInteger(handle, ML_MAGIC_GBT_V2);
      FileWriteDouble(handle, m_best_val_loss);
      FileWriteInteger(handle, m_no_improve_rounds);
      FileWriteInteger(handle, m_early_stopped ? 1 : 0);

      // DART weights
      FileWriteInteger(handle, ML_MAGIC_DART);
      FileWriteArray(handle, m_tree_weights, 0, m_n_trees);

      // GBT V3 extension block — NaN direction learning
      FileWriteInteger(handle, ML_MAGIC_GBT_V3);
      FileWriteInteger(handle, m_uses_nan_dirs ? 1 : 0);
      if(m_uses_nan_dirs)
         FileWriteArray(handle, m_nan_directions, 0, m_n_trees * m_depth);

      // GBT V4 extension block — grow policy + lossguide trees
      FileWriteInteger(handle, ML_MAGIC_V4);
      FileWriteInteger(handle, ML_CB_GrowPolicy);
      if(ML_CB_GrowPolicy == 1)
      {
         FileWriteInteger(handle, m_n_lg_trees);
         FileWriteInteger(handle, m_lg_alloc);
         if(m_lg_alloc > 0)
            FileWriteArray(handle, m_lg_trees, 0, m_lg_alloc);
         if(m_n_lg_trees > 0)
         {
            FileWriteArray(handle, m_lg_offsets, 0, m_n_lg_trees);
            FileWriteArray(handle, m_lg_sizes, 0, m_n_lg_trees);
            FileWriteArray(handle, m_tree_weights, 0, m_n_lg_trees);  // DART weights for lossguide
         }
      }

      FileWriteInteger(handle, ML_MAGIC_END);
      FileClose(handle);
      return true;
   }

   virtual bool LoadFromFile(string filename)
   {
      int handle = FileOpen(filename, FILE_READ | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE) return false;

      int magic = FileReadInteger(handle);
      if(magic != ML_MAGIC_SAVE) { FileClose(handle); return false; }
      int model_type = FileReadInteger(handle);
      if(model_type != (int)ML_CATBOOST) { FileClose(handle); return false; }

      m_task = (ENUM_ML_TASK)FileReadInteger(handle);
      m_n_features = FileReadInteger(handle);
      m_n_trees = FileReadInteger(handle);
      m_depth = FileReadInteger(handle);
      m_leaves_per_tree = 1 << m_depth;
      m_base_score = FileReadDouble(handle);
      m_train_count = FileReadInteger(handle);

      int splits_total = m_n_trees * m_depth;
      int leaves_total = m_n_trees * m_leaves_per_tree;
      int max_splits = ML_MaxTrees * m_depth;
      int max_leaves = ML_MaxTrees * m_leaves_per_tree;
      ArrayResize(m_split_features, MathMax(max_splits, 1));
      ArrayResize(m_split_thresholds, MathMax(max_splits, 1));
      ArrayResize(m_leaf_values, MathMax(max_leaves, 1));
      FileReadArray(handle, m_split_features, 0, splits_total);
      FileReadArray(handle, m_split_thresholds, 0, splits_total);
      FileReadArray(handle, m_leaf_values, 0, leaves_total);

      ArrayResize(m_feature_importance, m_n_features);
      FileReadArray(handle, m_feature_importance, 0, m_n_features);
      m_bins.Load(handle);

      // Try reading GBT V2 extension block
      int next_magic = FileReadInteger(handle);
      if(next_magic == ML_MAGIC_GBT_V2)
      {
         m_best_val_loss = FileReadDouble(handle);
         m_no_improve_rounds = FileReadInteger(handle);
         m_early_stopped = (FileReadInteger(handle) != 0);
         next_magic = FileReadInteger(handle);
      }
      else
      {
         // Old format — use defaults
         m_best_val_loss = 1e30;
         m_no_improve_rounds = 0;
         m_early_stopped = false;
      }
      // Try reading DART weights
      if(next_magic == ML_MAGIC_DART)
      {
         ArrayResize(m_tree_weights, ML_MaxTrees + 10);
         FileReadArray(handle, m_tree_weights, 0, m_n_trees);
         next_magic = FileReadInteger(handle);
      }
      else
      {
         ArrayResize(m_tree_weights, ML_MaxTrees + 10);
         ArrayInitialize(m_tree_weights, 1.0);
      }
      // Try reading GBT V3 — NaN direction learning
      if(next_magic == ML_MAGIC_GBT_V3)
      {
         m_uses_nan_dirs = (FileReadInteger(handle) != 0);
         if(m_uses_nan_dirs)
         {
            ArrayResize(m_nan_directions, ML_MaxTrees * m_depth);
            FileReadArray(handle, m_nan_directions, 0, m_n_trees * m_depth);
         }
         next_magic = FileReadInteger(handle);
      }
      else
      {
         m_uses_nan_dirs = false;
         ArrayResize(m_nan_directions, ML_MaxTrees * m_depth);
         ArrayInitialize(m_nan_directions, ML_NAN_GOES_RIGHT);
      }
      // Try reading GBT V4 — grow policy + lossguide trees
      if(next_magic == ML_MAGIC_V4)
      {
         int saved_policy = FileReadInteger(handle);
         if(saved_policy != ML_CB_GrowPolicy)
            Print("ML CB Warning: file saved with GrowPolicy=", saved_policy,
                  " but current is ", ML_CB_GrowPolicy, " — mismatched data skipped");
         if(saved_policy == 1)
         {
            m_n_lg_trees = FileReadInteger(handle);
            m_lg_alloc   = FileReadInteger(handle);
            if(m_lg_alloc > 0)
            {
               ArrayResize(m_lg_trees, m_lg_alloc + 1024);
               FileReadArray(handle, m_lg_trees, 0, m_lg_alloc);
            }
            if(m_n_lg_trees > 0)
            {
               ArrayResize(m_lg_offsets, ML_MaxTrees + 10);
               ArrayResize(m_lg_sizes, ML_MaxTrees + 10);
               FileReadArray(handle, m_lg_offsets, 0, m_n_lg_trees);
               FileReadArray(handle, m_lg_sizes, 0, m_n_lg_trees);
               // Restore DART weights for lossguide trees
               if(ML_CB_GrowPolicy == 1)
               {
                  // Only overwrite m_tree_weights if current policy is also lossguide
                  ArrayResize(m_tree_weights, ML_MaxTrees + 10);
                  ArrayInitialize(m_tree_weights, 1.0);
                  FileReadArray(handle, m_tree_weights, 0, m_n_lg_trees);
               }
               else
               {
                  // Policy mismatch: read and discard lossguide DART weights
                  double discard[];
                  ArrayResize(discard, m_n_lg_trees);
                  FileReadArray(handle, discard, 0, m_n_lg_trees);
               }
            }
         }
         next_magic = FileReadInteger(handle);
      }

      // Always reset incremental preds (buffer is not saved)
      m_preds_buf_count = 0;
      m_preds_buf_wpos = 0;

      int end_magic = next_magic;
      FileClose(handle);
      return (end_magic == ML_MAGIC_END);
   }

   virtual void Reset()
   {
      CMLModelBase::Reset();
      m_n_trees = 0;
      m_uses_nan_dirs = false;
      ArrayInitialize(m_tree_weights, 1.0);
      ArrayInitialize(m_nan_directions, ML_NAN_GOES_RIGHT);
      // Reset lossguide ensemble
      if(ML_CB_GrowPolicy == 1)
      {
         m_n_lg_trees = 0;
         m_lg_alloc   = 0;
      }
   }

   // Convert oblivious tree t into flat V2 format for SHAP_Recurse.
   // Recursively writes nodes: level d splits on the same (feature, threshold).
   // Returns next write position in flat[].
   int ObliviousToFlat(int tree_idx, int level, int leaf_start,
                       double &flat[], int pos, int tree_start)
   {
      if(level >= m_depth)
      {
         flat[pos++] = ML_LEAF_MARKER;
         flat[pos++] = m_leaf_values[tree_idx * m_leaves_per_tree + leaf_start];
         flat[pos++] = 0.0;
         flat[pos++] = 0.0;
         return pos;
      }
      int so = tree_idx * m_depth;
      int node_pos = pos;
      flat[pos++] = (double)m_split_features[so + level];
      flat[pos++] = m_split_thresholds[so + level];
      flat[pos++] = 0.0;  // right_offset placeholder
      flat[pos++] = m_uses_nan_dirs ? m_nan_directions[so + level] : ML_NAN_GOES_RIGHT;

      int half = 1 << (m_depth - level - 1);
      pos = ObliviousToFlat(tree_idx, level + 1, leaf_start, flat, pos, tree_start);
      flat[node_pos + 2] = (double)(pos - tree_start);
      pos = ObliviousToFlat(tree_idx, level + 1, leaf_start + half, flat, pos, tree_start);
      return pos;
   }

   // Compute per-feature SHAP values for a single sample.
   // phi[] sized (n_features + 1): phi[n_features] = base score intercept.
   // Both symmetric and lossguide trees use the Lundberg TreeSHAP recursion
   // via SHAP_Recurse on flat V2 tree format.
   virtual void ComputeSHAP(const double &features[], int count, double &phi[])
   {
      ArrayResize(phi, m_n_features + 1);
      ArrayInitialize(phi, 0.0);
      phi[m_n_features] = m_base_score;

      // ── Symmetric (oblivious) tree SHAP ──
      // Convert each oblivious tree to flat V2 and use SHAP_Recurse.
      // For depth D, the flat tree has (2^(D+1) - 1) nodes * 4 doubles.
      if(ML_CB_GrowPolicy == 0 && m_n_trees > 0)
      {
         int total_nodes = (m_leaves_per_tree * 2 - 1);
         double flat[];
         ArrayResize(flat, total_nodes * ML_NODE_WIDTH_V2);
         SHAPPath path;
         double tree_phi[];
         ArrayResize(tree_phi, m_n_features);

         for(int t = 0; t < m_n_trees; t++)
         {
            int pos = 0;
            ObliviousToFlat(t, 0, 0, flat, pos, 0);

            ArrayInitialize(tree_phi, 0.0);
            ZeroMemory(path);
            path.len = 0;
            SHAP_Recurse(flat, 0, 0, features, m_n_features,
                         tree_phi, path, 1.0, 1.0, -1);

            double scale = ML_LearningRate * m_tree_weights[t];
            for(int f = 0; f < m_n_features; f++)
               phi[f] += scale * tree_phi[f];
         }
      }

      // ── Lossguide tree SHAP (flat V2 format) ──
      // Uses SHAP_Recurse identical to LightGBM/XGBoost.
      if(ML_CB_GrowPolicy == 1)
      {
         // lossguide SHAP: handled after Wave 2 adds m_lg_trees full V2 support.
         // m_lg_trees, m_lg_offsets, m_n_lg_trees are present in this class.
         if(m_n_lg_trees > 0)
         {
            SHAPPath path;
            double tree_phi[];
            ArrayResize(tree_phi, m_n_features);

            for(int t = 0; t < m_n_lg_trees; t++)
            {
               ArrayInitialize(tree_phi, 0.0);
               ZeroMemory(path);
               path.len = 0;

               int root = m_lg_offsets[t];
               SHAP_Recurse(m_lg_trees, root, root, features, m_n_features,
                            tree_phi, path, 1.0, 1.0, -1);

               double scale = ML_LearningRate * m_tree_weights[t];
               for(int f = 0; f < m_n_features; f++)
                  phi[f] += scale * tree_phi[f];
            }
         }
      }
   }
};

//===================================================================
//===================================================================
// CDeepTabular_Model — MLP with PLE encoding, LayerNorm, SiLU
//===================================================================
//===================================================================
class CDeepTabular_Model : public CMLModelBase
{
private:
   // Architecture params
   int    m_ple_bins;
   int    m_ple_input_dim;    // n_features * ple_bins
   int    m_h1;               // hidden layer 1 size
   int    m_h2;               // hidden layer 2 size
   double m_lr;
   double m_dropout;
   int    m_batch_size;
   int    m_epochs;

   // Weights
   matrix m_w1, m_w2, m_w3;
   vector m_b1, m_b2, m_b3;
   vector m_ln_gamma1, m_ln_beta1;
   vector m_ln_gamma2, m_ln_beta2;

   // Adam state
   matrix m_mw1, m_vw1, m_mw2, m_vw2, m_mw3, m_vw3;
   vector m_mb1, m_vb1, m_mb2, m_vb2, m_mb3, m_vb3;
   vector m_mg1, m_vg1, m_mb1_ln, m_vb1_ln;
   vector m_mg2, m_vg2, m_mb2_ln, m_vb2_ln;
   int    m_adam_step;
   int    m_total_steps;     // cumulative training steps for LR scheduling

   // PLE boundaries
   double m_ple_bounds[];    // [n_features * (ple_bins + 1)]
   bool   m_ple_computed;

   // BatchEnsemble (TabM) — rank-1 perturbation vectors per member
   int    m_ensemble_k;           // effective ensemble size (1=disabled)
   // r[k][dim] perturbs input, s[k][dim] perturbs output of each layer
   matrix m_r1, m_s1;             // [k x ple_input_dim], [k x h1]
   matrix m_r2, m_s2;             // [k x h1], [k x h2]
   matrix m_r3, m_s3;             // [k x h2], [k x 1]
   // Adam moments for r/s vectors
   matrix m_r1_m, m_r1_v, m_s1_m, m_s1_v;
   matrix m_r2_m, m_r2_v, m_s2_m, m_s2_v;
   matrix m_r3_m, m_r3_v, m_s3_m, m_s3_v;

   // Forward pass cache (for backward)
   vector m_ple_out;
   vector m_z1, m_ln1, m_sig1, m_a1, m_d1;
   vector m_z2, m_ln2, m_sig2, m_a2, m_d2;
   vector m_xhat1, m_xhat2;
   double m_std_inv1, m_std_inv2;
   vector m_drop_mask1, m_drop_mask2;
   bool   m_training_mode;
   int    m_cur_member;           // current ensemble member for Forward/Backward
   // TabM backward cache: pre-perturbation values needed for r/s gradients
   vector m_ple_orig;             // PLE output before r1 perturbation
   vector m_z1_raw;               // z1 before s1 scaling
   vector m_d1_orig;              // d1 before r2 perturbation
   vector m_z2_raw;               // z2 before s2 scaling
   vector m_d2_orig;              // d2 before r3 perturbation
   double m_out_raw;              // output before s3 scaling

   void PLEEncode(const double &features[], int count, vector &out)
   {
      out.Resize(m_ple_input_dim);
      out.Fill(0.0);

      if(!m_ple_computed)
      {
         // Fallback: just replicate raw features into PLE slots
         for(int f = 0; f < m_n_features && f < count; f++)
         {
            int base = f * m_ple_bins;
            double v = features[f];
            for(int b = 0; b < m_ple_bins; b++)
               out[base + b] = v / m_ple_bins;
         }
         return;
      }

      for(int f = 0; f < m_n_features && f < count; f++)
      {
         int base = f * m_ple_bins;
         int bnd_off = f * (m_ple_bins + 1);
         double x = features[f];

         for(int b = 0; b < m_ple_bins; b++)
         {
            double lo = m_ple_bounds[bnd_off + b];
            double hi = m_ple_bounds[bnd_off + b + 1];
            double range = hi - lo;
            if(range < ML_EPS) range = ML_EPS;
            double val = (x - lo) / range;
            out[base + b] = MathMax(0.0, MathMin(1.0, val));
         }
      }
   }

   void ComputePLEBoundaries()
   {
      int n = m_buffer.Count();
      if(n < 20) return;

      ArrayResize(m_ple_bounds, m_n_features * (m_ple_bins + 1));
      double vals[];
      ArrayResize(vals, n);

      for(int f = 0; f < m_n_features; f++)
      {
         for(int i = 0; i < n; i++)
            vals[i] = m_buffer.GetFeature(i, f);
         ArraySort(vals);

         int bnd_off = f * (m_ple_bins + 1);
         m_ple_bounds[bnd_off] = vals[0]; // min
         for(int b = 1; b < m_ple_bins; b++)
         {
            double p = (double)b / (double)m_ple_bins;
            double pos = p * (n - 1);
            int lo = (int)MathFloor(pos);
            if(lo >= n - 1) lo = n - 2;
            double frac = pos - lo;
            m_ple_bounds[bnd_off + b] = vals[lo] * (1.0 - frac) + vals[lo + 1] * frac;
         }
         m_ple_bounds[bnd_off + m_ple_bins] = vals[n - 1]; // max

         // Ensure strictly increasing
         for(int b = 1; b <= m_ple_bins; b++)
            if(m_ple_bounds[bnd_off + b] <= m_ple_bounds[bnd_off + b - 1])
               m_ple_bounds[bnd_off + b] = m_ple_bounds[bnd_off + b - 1] + ML_EPS;
      }
      m_ple_computed = true;
   }

   // Forward pass for a single sample
   // When m_cur_member >= 0 and m_ensemble_k > 1, applies TabM rank-1 perturbations:
   //   z_l = W_l * (r_l[ki] ⊙ h_{l-1}) ⊙ s_l[ki] + b_l
   double Forward(const double &features[], int count)
   {
      // PLE encoding
      PLEEncode(features, count, m_ple_out);

      // TabM: perturb PLE input with r1[ki]
      int ki = m_cur_member;
      bool use_tabm = (ki >= 0 && m_ensemble_k > 1);
      if(use_tabm)
      {
         m_ple_orig = m_ple_out;  // cache before perturbation
         for(int j = 0; j < m_ple_input_dim; j++)
            m_ple_out[j] *= m_r1[ki][j];
      }

      // Layer 1: z1 = ple * W1 + b1
      m_z1 = m_ple_out.MatMul(m_w1);
      // TabM: perturb layer 1 output with s1[ki]
      if(use_tabm)
      {
         m_z1_raw = m_z1;  // cache before s1 scaling
         for(int j = 0; j < m_h1; j++)
            m_z1[j] *= m_s1[ki][j];
      }
      m_z1 += m_b1;

      // LayerNorm 1
      double mean1 = 0.0;
      int h1 = m_h1;
      for(int i = 0; i < h1; i++) mean1 += m_z1[i];
      mean1 /= h1;
      double var1 = 0.0;
      for(int i = 0; i < h1; i++) var1 += (m_z1[i] - mean1) * (m_z1[i] - mean1);
      var1 /= h1;
      m_std_inv1 = 1.0 / MathSqrt(var1 + ML_EPS);
      m_xhat1.Resize(h1);
      for(int i = 0; i < h1; i++) m_xhat1[i] = (m_z1[i] - mean1) * m_std_inv1;
      m_ln1.Resize(h1);
      for(int i = 0; i < h1; i++) m_ln1[i] = m_xhat1[i] * m_ln_gamma1[i] + m_ln_beta1[i];

      // SiLU: a1 = ln1 * sigmoid(ln1)
      m_ln1.Activation(m_sig1, AF_SIGMOID);
      m_a1 = m_ln1 * m_sig1;

      // Dropout 1
      m_d1 = m_a1;
      if(m_training_mode && m_dropout > 0)
      {
         double scale = 1.0 / (1.0 - m_dropout);
         m_drop_mask1.Resize(h1);
         for(int i = 0; i < h1; i++)
         {
            if(ML_RandDouble() < m_dropout)
            { m_d1[i] = 0.0; m_drop_mask1[i] = 0.0; }
            else
            { m_d1[i] *= scale; m_drop_mask1[i] = scale; }
         }
      }

      // TabM: perturb layer 2 input with r2[ki]
      vector d1_in = m_d1;
      if(use_tabm)
      {
         m_d1_orig = m_d1;  // cache before r2 perturbation
         for(int j = 0; j < m_h1; j++)
            d1_in[j] *= m_r2[ki][j];
      }

      // Layer 2: z2 = d1_in * W2 + b2
      m_z2 = d1_in.MatMul(m_w2);
      // TabM: perturb layer 2 output with s2[ki]
      if(use_tabm)
      {
         m_z2_raw = m_z2;  // cache before s2 scaling
         for(int j = 0; j < m_h2; j++)
            m_z2[j] *= m_s2[ki][j];
      }
      m_z2 += m_b2;

      // LayerNorm 2
      double mean2 = 0.0;
      int h2 = m_h2;
      for(int i = 0; i < h2; i++) mean2 += m_z2[i];
      mean2 /= h2;
      double var2 = 0.0;
      for(int i = 0; i < h2; i++) var2 += (m_z2[i] - mean2) * (m_z2[i] - mean2);
      var2 /= h2;
      m_std_inv2 = 1.0 / MathSqrt(var2 + ML_EPS);
      m_xhat2.Resize(h2);
      for(int i = 0; i < h2; i++) m_xhat2[i] = (m_z2[i] - mean2) * m_std_inv2;
      m_ln2.Resize(h2);
      for(int i = 0; i < h2; i++) m_ln2[i] = m_xhat2[i] * m_ln_gamma2[i] + m_ln_beta2[i];

      // SiLU: a2 = ln2 * sigmoid(ln2)
      m_ln2.Activation(m_sig2, AF_SIGMOID);
      m_a2 = m_ln2 * m_sig2;

      // Dropout 2
      m_d2 = m_a2;
      if(m_training_mode && m_dropout > 0)
      {
         double scale = 1.0 / (1.0 - m_dropout);
         m_drop_mask2.Resize(h2);
         for(int i = 0; i < h2; i++)
         {
            if(ML_RandDouble() < m_dropout)
            { m_d2[i] = 0.0; m_drop_mask2[i] = 0.0; }
            else
            { m_d2[i] *= scale; m_drop_mask2[i] = scale; }
         }
      }

      // TabM: perturb output layer input with r3[ki]
      vector d2_in = m_d2;
      if(use_tabm)
      {
         m_d2_orig = m_d2;  // cache before r3 perturbation
         for(int j = 0; j < m_h2; j++)
            d2_in[j] *= m_r3[ki][j];
      }

      // Output: scalar = d2_in * W3 + b3
      vector out = d2_in.MatMul(m_w3);
      double raw_out = out[0];
      // TabM: perturb output with s3[ki]
      if(use_tabm)
      {
         m_out_raw = raw_out;  // cache before s3 scaling
         raw_out *= m_s3[ki][0];
      }
      return raw_out + m_b3[0];
   }

   // Backward pass: compute gradients for a single sample
   // When TabM is active (m_cur_member >= 0, m_ensemble_k > 1), also computes
   // gradients for r/s perturbation vectors (accumulated into gr1..gs3).
   void Backward(double d_output,
                 matrix &gw1, vector &gb1, matrix &gw2, vector &gb2,
                 matrix &gw3, vector &gb3,
                 vector &gg1, vector &gbeta1, vector &gg2, vector &gbeta2,
                 vector &gr1, vector &gs1, vector &gr2, vector &gs2,
                 vector &gr3, vector &gs3)
   {
      int h1 = m_h1, h2 = m_h2;
      int ki = m_cur_member;
      bool use_tabm = (ki >= 0 && m_ensemble_k > 1);

      // === Output layer ===
      // TabM: d_output flows through s3 scaling first
      // output = s3 * (d2_in * W3) + b3,  where d2_in = r3 ⊙ d2
      // d_s3 = d_output * out_raw (out before s3)
      // d_raw = d_output * s3
      double d_raw = d_output;
      if(use_tabm)
      {
         gs3[0] += d_output * m_out_raw;
         d_raw = d_output * m_s3[ki][0];
      }

      // dW3: actual input to W3 was d2_in (= r3⊙d2 if TabM, else d2)
      vector d_out_v;
      d_out_v.Resize(1);
      d_out_v[0] = d_raw;
      if(use_tabm)
      {
         // d2_in = r3 ⊙ d2_orig
         vector d2_in;
         d2_in.Resize(h2);
         for(int i = 0; i < h2; i++) d2_in[i] = m_d2_orig[i] * m_r3[ki][i];
         gw3 += d2_in.Outer(d_out_v);
      }
      else
         gw3 += m_d2.Outer(d_out_v);
      // Bias gradient: b3 is added AFTER s3 scaling, so d_b3 = d_output (not d_raw)
      gb3[0] += d_output;

      // d_d2_in = d_raw * W3^T → vector of size h2
      vector d_d2_in;
      d_d2_in.Resize(h2);
      for(int i = 0; i < h2; i++) d_d2_in[i] = d_raw * m_w3[i][0];

      // TabM: backprop through r3 perturbation
      // d2_in = r3 ⊙ d2  =>  d_r3 = d_d2_in ⊙ d2_orig,  d_d2 = d_d2_in ⊙ r3
      vector d_d2;
      if(use_tabm)
      {
         for(int i = 0; i < h2; i++) gr3[i] += d_d2_in[i] * m_d2_orig[i];
         d_d2.Resize(h2);
         for(int i = 0; i < h2; i++) d_d2[i] = d_d2_in[i] * m_r3[ki][i];
      }
      else
         d_d2 = d_d2_in;

      // Undo dropout 2
      if(m_training_mode && m_dropout > 0)
         d_d2 = d_d2 * m_drop_mask2;

      // SiLU backward: d_ln2 = d_a2 * (sig2 + ln2 * sig2 * (1 - sig2))
      vector silu_grad2;
      silu_grad2.Resize(h2);
      for(int i = 0; i < h2; i++)
         silu_grad2[i] = m_sig2[i] * (1.0 + m_ln2[i] * (1.0 - m_sig2[i]));
      vector d_ln2 = d_d2 * silu_grad2;

      // LayerNorm 2 backward
      gg2 += d_ln2 * m_xhat2;
      gbeta2 += d_ln2;
      vector d_xhat2 = d_ln2 * m_ln_gamma2;
      double sum_dx2 = 0, sum_dx_xhat2 = 0;
      for(int i = 0; i < h2; i++)
      {
         sum_dx2 += d_xhat2[i];
         sum_dx_xhat2 += d_xhat2[i] * m_xhat2[i];
      }
      vector d_z2;
      d_z2.Resize(h2);
      for(int i = 0; i < h2; i++)
         d_z2[i] = (m_std_inv2 / h2) * (h2 * d_xhat2[i] - sum_dx2 -
                                          m_xhat2[i] * sum_dx_xhat2);

      // TabM: backprop through s2 scaling
      // z2 = s2 ⊙ z2_raw + b2  =>  d_s2 = d_z2 ⊙ z2_raw,  d_z2_raw = d_z2 ⊙ s2
      vector d_z2_raw;
      if(use_tabm)
      {
         for(int i = 0; i < h2; i++) gs2[i] += d_z2[i] * m_z2_raw[i];
         d_z2_raw.Resize(h2);
         for(int i = 0; i < h2; i++) d_z2_raw[i] = d_z2[i] * m_s2[ki][i];
      }
      else
         d_z2_raw = d_z2;

      // Layer 2 weight gradients: actual input was d1_in (= r2⊙d1 if TabM)
      if(use_tabm)
      {
         vector d1_in;
         d1_in.Resize(h1);
         for(int i = 0; i < h1; i++) d1_in[i] = m_d1_orig[i] * m_r2[ki][i];
         gw2 += d1_in.Outer(d_z2_raw);
      }
      else
         gw2 += m_d1.Outer(d_z2_raw);
      // Bias gradient: b2 is added AFTER s2 scaling, so d_b2 = d_z2 (not d_z2_raw)
      gb2 += d_z2;

      // d_d1_in = d_z2_raw * W2^T
      vector d_d1_in;
      d_d1_in.Resize(h1);
      d_d1_in.Fill(0.0);
      for(int i = 0; i < h1; i++)
         for(int j = 0; j < h2; j++)
            d_d1_in[i] += d_z2_raw[j] * m_w2[i][j];

      // TabM: backprop through r2 perturbation
      vector d_d1;
      if(use_tabm)
      {
         for(int i = 0; i < h1; i++) gr2[i] += d_d1_in[i] * m_d1_orig[i];
         d_d1.Resize(h1);
         for(int i = 0; i < h1; i++) d_d1[i] = d_d1_in[i] * m_r2[ki][i];
      }
      else
         d_d1 = d_d1_in;

      // Undo dropout 1
      if(m_training_mode && m_dropout > 0)
         d_d1 = d_d1 * m_drop_mask1;

      // SiLU backward layer 1
      vector silu_grad1;
      silu_grad1.Resize(h1);
      for(int i = 0; i < h1; i++)
         silu_grad1[i] = m_sig1[i] * (1.0 + m_ln1[i] * (1.0 - m_sig1[i]));
      vector d_ln1 = d_d1 * silu_grad1;

      // LayerNorm 1 backward
      gg1 += d_ln1 * m_xhat1;
      gbeta1 += d_ln1;
      vector d_xhat1 = d_ln1 * m_ln_gamma1;
      double sum_dx1 = 0, sum_dx_xhat1 = 0;
      for(int i = 0; i < h1; i++)
      {
         sum_dx1 += d_xhat1[i];
         sum_dx_xhat1 += d_xhat1[i] * m_xhat1[i];
      }
      vector d_z1;
      d_z1.Resize(h1);
      for(int i = 0; i < h1; i++)
         d_z1[i] = (m_std_inv1 / h1) * (h1 * d_xhat1[i] - sum_dx1 -
                                          m_xhat1[i] * sum_dx_xhat1);

      // TabM: backprop through s1 scaling
      // z1 = s1 ⊙ z1_raw + b1  =>  d_s1 = d_z1 ⊙ z1_raw,  d_z1_raw = d_z1 ⊙ s1
      vector d_z1_raw;
      if(use_tabm)
      {
         for(int i = 0; i < h1; i++) gs1[i] += d_z1[i] * m_z1_raw[i];
         d_z1_raw.Resize(h1);
         for(int i = 0; i < h1; i++) d_z1_raw[i] = d_z1[i] * m_s1[ki][i];
      }
      else
         d_z1_raw = d_z1;

      // Layer 1 weight gradients: input was (r1⊙ple if TabM, else ple)
      // m_ple_out is already perturbed in Forward, so use it directly
      gw1 += m_ple_out.Outer(d_z1_raw);
      // Bias gradient: b1 is added AFTER s1 scaling, so d_b1 = d_z1 (not d_z1_raw)
      gb1 += d_z1;

      // TabM: compute d_r1 gradient
      if(use_tabm)
      {
         // d_ple_perturbed = d_z1_raw * W1^T
         vector d_ple_perturbed;
         d_ple_perturbed.Resize(m_ple_input_dim);
         d_ple_perturbed.Fill(0.0);
         for(int i = 0; i < m_ple_input_dim; i++)
            for(int j = 0; j < h1; j++)
               d_ple_perturbed[i] += d_z1_raw[j] * m_w1[i][j];
         // d_r1[j] = d_ple_perturbed[j] * ple_orig[j]
         for(int i = 0; i < m_ple_input_dim; i++)
            gr1[i] += d_ple_perturbed[i] * m_ple_orig[i];
      }
   }

   void AdamUpdate(matrix &w, matrix &m, matrix &v, matrix &grad, double lr, int step)
   {
      double bc1 = 1.0 - MathPow(ML_ADAM_BETA1, step);
      double bc2 = 1.0 - MathPow(ML_ADAM_BETA2, step);
      int rows = (int)w.Rows(), cols = (int)w.Cols();
      for(int r = 0; r < rows; r++)
      {
         for(int c = 0; c < cols; c++)
         {
            m[r][c] = ML_ADAM_BETA1 * m[r][c] + (1.0 - ML_ADAM_BETA1) * grad[r][c];
            v[r][c] = ML_ADAM_BETA2 * v[r][c] + (1.0 - ML_ADAM_BETA2) * grad[r][c] * grad[r][c];
            double m_hat = m[r][c] / bc1;
            double v_hat = v[r][c] / bc2;
            w[r][c] -= lr * m_hat / (MathSqrt(v_hat) + ML_EPS);
         }
      }
   }

   // Adam update for a single row of a matrix (used for per-member TabM r/s vectors)
   void AdamUpdateRow(matrix &w, int row, matrix &m_mat, matrix &v_mat,
                      vector &grad, double lr, int step)
   {
      double bc1 = 1.0 - MathPow(ML_ADAM_BETA1, step);
      double bc2 = 1.0 - MathPow(ML_ADAM_BETA2, step);
      int sz = (int)grad.Size();
      for(int i = 0; i < sz; i++)
      {
         m_mat[row][i] = ML_ADAM_BETA1 * m_mat[row][i] + (1.0 - ML_ADAM_BETA1) * grad[i];
         v_mat[row][i] = ML_ADAM_BETA2 * v_mat[row][i] + (1.0 - ML_ADAM_BETA2) * grad[i] * grad[i];
         double mh = m_mat[row][i] / bc1;
         double vh = v_mat[row][i] / bc2;
         w[row][i] -= lr * mh / (MathSqrt(vh) + ML_EPS);
      }
   }

   void AdamUpdateVec(vector &w, vector &m_vec, vector &v_vec, vector &grad,
                      double lr, int step)
   {
      double bc1 = 1.0 - MathPow(ML_ADAM_BETA1, step);
      double bc2 = 1.0 - MathPow(ML_ADAM_BETA2, step);
      int sz = (int)w.Size();
      for(int i = 0; i < sz; i++)
      {
         m_vec[i] = ML_ADAM_BETA1 * m_vec[i] + (1.0 - ML_ADAM_BETA1) * grad[i];
         v_vec[i] = ML_ADAM_BETA2 * v_vec[i] + (1.0 - ML_ADAM_BETA2) * grad[i] * grad[i];
         double mh = m_vec[i] / bc1;
         double vh = v_vec[i] / bc2;
         w[i] -= lr * mh / (MathSqrt(vh) + ML_EPS);
      }
   }

   // LR scheduling: 0=None, 1=Cosine, 2=WarmupCosine
   double GetScheduledLR()
   {
      if(ML_DT_LRSchedule == 0) return m_lr;

      double lr_max = m_lr;
      double lr_min = lr_max * 0.01;  // 1% of max
      int T = ML_MaxTrees * ML_TreesPerRound;  // total budget estimate
      if(T < 1) T = 100;
      double t = (double)m_total_steps;

      if(ML_DT_LRSchedule == 1)  // Cosine annealing
      {
         return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + MathCos(M_PI * t / T));
      }
      else  // WarmupCosine
      {
         double warmup_steps = T * 0.1;
         if(t < warmup_steps)
            return lr_max * (t / MathMax(warmup_steps, 1.0));  // linear warmup
         double decay_t = t - warmup_steps;
         double decay_T = T - warmup_steps;
         if(decay_T < 1.0) decay_T = 1.0;
         return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + MathCos(M_PI * decay_t / decay_T));
      }
   }

   void InitWeights()
   {
      // Xavier initialization
      double lim1 = MathSqrt(6.0 / (m_ple_input_dim + m_h1));
      m_w1.Resize(m_ple_input_dim, m_h1);
      for(int r = 0; r < m_ple_input_dim; r++)
         for(int c = 0; c < m_h1; c++)
            m_w1[r][c] = (ML_RandDouble() * 2.0 - 1.0) * lim1;

      double lim2 = MathSqrt(6.0 / (m_h1 + m_h2));
      m_w2.Resize(m_h1, m_h2);
      for(int r = 0; r < m_h1; r++)
         for(int c = 0; c < m_h2; c++)
            m_w2[r][c] = (ML_RandDouble() * 2.0 - 1.0) * lim2;

      double lim3 = MathSqrt(6.0 / (m_h2 + 1));
      m_w3.Resize(m_h2, 1);
      for(int r = 0; r < m_h2; r++)
         m_w3[r][0] = (ML_RandDouble() * 2.0 - 1.0) * lim3;

      m_b1.Resize(m_h1); m_b1.Fill(0.0);
      m_b2.Resize(m_h2); m_b2.Fill(0.0);
      m_b3.Resize(1);    m_b3.Fill(0.0);

      m_ln_gamma1.Resize(m_h1); m_ln_gamma1.Fill(1.0);
      m_ln_beta1.Resize(m_h1);  m_ln_beta1.Fill(0.0);
      m_ln_gamma2.Resize(m_h2); m_ln_gamma2.Fill(1.0);
      m_ln_beta2.Resize(m_h2);  m_ln_beta2.Fill(0.0);

      // Adam moment buffers
      m_mw1.Resize(m_ple_input_dim, m_h1); m_mw1.Fill(0.0);
      m_vw1.Resize(m_ple_input_dim, m_h1); m_vw1.Fill(0.0);
      m_mw2.Resize(m_h1, m_h2); m_mw2.Fill(0.0);
      m_vw2.Resize(m_h1, m_h2); m_vw2.Fill(0.0);
      m_mw3.Resize(m_h2, 1); m_mw3.Fill(0.0);
      m_vw3.Resize(m_h2, 1); m_vw3.Fill(0.0);

      m_mb1.Resize(m_h1); m_mb1.Fill(0.0);
      m_vb1.Resize(m_h1); m_vb1.Fill(0.0);
      m_mb2.Resize(m_h2); m_mb2.Fill(0.0);
      m_vb2.Resize(m_h2); m_vb2.Fill(0.0);
      m_mb3.Resize(1); m_mb3.Fill(0.0);
      m_vb3.Resize(1); m_vb3.Fill(0.0);

      // LN Adam
      m_mg1.Resize(m_h1);    m_mg1.Fill(0.0);
      m_vg1.Resize(m_h1);    m_vg1.Fill(0.0);
      m_mb1_ln.Resize(m_h1); m_mb1_ln.Fill(0.0);
      m_vb1_ln.Resize(m_h1); m_vb1_ln.Fill(0.0);
      m_mg2.Resize(m_h2);    m_mg2.Fill(0.0);
      m_vg2.Resize(m_h2);    m_vg2.Fill(0.0);
      m_mb2_ln.Resize(m_h2); m_mb2_ln.Fill(0.0);
      m_vb2_ln.Resize(m_h2); m_vb2_ln.Fill(0.0);

      m_adam_step = 0;
      m_cur_member = -1;  // no member selected

      // BatchEnsemble r/s vectors
      InitBatchEnsemble();
   }

   void InitBatchEnsemble()
   {
      m_ensemble_k = MathMax(1, ML_DT_EnsembleK);
      if(m_ensemble_k <= 1)
      {
         m_ensemble_k = 1;
         return;  // no allocation needed — Forward() uses identity path
      }

      int k = m_ensemble_k;
      // r vectors perturb layer inputs, s vectors perturb layer outputs
      // Initialize r ~ N(1.0, 0.1), s ~ N(1.0, 0.1)
      m_r1.Resize(k, m_ple_input_dim); m_s1.Resize(k, m_h1);
      m_r2.Resize(k, m_h1);            m_s2.Resize(k, m_h2);
      m_r3.Resize(k, m_h2);            m_s3.Resize(k, 1);
      for(int ki = 0; ki < k; ki++)
      {
         for(int j = 0; j < m_ple_input_dim; j++) m_r1[ki][j] = ML_RandNormal(1.0, 0.1);
         for(int j = 0; j < m_h1; j++)            m_s1[ki][j] = ML_RandNormal(1.0, 0.1);
         for(int j = 0; j < m_h1; j++)            m_r2[ki][j] = ML_RandNormal(1.0, 0.1);
         for(int j = 0; j < m_h2; j++)            m_s2[ki][j] = ML_RandNormal(1.0, 0.1);
         for(int j = 0; j < m_h2; j++)            m_r3[ki][j] = ML_RandNormal(1.0, 0.1);
         m_s3[ki][0] = ML_RandNormal(1.0, 0.1);
      }

      // Adam moments for r/s — all zero
      m_r1_m.Resize(k, m_ple_input_dim); m_r1_m.Fill(0.0);
      m_r1_v.Resize(k, m_ple_input_dim); m_r1_v.Fill(0.0);
      m_s1_m.Resize(k, m_h1);            m_s1_m.Fill(0.0);
      m_s1_v.Resize(k, m_h1);            m_s1_v.Fill(0.0);
      m_r2_m.Resize(k, m_h1);            m_r2_m.Fill(0.0);
      m_r2_v.Resize(k, m_h1);            m_r2_v.Fill(0.0);
      m_s2_m.Resize(k, m_h2);            m_s2_m.Fill(0.0);
      m_s2_v.Resize(k, m_h2);            m_s2_v.Fill(0.0);
      m_r3_m.Resize(k, m_h2);            m_r3_m.Fill(0.0);
      m_r3_v.Resize(k, m_h2);            m_r3_v.Fill(0.0);
      m_s3_m.Resize(k, 1);               m_s3_m.Fill(0.0);
      m_s3_v.Resize(k, 1);               m_s3_v.Fill(0.0);
   }

public:
   CDeepTabular_Model() : m_ple_bins(0), m_ple_input_dim(0), m_h1(0), m_h2(0),
                           m_lr(0.001), m_dropout(0.2), m_batch_size(64),
                           m_epochs(5), m_adam_step(0), m_total_steps(0),
                           m_ple_computed(false),
                           m_training_mode(false), m_std_inv1(0), m_std_inv2(0),
                           m_ensemble_k(1), m_cur_member(-1) {}
   ~CDeepTabular_Model() {}

   void Init(ENUM_ML_TASK task, int n_features)
   {
      InitBase(task, n_features);
      m_ple_bins = ML_PLE_Bins;
      m_ple_input_dim = n_features * m_ple_bins;
      m_h1 = ML_DT_Hidden1;
      m_h2 = ML_DT_Hidden2;
      m_lr = ML_DT_LearningRate;
      m_dropout = ML_DT_Dropout;
      m_batch_size = ML_DT_BatchSize;
      m_epochs = ML_DT_Epochs;
      m_ple_computed = false;
      m_total_steps = 0;

      InitWeights();
      ML_SeedRng((ulong)GetTickCount() + 3);
   }

   virtual double Predict(const double &features[], int count)
   {
      m_training_mode = false;
      if(m_ensemble_k <= 1)
      {
         m_cur_member = -1;
         return Forward(features, count);
      }
      // TabM: average K member predictions
      double sum = 0.0;
      for(int ki = 0; ki < m_ensemble_k; ki++)
      {
         m_cur_member = ki;
         sum += Forward(features, count);
      }
      m_cur_member = -1;
      return sum / m_ensemble_k;
   }

   virtual void Train(bool force = false)
   {
      int n = m_buffer.Count();
      if(!force && n < ML_ColdStartMin) return;

      // 4A: On first training, compute base score and init output bias
      if(m_train_count == 0)
      {
         ComputeBaseScore();
         m_b3[0] = m_base_score;
      }

      // Compute PLE boundaries on first training
      if(!m_ple_computed)
         ComputePLEBoundaries();

      m_training_mode = true;

      int indices[];
      ArrayResize(indices, n);

      double features[];
      ArrayResize(features, m_n_features);

      for(int epoch = 0; epoch < m_epochs; epoch++)
      {
         // Shuffle indices
         for(int i = 0; i < n; i++) indices[i] = i;
         ML_Shuffle(indices, n);

         // Process mini-batches
         for(int batch_start = 0; batch_start < n; batch_start += m_batch_size)
         {
            int batch_end = MathMin(batch_start + m_batch_size, n);
            int bs = batch_end - batch_start;

            // Accumulate gradients for shared parameters
            matrix gw1, gw2, gw3;
            vector gb1, gb2, gb3, gg1, gbeta1, gg2, gbeta2;
            gw1.Resize(m_ple_input_dim, m_h1); gw1.Fill(0.0);
            gw2.Resize(m_h1, m_h2);            gw2.Fill(0.0);
            gw3.Resize(m_h2, 1);               gw3.Fill(0.0);
            gb1.Resize(m_h1); gb1.Fill(0.0);
            gb2.Resize(m_h2); gb2.Fill(0.0);
            gb3.Resize(1);    gb3.Fill(0.0);
            gg1.Resize(m_h1); gg1.Fill(0.0);
            gbeta1.Resize(m_h1); gbeta1.Fill(0.0);
            gg2.Resize(m_h2); gg2.Fill(0.0);
            gbeta2.Resize(m_h2); gbeta2.Fill(0.0);

            // TabM: per-member r/s gradient accumulators
            bool use_tabm = (m_ensemble_k > 1);
            // gr/gs vectors: one per member, allocated only when TabM active
            // Use flat arrays of vectors: gr1_all[k], etc.
            vector gr1_all[], gs1_all[], gr2_all[], gs2_all[], gr3_all[], gs3_all[];
            int member_counts[];  // samples assigned to each member
            if(use_tabm)
            {
               ArrayResize(gr1_all, m_ensemble_k);
               ArrayResize(gs1_all, m_ensemble_k);
               ArrayResize(gr2_all, m_ensemble_k);
               ArrayResize(gs2_all, m_ensemble_k);
               ArrayResize(gr3_all, m_ensemble_k);
               ArrayResize(gs3_all, m_ensemble_k);
               ArrayResize(member_counts, m_ensemble_k);
               for(int ki = 0; ki < m_ensemble_k; ki++)
               {
                  gr1_all[ki].Resize(m_ple_input_dim); gr1_all[ki].Fill(0.0);
                  gs1_all[ki].Resize(m_h1);            gs1_all[ki].Fill(0.0);
                  gr2_all[ki].Resize(m_h1);            gr2_all[ki].Fill(0.0);
                  gs2_all[ki].Resize(m_h2);            gs2_all[ki].Fill(0.0);
                  gr3_all[ki].Resize(m_h2);            gr3_all[ki].Fill(0.0);
                  gs3_all[ki].Resize(1);               gs3_all[ki].Fill(0.0);
                  member_counts[ki] = 0;
               }
            }

            // Dummy zero vectors for non-TabM backward calls
            vector gr1_dummy, gs1_dummy, gr2_dummy, gs2_dummy, gr3_dummy, gs3_dummy;
            if(!use_tabm)
            {
               gr1_dummy.Resize(1); gr1_dummy.Fill(0.0);
               gs1_dummy.Resize(1); gs1_dummy.Fill(0.0);
               gr2_dummy.Resize(1); gr2_dummy.Fill(0.0);
               gs2_dummy.Resize(1); gs2_dummy.Fill(0.0);
               gr3_dummy.Resize(1); gr3_dummy.Fill(0.0);
               gs3_dummy.Resize(1); gs3_dummy.Fill(0.0);
            }

            for(int b = batch_start; b < batch_end; b++)
            {
               int si = indices[b];
               m_buffer.GetFeatures(si, features);

               // TabM: pick a random member for this sample
               if(use_tabm)
                  m_cur_member = ML_RandInt(m_ensemble_k);
               else
                  m_cur_member = -1;

               double pred = Forward(features, m_n_features);
               double label = m_buffer.GetLabel(si);
               double weight = m_buffer.GetWeight(si);

               double d_output;
               if(m_task == ML_TASK_REGRESSION)
                  d_output = ML_HuberGrad(pred, label) * weight;
               else
                  d_output = ML_LoglossGrad(pred, label) * weight;

               // Gradient clipping at output
               d_output = MathMax(-10.0, MathMin(10.0, d_output));

               if(use_tabm)
               {
                  int ki = m_cur_member;
                  Backward(d_output, gw1, gb1, gw2, gb2, gw3, gb3,
                           gg1, gbeta1, gg2, gbeta2,
                           gr1_all[ki], gs1_all[ki], gr2_all[ki], gs2_all[ki],
                           gr3_all[ki], gs3_all[ki]);
                  member_counts[ki]++;
               }
               else
               {
                  Backward(d_output, gw1, gb1, gw2, gb2, gw3, gb3,
                           gg1, gbeta1, gg2, gbeta2,
                           gr1_dummy, gs1_dummy, gr2_dummy, gs2_dummy,
                           gr3_dummy, gs3_dummy);
               }
            }

            // Average shared gradients
            double inv_bs = 1.0 / bs;
            gw1 *= inv_bs; gw2 *= inv_bs; gw3 *= inv_bs;
            gb1 *= inv_bs; gb2 *= inv_bs; gb3 *= inv_bs;
            gg1 *= inv_bs; gbeta1 *= inv_bs;
            gg2 *= inv_bs; gbeta2 *= inv_bs;

            // 4D: Global gradient norm clipping (shared params only)
            if(ML_DT_GradClipNorm > 0)
            {
               double norm_sq = 0.0;
               for(int r = 0; r < (int)gw1.Rows(); r++)
                  for(int c = 0; c < (int)gw1.Cols(); c++)
                     norm_sq += gw1[r][c] * gw1[r][c];
               for(int r = 0; r < (int)gw2.Rows(); r++)
                  for(int c = 0; c < (int)gw2.Cols(); c++)
                     norm_sq += gw2[r][c] * gw2[r][c];
               for(int r = 0; r < (int)gw3.Rows(); r++)
                  for(int c = 0; c < (int)gw3.Cols(); c++)
                     norm_sq += gw3[r][c] * gw3[r][c];
               for(int i = 0; i < (int)gb1.Size(); i++) norm_sq += gb1[i] * gb1[i];
               for(int i = 0; i < (int)gb2.Size(); i++) norm_sq += gb2[i] * gb2[i];
               for(int i = 0; i < (int)gb3.Size(); i++) norm_sq += gb3[i] * gb3[i];
               for(int i = 0; i < (int)gg1.Size(); i++) norm_sq += gg1[i] * gg1[i];
               for(int i = 0; i < (int)gbeta1.Size(); i++) norm_sq += gbeta1[i] * gbeta1[i];
               for(int i = 0; i < (int)gg2.Size(); i++) norm_sq += gg2[i] * gg2[i];
               for(int i = 0; i < (int)gbeta2.Size(); i++) norm_sq += gbeta2[i] * gbeta2[i];

               double norm = MathSqrt(norm_sq);
               if(norm > ML_DT_GradClipNorm)
               {
                  double scale = ML_DT_GradClipNorm / (norm + 1e-15);
                  gw1 *= scale; gw2 *= scale; gw3 *= scale;
                  gb1 *= scale; gb2 *= scale; gb3 *= scale;
                  gg1 *= scale; gbeta1 *= scale;
                  gg2 *= scale; gbeta2 *= scale;
               }
            }

            // 4C: Get scheduled learning rate
            double cur_lr = GetScheduledLR();

            // Adam update for shared parameters
            m_adam_step++;
            AdamUpdate(m_w1, m_mw1, m_vw1, gw1, cur_lr, m_adam_step);
            AdamUpdate(m_w2, m_mw2, m_vw2, gw2, cur_lr, m_adam_step);
            AdamUpdate(m_w3, m_mw3, m_vw3, gw3, cur_lr, m_adam_step);
            AdamUpdateVec(m_b1, m_mb1, m_vb1, gb1, cur_lr, m_adam_step);
            AdamUpdateVec(m_b2, m_mb2, m_vb2, gb2, cur_lr, m_adam_step);
            AdamUpdateVec(m_b3, m_mb3, m_vb3, gb3, cur_lr, m_adam_step);
            AdamUpdateVec(m_ln_gamma1, m_mg1, m_vg1, gg1, cur_lr, m_adam_step);
            AdamUpdateVec(m_ln_beta1, m_mb1_ln, m_vb1_ln, gbeta1, cur_lr, m_adam_step);
            AdamUpdateVec(m_ln_gamma2, m_mg2, m_vg2, gg2, cur_lr, m_adam_step);
            AdamUpdateVec(m_ln_beta2, m_mb2_ln, m_vb2_ln, gbeta2, cur_lr, m_adam_step);

            // TabM: Adam update for per-member r/s vectors
            if(use_tabm)
            {
               for(int ki = 0; ki < m_ensemble_k; ki++)
               {
                  if(member_counts[ki] == 0) continue;
                  // Average per-member gradients by number of samples assigned
                  double inv_mc = 1.0 / member_counts[ki];
                  gr1_all[ki] *= inv_mc; gs1_all[ki] *= inv_mc;
                  gr2_all[ki] *= inv_mc; gs2_all[ki] *= inv_mc;
                  gr3_all[ki] *= inv_mc; gs3_all[ki] *= inv_mc;

                  AdamUpdateRow(m_r1, ki, m_r1_m, m_r1_v, gr1_all[ki], cur_lr, m_adam_step);
                  AdamUpdateRow(m_s1, ki, m_s1_m, m_s1_v, gs1_all[ki], cur_lr, m_adam_step);
                  AdamUpdateRow(m_r2, ki, m_r2_m, m_r2_v, gr2_all[ki], cur_lr, m_adam_step);
                  AdamUpdateRow(m_s2, ki, m_s2_m, m_s2_v, gs2_all[ki], cur_lr, m_adam_step);
                  AdamUpdateRow(m_r3, ki, m_r3_m, m_r3_v, gr3_all[ki], cur_lr, m_adam_step);
                  AdamUpdateRow(m_s3, ki, m_s3_m, m_s3_v, gs3_all[ki], cur_lr, m_adam_step);
               }
            }

            // 4B: AdamW decoupled weight decay (weight matrices only, not biases/LN)
            if(ML_DT_WeightDecay > 0)
            {
               double wd = cur_lr * ML_DT_WeightDecay;
               for(int r = 0; r < (int)m_w1.Rows(); r++)
                  for(int c = 0; c < (int)m_w1.Cols(); c++)
                     m_w1[r][c] -= wd * m_w1[r][c];
               for(int r = 0; r < (int)m_w2.Rows(); r++)
                  for(int c = 0; c < (int)m_w2.Cols(); c++)
                     m_w2[r][c] -= wd * m_w2[r][c];
               for(int r = 0; r < (int)m_w3.Rows(); r++)
                  for(int c = 0; c < (int)m_w3.Cols(); c++)
                     m_w3[r][c] -= wd * m_w3[r][c];
            }

            // 4C: Increment total steps for LR scheduling
            m_total_steps++;
         }
      }

      m_training_mode = false;
      m_cur_member = -1;
      m_train_count++;

      // Compute feature importance from W1 column magnitudes
      for(int f = 0; f < m_n_features; f++)
      {
         double imp = 0.0;
         int base = f * m_ple_bins;
         for(int b = 0; b < m_ple_bins; b++)
            for(int j = 0; j < m_h1; j++)
               imp += MathAbs(m_w1[base + b][j]);
         m_feature_importance[f] = imp;
      }
   }

   virtual bool SaveToFile(string filename)
   {
      int handle = FileOpen(filename, FILE_WRITE | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE) return false;

      FileWriteInteger(handle, ML_MAGIC_SAVE);
      FileWriteInteger(handle, (int)ML_DEEP_TABULAR);
      FileWriteInteger(handle, (int)m_task);
      FileWriteInteger(handle, m_n_features);
      FileWriteInteger(handle, m_ple_bins);
      FileWriteInteger(handle, m_h1);
      FileWriteInteger(handle, m_h2);
      FileWriteDouble(handle, m_lr);
      FileWriteInteger(handle, m_adam_step);
      FileWriteInteger(handle, m_train_count);

      // Save weights
      ML_SaveMatrix(handle, m_w1);
      ML_SaveVector(handle, m_b1);
      ML_SaveMatrix(handle, m_w2);
      ML_SaveVector(handle, m_b2);
      ML_SaveMatrix(handle, m_w3);
      ML_SaveVector(handle, m_b3);

      // LayerNorm
      ML_SaveVector(handle, m_ln_gamma1);
      ML_SaveVector(handle, m_ln_beta1);
      ML_SaveVector(handle, m_ln_gamma2);
      ML_SaveVector(handle, m_ln_beta2);

      // Adam state
      ML_SaveMatrix(handle, m_mw1); ML_SaveMatrix(handle, m_vw1);
      ML_SaveMatrix(handle, m_mw2); ML_SaveMatrix(handle, m_vw2);
      ML_SaveMatrix(handle, m_mw3); ML_SaveMatrix(handle, m_vw3);
      ML_SaveVector(handle, m_mb1); ML_SaveVector(handle, m_vb1);
      ML_SaveVector(handle, m_mb2); ML_SaveVector(handle, m_vb2);
      ML_SaveVector(handle, m_mb3); ML_SaveVector(handle, m_vb3);
      ML_SaveVector(handle, m_mg1); ML_SaveVector(handle, m_vg1);
      ML_SaveVector(handle, m_mb1_ln); ML_SaveVector(handle, m_vb1_ln);
      ML_SaveVector(handle, m_mg2); ML_SaveVector(handle, m_vg2);
      ML_SaveVector(handle, m_mb2_ln); ML_SaveVector(handle, m_vb2_ln);

      // PLE boundaries
      FileWriteInteger(handle, m_ple_computed ? 1 : 0);
      if(m_ple_computed)
         FileWriteArray(handle, m_ple_bounds, 0, m_n_features * (m_ple_bins + 1));

      // Feature importance
      FileWriteArray(handle, m_feature_importance, 0, m_n_features);

      // LR scheduling state (added after original format)
      FileWriteInteger(handle, 0x4C525343);  // "LRSC" magic marker
      FileWriteInteger(handle, m_total_steps);

      // TabM BatchEnsemble r/s vectors
      FileWriteInteger(handle, ML_MAGIC_DT_V2);  // "DTV2"
      FileWriteInteger(handle, m_ensemble_k);
      if(m_ensemble_k > 1)
      {
         ML_SaveMatrix(handle, m_r1); ML_SaveMatrix(handle, m_s1);
         ML_SaveMatrix(handle, m_r2); ML_SaveMatrix(handle, m_s2);
         ML_SaveMatrix(handle, m_r3); ML_SaveMatrix(handle, m_s3);
         // Adam moments for r/s
         ML_SaveMatrix(handle, m_r1_m); ML_SaveMatrix(handle, m_r1_v);
         ML_SaveMatrix(handle, m_s1_m); ML_SaveMatrix(handle, m_s1_v);
         ML_SaveMatrix(handle, m_r2_m); ML_SaveMatrix(handle, m_r2_v);
         ML_SaveMatrix(handle, m_s2_m); ML_SaveMatrix(handle, m_s2_v);
         ML_SaveMatrix(handle, m_r3_m); ML_SaveMatrix(handle, m_r3_v);
         ML_SaveMatrix(handle, m_s3_m); ML_SaveMatrix(handle, m_s3_v);
      }

      FileWriteInteger(handle, ML_MAGIC_END);
      FileClose(handle);
      return true;
   }

   virtual bool LoadFromFile(string filename)
   {
      int handle = FileOpen(filename, FILE_READ | FILE_BIN | FILE_COMMON);
      if(handle == INVALID_HANDLE) return false;

      int magic = FileReadInteger(handle);
      if(magic != ML_MAGIC_SAVE) { FileClose(handle); return false; }
      int model_type = FileReadInteger(handle);
      if(model_type != (int)ML_DEEP_TABULAR) { FileClose(handle); return false; }

      m_task = (ENUM_ML_TASK)FileReadInteger(handle);
      m_n_features = FileReadInteger(handle);
      m_ple_bins = FileReadInteger(handle);
      m_h1 = FileReadInteger(handle);
      m_h2 = FileReadInteger(handle);
      m_lr = FileReadDouble(handle);
      m_adam_step = FileReadInteger(handle);
      m_train_count = FileReadInteger(handle);
      m_ple_input_dim = m_n_features * m_ple_bins;

      ML_LoadMatrix(handle, m_w1);
      ML_LoadVector(handle, m_b1);
      ML_LoadMatrix(handle, m_w2);
      ML_LoadVector(handle, m_b2);
      ML_LoadMatrix(handle, m_w3);
      ML_LoadVector(handle, m_b3);

      // H7 fix: Validate loaded dimensions match expected PLE input dim
      if((int)m_w1.Rows() != m_ple_input_dim || (int)m_w1.Cols() != m_h1)
      {
         Print("ML Engine: DeepTabular dimension mismatch — file has ", m_w1.Rows(), "x", m_w1.Cols(),
               " but expected ", m_ple_input_dim, "x", m_h1, ". Discarding saved model.");
         FileClose(handle);
         Reset();
         return false;
      }

      ML_LoadVector(handle, m_ln_gamma1);
      ML_LoadVector(handle, m_ln_beta1);
      ML_LoadVector(handle, m_ln_gamma2);
      ML_LoadVector(handle, m_ln_beta2);

      ML_LoadMatrix(handle, m_mw1); ML_LoadMatrix(handle, m_vw1);
      ML_LoadMatrix(handle, m_mw2); ML_LoadMatrix(handle, m_vw2);
      ML_LoadMatrix(handle, m_mw3); ML_LoadMatrix(handle, m_vw3);
      ML_LoadVector(handle, m_mb1); ML_LoadVector(handle, m_vb1);
      ML_LoadVector(handle, m_mb2); ML_LoadVector(handle, m_vb2);
      ML_LoadVector(handle, m_mb3); ML_LoadVector(handle, m_vb3);
      ML_LoadVector(handle, m_mg1); ML_LoadVector(handle, m_vg1);
      ML_LoadVector(handle, m_mb1_ln); ML_LoadVector(handle, m_vb1_ln);
      ML_LoadVector(handle, m_mg2); ML_LoadVector(handle, m_vg2);
      ML_LoadVector(handle, m_mb2_ln); ML_LoadVector(handle, m_vb2_ln);

      m_ple_computed = (FileReadInteger(handle) != 0);
      if(m_ple_computed)
      {
         ArrayResize(m_ple_bounds, m_n_features * (m_ple_bins + 1));
         FileReadArray(handle, m_ple_bounds, 0, m_n_features * (m_ple_bins + 1));
      }

      ArrayResize(m_feature_importance, m_n_features);
      FileReadArray(handle, m_feature_importance, 0, m_n_features);

      // Try to read optional LR scheduling state (backward compat: old files have ML_MAGIC_END here)
      int maybe_lrsc = FileReadInteger(handle);
      if(maybe_lrsc == 0x4C525343)  // "LRSC" magic marker
      {
         m_total_steps = FileReadInteger(handle);
         // Continue to read next optional section
         maybe_lrsc = FileReadInteger(handle);
      }
      else
      {
         m_total_steps = 0;
      }

      // Try to read optional TabM BatchEnsemble state
      if(maybe_lrsc == ML_MAGIC_DT_V2)  // "DTV2"
      {
         int saved_k = FileReadInteger(handle);
         if(saved_k > 1 && saved_k == m_ensemble_k)
         {
            // Load r/s vectors — dimensions must match current config
            ML_LoadMatrix(handle, m_r1); ML_LoadMatrix(handle, m_s1);
            ML_LoadMatrix(handle, m_r2); ML_LoadMatrix(handle, m_s2);
            ML_LoadMatrix(handle, m_r3); ML_LoadMatrix(handle, m_s3);
            ML_LoadMatrix(handle, m_r1_m); ML_LoadMatrix(handle, m_r1_v);
            ML_LoadMatrix(handle, m_s1_m); ML_LoadMatrix(handle, m_s1_v);
            ML_LoadMatrix(handle, m_r2_m); ML_LoadMatrix(handle, m_r2_v);
            ML_LoadMatrix(handle, m_s2_m); ML_LoadMatrix(handle, m_s2_v);
            ML_LoadMatrix(handle, m_r3_m); ML_LoadMatrix(handle, m_r3_v);
            ML_LoadMatrix(handle, m_s3_m); ML_LoadMatrix(handle, m_s3_v);
         }
         else if(saved_k > 1)
         {
            // Ensemble size changed — skip saved data, reinit
            // Need to skip 6 r/s matrices + 12 Adam moment matrices = 18 matrices
            matrix dummy;
            for(int skip = 0; skip < 18; skip++)
               ML_LoadMatrix(handle, dummy);
            // Reinitialize with current ensemble size
            InitBatchEnsemble();
            Print("ML Engine: TabM ensemble size changed (", saved_k, " -> ",
                  m_ensemble_k, "), reinitializing r/s vectors.");
         }
         // If saved_k == 1 and current k > 1, InitBatchEnsemble already ran in InitWeights
         maybe_lrsc = FileReadInteger(handle);
      }
      else
      {
         // No DTV2 section — if ensemble is active, InitBatchEnsemble already ran
      }

      FileClose(handle);
      return (maybe_lrsc == ML_MAGIC_END);
   }

   virtual void Reset()
   {
      CMLModelBase::Reset();
      m_ple_computed = false;
      m_adam_step = 0;
      m_total_steps = 0;
      m_cur_member = -1;
      InitWeights();   // also calls InitBatchEnsemble()
   }
};

//===================================================================
// FACTORY FUNCTION
//===================================================================
CMLModelBase* ML_CreateModel(ENUM_ML_MODEL model_type, ENUM_ML_TASK task, int n_features)
{
   CMLModelBase *model = NULL;

   switch(model_type)
   {
      case ML_LIGHTGBM:
      {
         CLightGBM_Model *lgb = new CLightGBM_Model();
         lgb.Init(task, n_features);
         model = lgb;
         break;
      }
      case ML_XGBOOST:
      {
         CXGBoost_Model *xgb = new CXGBoost_Model();
         xgb.Init(task, n_features);
         model = xgb;
         break;
      }
      case ML_CATBOOST:
      {
         CCatBoost_Model *cat = new CCatBoost_Model();
         cat.Init(task, n_features);
         model = cat;
         break;
      }
      case ML_DEEP_TABULAR:
      {
         CDeepTabular_Model *dt = new CDeepTabular_Model();
         dt.Init(task, n_features);
         model = dt;
         break;
      }
   }

   return model;
}

// Helper: get model type name string
string ML_ModelName(ENUM_ML_MODEL model_type)
{
   switch(model_type)
   {
      case ML_LIGHTGBM:    return "LGB";
      case ML_XGBOOST:     return "XGB";
      case ML_CATBOOST:    return "CAT";
      case ML_DEEP_TABULAR: return "DT";
   }
   return "UNK";
}

// Helper: generate save filename
// H1 fix: include config_code so different feature configurations use different files
// H3 fix: caller passes "GLOBAL" as symbol when ShadowQL_SymbolSpecific == false
// M4 fix: removed unused 'prefix' parameter
string ML_GetFilename(string symbol, ENUM_ML_MODEL model_type,
                      ENUM_ML_TASK task, string config_code = "")
{
   string task_str = (task == ML_TASK_REGRESSION) ? "Dir" : "Exit";
   string name = "ML_" + task_str + "_" + ML_ModelName(model_type) + "_" + symbol;
   if(config_code != "") name += "_" + config_code;
   return name + ".bin";
}

#endif // CML_ENGINE_MQH
