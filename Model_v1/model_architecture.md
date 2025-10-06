
---

# DTI–ADR Multi-Task Model — Full Technical Spec

## 0) Notation & Constants

* Let:

  * `D` = number of drugs, `P` = number of proteins, `K=4048` = number of ADRs.
  * `x_d ∈ ℝ^{n_d}` = raw drug embedding (Smiles2Vec / ChemBERTa / EGNN).
  * `x_p ∈ ℝ^{n_p}` = raw protein embedding (ESM / GVP).
  * `t_d ∈ ℝ^{K}` = TF-IDF row for the drug (sparse or dense).
* Choose shared space size `d = 512` (good default).
* DTI labels: `y_dti ∈ {0,1}`; ADR targets: `y_adr = t_d` (TF-IDF regression target).
* Class imbalance: `pos_weight = (#neg / #pos)` computed **on train split** (≈ 1.8–1.9 in your data).
* Temperature for contrastive: `τ ∈ (0.05, 0.2)` (start with 0.07).

---

## 1) Modules

### 1.1 Adapters (shared-space projectors)

**Purpose:** Normalize scale/shape and map heterogeneous embeddings into a common `d`-dimensional space.

**Design (same template for drug, protein, ADR):**

```
Adapter(in_dim, d, hidden_ratio=2, p_drop=0.1):
  Linear(in_dim, hidden_ratio*d)
  GELU
  Dropout(p_drop)
  Linear(hidden_ratio*d, d)
  LayerNorm(d)
```

* Drug adapter: `f_d: ℝ^{n_d} → ℝ^{d}`
* Protein adapter: `f_p: ℝ^{n_p} → ℝ^{d}`
* ADR adapter: `f_a: ℝ^{K} → ℝ^{d}`

**Forward outputs:**

* `u = f_d(x_d) ∈ ℝ^{d}`
* `v = f_p(x_p) ∈ ℝ^{d}`
* `w = f_a(t_d) ∈ ℝ^{d}` (training only; not needed at inference)

**Implementation notes:**

* Accept both dense and sparse ADR TF-IDF:

  * If sparse: convert to dense on-the-fly (careful with memory; batch K=4048 is fine).
  * Optional: add `Dropout` before the first linear for ADR to reduce overfitting to sparse patterns.

---

### 1.2 DTI Head (pair → binding probability)

Two options; both work. Pick one and stick to it for fairness across encoder combos.

**Option A: Cosine + temperature**

```
DTIHeadCosine(d):
  learnable logit-scale s (init: log(10))
  forward(u, v):
    cos = (u · v) / (||u|| * ||v|| + eps)      # shape [B]
    logit = exp(s) * cos                        # scale similarity
    return logit                                # BCEWithLogitsLoss expects logits
```

**Option B: Bilinear**

```
DTIHeadBilinear(d):
  weight W ∈ ℝ^{d×d}, bias b
  forward(u, v):
    logit = sum(u * (W v)) + b                  # shape [B]
    return logit
```

**Loss:** `BCEWithLogitsLoss(pos_weight=pos_weight)` on logits.

---

### 1.3 ADR Head (drug → K TF-IDF values)

We’re doing **regression to TF-IDF** with Huber/MSE. Because TF-IDF ≥ 0, use **Softplus** to keep predictions nonnegative without squashing large values too aggressively.

```
ADRHead(d, K):
  Linear(d, K)                                  # raw scores
  forward(u):
    yhat = Linear(u)                            # shape [B, K]
    yhat = Softplus(yhat)                       # ensure ≥ 0
    return yhat
```

**Loss:**

* **Huber** (`delta=1.0`) recommended:

  ```
  Huber(yhat, y):
    r = yhat - y
    if |r| ≤ δ: 0.5 * r^2
    else: δ*(|r| - 0.5*δ)
  ```
* Weighted variant (optional) to down-weight the sea of zeros:

  * Compute per-ADR weight `w_k = 1 / (ε + prevalence_k)` over train drugs.
  * Loss = mean over batch of mean_k ( w_k * Huber(yhat[:,k], y[:,k]) ).

---

### 1.4 Contrastive Losses

We use **InfoNCE** (softmax cross-entropy over in-batch negatives).

#### (a) Drug–Protein contrast (DP)

Given a batch of `B` *positive* DTI pairs (you can also include negatives as additional “hard negatives”):

* Normalize: `ũ = u / ||u||`, `ṽ = v / ||v||`.
* Similarity matrix `S = ũ ṽ^T / τ` → shape `[B, B]`.
* For each i, positive match is `(drug_i, protein_i)`.

**Loss DP (both directions):**

```
L_dp_u2p = (1/B) * Σ_i CE( logits=S[i,:], target=i )
L_dp_p2u = (1/B) * Σ_i CE( logits=S[:,i], target=i )
L_dp     = 0.5*(L_dp_u2p + L_dp_p2u)
```

#### (b) Drug–ADR contrast (DA)

For a batch of drugs:

* Get `w = f_a(t_d)` per drug (train only).
* Normalize ũ, ŵ.
* `S_da = ũ ŵ^T / τ` (shape `[B,B]`), positives at diagonal.

**Loss DA (both directions same as above):**

```
L_da = 0.5*(CE(S_da[i,:], i) + CE(S_da[:,i], i)) averaged over i
```

---

### 1.5 Pair-Conditioned ADR Scorer (inference-time module)

We precompute **ADR prototypes** in the shared space:

```
ComputePrototypes(train_set):
  For k in 1..K:
    D_k = { drugs d | TFIDF(d,k) > 0 }        # nonzero TF-IDF
    if D_k empty:
       c_k = zeros(d)
    else:
       c_k = mean_{d ∈ D_k} f_a(t_d)          # detach & store
  Return {c_k} as matrix C ∈ ℝ^{K×d}
```

At inference for a (drug, protein) pair:

```
PairADRScorer(d):
  learnable scalars α, β, γ (init α=β=1.0, γ=0.5)
  forward(u, v, C):                   # C shape [K,d]
    # similarities
    s = (u @ C.T)                     # drug–ADR, shape [K]
    m = (v @ C.T)                     # protein–ADR, shape [K]
    r = (u * v).sum(dim=-1, keepdim=True)   # scalar per pair
    score = sigmoid( α*s + β*m + γ*r )      # shape [K]
    return score
```

To return **joint risk**, also compute:

* `p_bind = sigmoid(DTIHead(u, v))`
* `final_score = p_bind * score` (optional; keeps ranking consistent with binding strength)

---

## 2) Batching & Data Pipeline

### 2.1 Entity stores

* `DrugTable`: map `drug_id → {x_d, t_d}`.
* `ProteinTable`: map `prot_id → {x_p}`.
* Store embeddings as float32; TF-IDF as float32 dense (K=4048 is manageable) or CSR sparse.

### 2.2 DTI pairs dataset

* Each row: `(drug_id, prot_id, y_dti)`.
* **Balanced sampler** for training:

  * For each mini-batch of size `B = 256` (example), sample `B/2` positives and `B/2` negatives.
  * Ensure positive pairs exist across many drugs/proteins to enrich contrastive negatives.

### 2.3 ADR targets dataset

* For each `drug_id`, retrieve `t_d` (TF-IDF from **train** fit, with fixed column order across splits).
* In DTI batches, collate the unique `drug_id`s; fetch `t_d` only once per batch and index.

### 2.4 Collate function (PyTorch DataLoader)

* Input: list of pairs.
* Output tensors:

  * `x_d_batch ∈ ℝ^{B×n_d}`
  * `x_p_batch ∈ ℝ^{B×n_p}`
  * `y_dti_batch ∈ {0,1}^{B}`
  * `t_d_batch ∈ ℝ^{U×K}` with `U ≤ B` unique drugs + an index map from pair→drug row.
* Device placement (GPU) & pinned memory for speed.

---

## 3) Forward & Loss Composition

### 3.1 Forward pass (training)

```
# Adapters
u_all = f_d(x_d_batch)                          # [B, d]
v_all = f_p(x_p_batch)                          # [B, d]

# ADR adapter (train only) for unique drugs in batch:
w_unique = f_a(t_d_unique)                      # [U, d]

# Map each pair to its drug-row w via index map if needed
w_for_pairs = w_unique[drug_index_for_each_pair]# [B, d]  (for DA you can do batch on unique drugs too)

# Supervised heads
logit_dti = DTIHead(u_all, v_all)               # [B]
yhat_adr  = ADRHead(u_unique)                    # [U, K]

# Losses
L_dti = BCEWithLogitsLoss(pos_weight)(logit_dti, y_dti_batch)

L_adr = Huber(yhat_adr, t_d_unique)             # weighted per-ADR optional
# or MSE if preferred; Huber is default

# Contrastive (DP on positives; DA on unique drugs)
u_pos, v_pos = select_positives(u_all, v_all, y_dti_batch)
L_dp = InfoNCE_bidirectional(u_pos, v_pos, tau=τ)

L_da = InfoNCE_bidirectional(u_unique_norm, w_unique_norm, tau=τ)

# Total
Loss = λ_dti*L_dti + λ_adr*L_adr + λ_con*(L_dp + L_da)
```

**Default weights:** `λ_dti=1.0`, `λ_adr=0.5`, `λ_con=0.2`.

### 3.2 Forward pass (inference)

```
u = f_d(x_d)                                   # [d]
v = f_p(x_p)                                   # [d]
logit = DTIHead(u, v)
p_bind = sigmoid(logit)

score_k = PairADRScorer(u, v, C)               # [K]
# ranked ADR list:
rank = argsort(score_k, descending=True)
topk = rank[:K_top]                             # e.g., K_top=10

# optional joint risk:
final_k = p_bind * score_k
```

---

## 4) Optimization & Schedules

* **Optimizer:** `AdamW(lr=2e-4, weight_decay=1e-4)`
* **LR schedule:** cosine decay w/ warmup (5% of total steps) or step LR (gamma=0.5 every N epochs).
* **Dropout:** 0.1–0.3 inside adapters.
* **Batch sizes:** start with `B=256` (fit to GPU memory).
* **Epochs:** 30–60; **early stop** on **val DTI PR-AUC** with patience=5–8.
* **Gradient clipping:** `clip_grad_norm_(params, 1.0)`.

---

## 5) Metrics & Calibration

### 5.1 DTI

* **Primary:** PR-AUC (robust under imbalance).
* AUROC, F1 at tuned threshold.
* **Threshold selection:** pick threshold on **val** to maximize F1 (or based on desired precision/recall trade-off).

### 5.2 ADR

* **Regression:** RMSE, MAE (micro).
* **Ranking:** Recall@k, Precision@k, NDCG@k (k=5,10).
* **Binarized view (optional):** micro/macro F1 at a TF-IDF threshold (e.g., >0 → 1).

---

## 6) Prototype Construction & Refresh

* Build prototypes `C ∈ ℝ^{K×d}` **after training** (or periodically during training, e.g., every 2–3 epochs).
* Use **train drugs only** (no leakage).
* Threshold for inclusion: `t_d[k] > 0` (or > small ε like `1e-6`).
* If an ADR has **no supporting drugs**, set `c_k = 0` and mask it at inference (or let it rank naturally—will fall to bottom).

---

## 7) Handling Imbalance & Sampling

* **pos_weight** in DTI BCE = `N_neg / N_pos` on train.
* **Balanced mini-batches**: equal positives/negatives when possible.
* Optional **hard negative mining**: keep top-N highest-scoring negatives from previous epoch and upsample them.

---

## 8) Reproducibility & Logging

* Set seeds for Python/NumPy/PyTorch; deterministic flags if needed.
* Log:

  * Loss terms: DTI, ADR, DP, DA, and total.
  * DTI: PR-AUC/ROC-AUC, F1 (with current threshold).
  * ADR: RMSE/MAE, Recall@k, NDCG@k.
* Save:

  * Best model by **val DTI PR-AUC**.
  * Final **C prototypes** and scaler (if any).
  * Config file with hyperparams and encoder choices.

---

## 9) Minimal Class Skeletons (PyTorch-style)

```python
class Adapter(nn.Module):
    def __init__(self, in_dim, d, hid_ratio=2, p_drop=0.1):
        super().__init__()
        h = hid_ratio * d
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(h, d), nn.LayerNorm(d)
        )
    def forward(self, x): return self.net(x)

class DTIHeadCosine(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
    def forward(self, u, v):
        u = F.normalize(u, dim=-1); v = F.normalize(v, dim=-1)
        cos = (u * v).sum(dim=-1)
        return torch.exp(self.log_scale) * cos  # logits

class DTIHeadBilinear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, u, v):
        return (u * self.W(v)).sum(dim=-1) + self.b

class ADRHead(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.fc = nn.Linear(d, K)
    def forward(self, u):
        y = self.fc(u)
        return F.softplus(y)  # >= 0

def info_nce_bidirectional(A, B, tau=0.07):
    A = F.normalize(A, dim=-1); B = F.normalize(B, dim=-1)
    logits = A @ B.t() / tau           # [B, B]
    labels = torch.arange(A.size(0), device=A.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    return 0.5*(loss1 + loss2)

class PairADRScorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.5))
    @torch.no_grad()
    def forward(self, u, v, C):  # u,v:[d], C:[K,d]
        s = (u @ C.t())             # [K]
        m = (v @ C.t())             # [K]
        r = (u * v).sum()           # scalar
        score = torch.sigmoid(self.alpha*s + self.beta*m + self.gamma*r)
        return score                # [K]
```

---

## 10) Training Loop Outline

```python
model = {
  "fd": Adapter(n_d, d),
  "fp": Adapter(n_p, d),
  "fa": Adapter(K, d),
  "dti": DTIHeadCosine(d),           # or DTIHeadBilinear
  "adr": ADRHead(d, K)
}
opt = torch.optim.AdamW(
    params=[*model["fd"].parameters(), *model["fp"].parameters(),
            *model["fa"].parameters(), *model["dti"].parameters(),
            *model["adr"].parameters()],
    lr=2e-4, weight_decay=1e-4
)

for epoch in range(EPOCHS):
    for batch in loader_train:
        x_d, x_p, y_dti, t_d, drug_idx = batch

        u = model["fd"](x_d)                     # [B,d]
        v = model["fp"](x_p)                     # [B,d]
        # unique drugs for ADR:
        uU, tU = unique_by(drug_idx, u, t_d)     # [U,d], [U,K]
        w = model["fa"](tU)                      # [U,d]

        # Supervised
        logit = model["dti"](u, v)               # [B]
        L_dti = F.binary_cross_entropy_with_logits(
                    logit, y_dti.float(),
                    pos_weight=torch.tensor(pos_weight, device=logit.device))
        yhat_adr = model["adr"](uU)              # [U,K]
        L_adr = huber_loss(yhat_adr, tU, delta=1.0, weights=per_adr_weights)

        # Contrastive
        u_pos, v_pos = select_positives(u, v, y_dti)
        L_dp = info_nce_bidirectional(u_pos, v_pos, tau=0.07)
        L_da = info_nce_bidirectional(uU, w, tau=0.07)

        loss = 1.0*L_dti + 0.5*L_adr + 0.2*(L_dp + L_da)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(chain_params(model), 1.0)
        opt.step()

    # validation: compute PR-AUC, select threshold; compute ADR metrics
    # early-stopping and checkpoint best model
```

**Notes:**

* `unique_by` returns per-batch unique drugs to avoid redundant ADR computation.
* `select_positives` gathers positive pairs for DP contrast; if batch has few positives, keep DP weight small or use a queue/memory bank to enrich negatives.
* `per_adr_weights` are optional inverse-prevalence weights.

---

## 11) Inference Procedure

```
# Given: trained model + prototypes C ∈ ℝ^{K×d} (precomputed)
# Input: x_d, x_p

with torch.no_grad():
    u = fd(x_d); v = fp(x_p)
    logit = dti(u, v)
    p_bind = torch.sigmoid(logit).item()

    score = PairADRScorer(d)(u, v, C)     # [K]
    topk_idx = torch.topk(score, k=K_top).indices
    topk_scores = score[topk_idx]

    final = p_bind * score                # optional joint risk
    topk_joint = torch.topk(final, k=K_top)

return p_bind, topk_idx, topk_scores, (topk_joint if used)
```

---

## 12) Sanity Checks & Gotchas

* **IDF leakage:** Ensure TF-IDF `idf` is fit on **train only**; apply same columns/order to val/test.
* **Scaling:** LayerNorm in adapters stabilizes cosine similarity & contrastive learning.
* **Sparsity:** `t_d` is sparse; Softplus on ADR head prevents negative predictions.
* **Thresholding:** Always pick the DTI decision threshold on **validation** (optimize F1 or desired precision).
* **Prototypes:** Build from **train drugs** only; refresh once after training; cache to disk (`C.npy`).

---

## 13) Recommended Defaults (good first run)

* `d = 512`, `hidden_ratio = 2`, `dropout = 0.1`
* **Heads:** Cosine DTI with learnable scale; ADR head with Softplus.
* **Loss weights:** `λ_dti=1.0`, `λ_adr=0.5`, `λ_con=0.2`, `τ=0.07`, Huber δ=1.0
* **Optimizer:** AdamW `lr=2e-4`, `wd=1e-4`
* **Batch:** 256, balanced pos/neg; 30–50 epochs; early stop on val PR-AUC
* **Metrics:** DTI PR-AUC primary; ADR Recall@k (5,10), NDCG@k; RMSE

---

## 14) Extension Hooks (optional, safe)

* **Hard-negative mining** for DTI after epoch 3+.
* **EMA weights** (exponential moving average) for smoother eval.
* **Temperature annealing** for contrastive (`τ` from 0.1 → 0.07).
* **Label smoothing** for ADR regression to reduce overconfidence on zeros.

---

This spec is intentionally “close to code.” Drop the class skeletons into a repo, wire the data loaders, and you’ll be able to swap encoder pairs and compare them apples-to-apples while producing **binding probabilities + pair-conditioned top-k ADRs**.
