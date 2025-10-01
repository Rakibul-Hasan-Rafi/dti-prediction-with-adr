# **Methodology: Dimensionality Reduction of ADR Data**

### 1. Data Preparation

The adverse drug reaction (ADR) dataset consisted of pairwise associations between drugs, identified using **RxNorm ingredient IDs**, and ADRs, identified using **MedDRA IDs**. Each entry denoted the presence of an ADR for a given drug.

Preprocessing steps included:

* **Standardization:** Drug IDs were cast as strings, ADR IDs as integers.
* **Deduplication:** Duplicate drug–ADR pairs were removed to avoid multiple counts of the same association.
* **Matrix construction:** A sparse incidence matrix (X) of size *(number of drugs × number of ADRs)* was generated, where:

  * Rows represented drugs.
  * Columns represented ADRs.
  * Entries were `1` if the ADR was associated with the drug, `0` otherwise.

In our dataset, this resulted in **1028 drugs** and **4817 ADR terms**, with a sparsity of approximately **98.6%**.

---

### 2. Motivation for Dimensionality Reduction

The raw ADR matrix is **unsuitable** for direct use in predictive modeling for several reasons:

1. **High dimensionality:** Each drug is represented by 4,817 binary features, which can lead to overfitting.
2. **Sparsity:** The dataset is dominated by zeros, reducing statistical efficiency.
3. **Independence assumption:** One-hot encoding treats ADRs as unrelated, even though clinically similar ADRs often co-occur (e.g., headache and migraine).

A **compact, dense representation** of ADR information was therefore necessary to capture meaningful co-occurrence structure and improve downstream learning efficiency.

---

### 3. Dimensionality Reduction with Truncated SVD

To address these challenges, we applied **Truncated Singular Value Decomposition (SVD)**, a linear dimensionality reduction method particularly well-suited for sparse data.

* **Rationale:** Unlike traditional PCA, Truncated SVD can directly operate on sparse matrices in Compressed Sparse Row (CSR) format. It efficiently discovers latent orthogonal components that explain the greatest variance in the dataset.
* **Mathematical basis:** The drug–ADR matrix (X) was approximated as:
  [
  X \approx U_k \Sigma_k V_k^T
  ]
  where (U_k \Sigma_k) provides low-dimensional embeddings for drugs and (V_k) provides loadings for ADR terms.
* **Outcome:** Each drug was projected into a dense **k-dimensional ADR embedding**, where k is the chosen latent dimensionality.

---

### 4. Selection of Embedding Dimensionality

The number of latent components (k) was selected based on explained variance ratio:

* **k = 64:** ~52% variance explained.
* **k = 256:** ~85% variance explained.
* **k = 512:** ~96% variance explained (near-lossless but computationally heavier).

We adopted **k = 256** as the optimal balance between compactness and information retention. Each drug was thus represented by a **256-length ADR fingerprint**, capturing most co-occurrence structure while remaining computationally efficient.

---

### 5. Embedding Construction and Export

For each drug, the Truncated SVD pipeline produced a **256-dimensional vector**. These vectors were stored in a **Parquet file** with the following schema:

* `rxnorm_ingredient_id` (string): unique identifier of the drug.
* `adr_svd_0 … adr_svd_255` (float): the ADR embedding components.

This ensured that each drug was represented by exactly one ADR feature vector.

To guarantee reproducibility, the trained SVD pipeline, the ADR axis order, and associated metadata (e.g., number of components, explained variance ratio) were also stored alongside the embeddings.

---

### 6. Significance of ADR Embeddings

The ADR embeddings provide a **single, dense, reusable feature vector per drug**, which can be integrated into downstream drug–protein interaction (DTI) prediction models.

* **Compactness:** Reduces 4,817 sparse features to 256 dense features.
* **Information retention:** Preserves ~85% of the variance in ADR co-occurrence.
* **Semantic clustering:** Places drugs with similar ADR profiles close together in embedding space.
* **Reusability:** Enables consistent merging into DTI datasets via `rxnorm_ingredient_id`.

This representation allows ADR knowledge to be fused with drug structural and protein features, supporting safety-aware DTI prediction and drug ranking.

