import pandas as pd
from pathlib import Path

# ---- paths ----
DATA_DIR = Path(".")  # change if needed
VOCAB = DATA_DIR / "vocab_meddra_adverse_effect.csv"
PROD_AE = DATA_DIR / "product_adverse_effect.csv"
PROD_RX = DATA_DIR / "product_to_rxnorm.csv"
OUT_CSV = DATA_DIR / "final_rxnorm_meddra.csv"

# ---- dtype contracts ----
vocab_dtypes = {
    "meddra_id": "int64",
    "meddra_name": "string",
    # ignore extra columns safely
}
prod_rx_dtypes = {
    "label_id": "int64",
    "rxnorm_product_id": "string",
}
# Only read the columns we actually need from the huge file
prod_ae_usecols = ["product_label_id", "effect_meddra_id"]
prod_ae_dtypes = {
    "product_label_id": "int64",
    "effect_meddra_id": "int64",
}

# ---- load smaller tables fully with strict dtypes ----
vocab = pd.read_csv(VOCAB, dtype={"meddra_id": "Int64", "meddra_name": "string"}, usecols=["meddra_id", "meddra_name"])
# normalize to plain int64 for merge speed
vocab["meddra_id"] = vocab["meddra_id"].astype("int64")

prod_rx = pd.read_csv(PROD_RX, dtype={"label_id": "Int64", "rxnorm_product_id": "string"}, usecols=["label_id", "rxnorm_product_id"])
prod_rx["label_id"] = prod_rx["label_id"].astype("int64")

# ---- prepare output file ----
# create header once
pd.DataFrame(columns=["rxnorm_product_id", "meddra_id", "meddra_name"]).to_csv(OUT_CSV, index=False)

# ---- stream the big file and merge in chunks ----
chunksize = 1_000_000  # tune for your RAM
chunk_iter = pd.read_csv(
    PROD_AE,
    usecols=prod_ae_usecols,
    dtype={"product_label_id": "Int64", "effect_meddra_id": "Int64"},
    chunksize=chunksize,
    low_memory=False,
)

for i, chunk in enumerate(chunk_iter, 1):
    # normalize dtypes exactly
    chunk["product_label_id"] = chunk["product_label_id"].astype("int64")
    chunk["effect_meddra_id"] = chunk["effect_meddra_id"].astype("int64")

    # 1) effect_meddra_id -> vocab (get names)
    c1 = chunk.merge(
        vocab,
        left_on="effect_meddra_id",
        right_on="meddra_id",
        how="inner",
        copy=False
    )[["product_label_id", "meddra_id", "meddra_name"]]

    # 2) product_label_id -> product_to_rxnorm (get rxnorm product)
    c2 = c1.merge(
        prod_rx,
        left_on="product_label_id",
        right_on="label_id",
        how="inner",
        copy=False
    )[["rxnorm_product_id", "meddra_id", "meddra_name"]]

    # enforce final schema
    c2["rxnorm_product_id"] = c2["rxnorm_product_id"].astype("string")
    c2["meddra_id"] = c2["meddra_id"].astype("int64")
    c2["meddra_name"] = c2["meddra_name"].astype("string")

    # optional de-dup (comment out if you want raw many-to-many)
    c2 = c2.drop_duplicates()

    # append to output
    c2.to_csv(OUT_CSV, mode="a", header=False, index=False)

    # simple progress
    print(f"Processed chunk {i}, wrote {len(c2):,} rows")

print(f"Done. Wrote merged file: {OUT_CSV}")
