# pip install transformers torch --upgrade
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, EsmModel

def esm_embed_hf(seqs: List[Tuple[str, str]],
                 hub_id: str = "facebook/esm2_t33_650M_UR50D",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    """
    seqs: list of (seq_id, aa_sequence)
    returns:
      {
        'per_residue': {seq_id: torch.FloatTensor[L, D]},
        'per_sequence': {seq_id: torch.FloatTensor[D]}
      }
    """
    tokenizer = AutoTokenizer.from_pretrained(hub_id, do_lower_case=False)
    model = EsmModel.from_pretrained(hub_id).eval().to(device)

    per_residue, per_sequence = {}, {}
    BATCH = 8

    with torch.no_grad():
        for i in range(0, len(seqs), BATCH):
            batch = seqs[i:i+BATCH]
            ids, strings = zip(*batch)

            enc = tokenizer(
                list(strings),
                return_tensors="pt",
                padding=True,
                truncation=True,   # respects model max length (≈1022 tokens incl. specials)
                add_special_tokens=True,
            ).to(device)

            out = model(**enc)  # last_hidden_state: [B, T, D]
            hs = out.last_hidden_state

            # Build mask to exclude padding + special tokens (CLS/EOS)
            pad_id = tokenizer.pad_token_id
            cls_id = tokenizer.cls_token_id
            eos_id = tokenizer.eos_token_id
            ids_mat = enc["input_ids"]                          # [B, T]
            not_pad = (ids_mat != pad_id)
            not_special = (ids_mat != cls_id) & (ids_mat != eos_id)
            valid_mask = (not_pad & not_special).unsqueeze(-1)  # [B, T, 1]

            # Per-residue (variable length) and per-sequence (masked mean)
            for b, sid in enumerate(ids):
                valid_idx = valid_mask[b, :, 0].nonzero(as_tuple=True)[0]
                residue_repr = hs[b, valid_idx, :].detach().cpu()          # [L, D]
                seq_repr = residue_repr.mean(dim=0)                         # [D]
                per_residue[sid] = residue_repr
                per_sequence[sid] = seq_repr

    return {"per_residue": per_residue, "per_sequence": per_sequence}

# ---------- Example ----------
if __name__ == "__main__":
    seqs = [("P05362", "MKTIIALSYIFCLVFADYKDDDDK")]
    embs = esm_embed_hf(seqs)
    print(next(iter(embs["per_sequence"].values())).shape)
