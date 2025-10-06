import pandas as pd
import torch
from typing import List, Dict

class TFIDFADRMatrix:
    def __init__(self, adr_index, drug_index, idf_table):

        # Load the mapping tables
        self.adr_index = adr_index
        self.drug_index = drug_index
        self.idf_table = idf_table
        
        # Create dictionaries for fast lookup
        self.meddra_to_col = dict(zip(self.adr_index['meddra_id'], self.adr_index['col_id']))
        self.col_to_meddra = dict(zip(self.adr_index['col_id'], self.adr_index['meddra_id']))
        self.rxcui_to_row = dict(zip(self.drug_index['rxcui'], self.drug_index['row_id']))
        self.row_to_rxcui = dict(zip(self.drug_index['row_id'], self.drug_index['rxcui']))
        
        # Store dimensions
        self.num_drugs = len(self.drug_index)
        self.num_adrs = len(self.adr_index)
        
        self.tfidf_matrix = None
        
        
        print(f"Loaded TF-IDF data: {self.num_drugs} drugs, {self.num_adrs} ADRs")
    
    def load_tfidf_matrix(self, matrix_path: str):
        pass
    
    def get_drug_adr_scores(self, rxcui: str, top_k: int = None) -> Dict[int, float]:

        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not loaded")
        
        if rxcui not in self.rxcui_to_row:
            raise ValueError(f"Drug {rxcui} not found in index")
        
        row_id = self.rxcui_to_row[rxcui]
        drug_scores = self.tfidf_matrix[row_id, :]
        
        # Map to meddra_ids
        results = {}
        for col_id, score in enumerate(drug_scores):
            if score > 0:  # Only non-zero scores
                meddra_id = self.col_to_meddra[col_id]
                results[meddra_id] = float(score)
        
        # Sort by score descending
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        if top_k:
            return dict(list(sorted_results.items())[:top_k])
        return sorted_results
    
    def get_adr_drug_scores(self, meddra_id: int, top_k: int = None) -> Dict[str, float]:

        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not loaded")
        
        if meddra_id not in self.meddra_to_col:
            raise ValueError(f"ADR {meddra_id} not found in index")
        
        col_id = self.meddra_to_col[meddra_id]
        adr_scores = self.tfidf_matrix[:, col_id]
        
        # Map to rxcuis
        results = {}
        for row_id, score in enumerate(adr_scores):
            if score > 0:  # Only non-zero scores
                rxcui = self.row_to_rxcui[row_id]
                results[rxcui] = float(score)
        
        # Sort by score descending
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        if top_k:
            return dict(list(sorted_results.items())[:top_k])
        return sorted_results
    
    def filter_common_adrs(self, idf_threshold: float = 1.0) -> List[int]:

        filtered_adrs = self.idf_table[self.idf_table['idf'] > idf_threshold]['meddra_id'].tolist()
        
        print(f"Filtered ADRs: {len(filtered_adrs)}/{len(self.idf_table)} "
              f"({len(filtered_adrs)/len(self.idf_table):.1%}) remain with IDF > {idf_threshold}")
        
        return filtered_adrs
    
    def get_adr_stats(self, meddra_id: int) -> Dict:

        if meddra_id not in self.meddra_to_col:
            raise ValueError(f"ADR {meddra_id} not found in index")
        
        stats = self.idf_table[self.idf_table['meddra_id'] == meddra_id]
        if len(stats) == 0:
            return {"meddra_id": meddra_id, "df": 0, "idf": 0.0}
        
        return {
            "meddra_id": meddra_id,
            "df": int(stats['df'].iloc[0]),
            "idf": float(stats['idf'].iloc[0]),
            "col_id": self.meddra_to_col[meddra_id]
        }
    
    def get_drug_vector(self, rxcui: str, filtered_adrs: List[int] = None) -> torch.Tensor:

        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not loaded")
        
        if rxcui not in self.rxcui_to_row:
            raise ValueError(f"Drug {rxcui} not found in index")
        
        row_id = self.rxcui_to_row[rxcui]
        
        if filtered_adrs is None:
            # Return full vector
            return torch.tensor(self.tfidf_matrix[row_id, :], dtype=torch.float32)
        else:
            # Return filtered vector
            filtered_indices = [self.meddra_to_col[adr] for adr in filtered_adrs 
                              if adr in self.meddra_to_col]
            filtered_scores = self.tfidf_matrix[row_id, filtered_indices]
            return torch.tensor(filtered_scores, dtype=torch.float32)
    
    def get_reduced_label_mapping(self, idf_threshold: float = 1.0) -> Dict[int, int]:

        filtered_adrs = self.filter_common_adrs(idf_threshold)
        
        # Create new compact mapping
        reduced_mapping = {meddra_id: new_idx for new_idx, meddra_id in enumerate(filtered_adrs)}
        
        print(f"Reduced output size: {len(reduced_mapping)} (from {self.num_adrs})")
        
        return reduced_mapping
    
    def get_top_adrs_by_idf(self, k: int = 1000) -> List[int]:
        top_adrs = self.idf_table.nlargest(k, 'idf')['meddra_id'].tolist()
        return top_adrs
    
    def get_adr_frequency_stats(self) -> Dict:
        return {
            "total_adrs": len(self.adr_index),
            "total_drugs": len(self.drug_index),
            "avg_df": float(self.idf_table['df'].mean()),
            "avg_idf": float(self.idf_table['idf'].mean()),
            "min_idf": float(self.idf_table['idf'].min()),
            "max_idf": float(self.idf_table['idf'].max()),
            "common_adrs_count": len(self.idf_table[self.idf_table['idf'] < 1.0])
        }

