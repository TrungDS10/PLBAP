import torch
import re
# Import the correct tokenizers and models from transformers
from transformers import AutoTokenizer, AutoModel # General purpose for ESM and ChemBERTa

# Note: T5Tokenizer and T5EncoderModel are for ProtT5, which we are replacing.

class EmbeddingExtractor:
    def __init__(self, device_str): # Changed 'device' to 'device_str' to avoid conflict if torch is imported globally
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        print(f"EmbeddingExtractor using device: {self.device}")

        # --- Load ESM for Protein Embeddings ---
        self.prot_model_name = "facebook/esm2_t30_150M_UR50D" # Or your chosen ESM model
        print(f"Loading protein model: {self.prot_model_name}")
        self.prot_tokenizer = AutoTokenizer.from_pretrained(self.prot_model_name)
        self.prot_model = AutoModel.from_pretrained(self.prot_model_name).to(self.device).eval()

        # --- Load ChemBERTa for Molecule Embeddings ---
        self.mol_model_name = "DeepChem/ChemBERTa-77M-MTR" # Or your chosen ChemBERTa model
        print(f"Loading molecule model: {self.mol_model_name}")
        self.mol_tokenizer = AutoTokenizer.from_pretrained(self.mol_model_name)
        self.mol_model = AutoModel.from_pretrained(self.mol_model_name).to(self.device).eval()

        # Define max lengths for tokenization, consistent with dataset_creation.py
        self.max_prot_len = 1024
        self.max_smiles_len = 512 # Crucial: <= ChemBERTa's max_position_embeddings (515)

    def get_protein_embedding(self, sequence):
        """
        Generates ESM embedding for a single protein sequence.
        """
        # Assuming sequence is already preprocessed if needed (UZOB -> X)
        
        inputs = self.prot_tokenizer(
            sequence, # Can be a single sequence string or a list of one string
            padding="max_length", # Or True, or 'longest' if batching multiple
            truncation=True,
            max_length=self.max_prot_len,
            return_tensors='pt',
            return_attention_mask=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.prot_model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask # Shape: (1, embedding_dim)
        
        return mean_embedding # Return as tensor, will be handled by inference script

    def get_molecule_embedding(self, smiles):
        """
        Generates ChemBERTa embedding for a single SMILES string.
        """
        inputs = self.mol_tokenizer(
            smiles, # Can be a single SMILES string or a list of one string
            padding="max_length", # Or True, or 'longest' if batching multiple
            truncation=True,
            max_length=self.max_smiles_len,
            return_tensors='pt',
            return_attention_mask=True
        )
        # Explicitly create token_type_ids for RoBERTa-based models like ChemBERTa
        inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'], device=self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}


        with torch.no_grad():
            outputs = self.mol_model(**inputs)
        
        # Use the embedding of the [CLS] token for ChemBERTa
        cls_embedding = outputs.last_hidden_state[:, 0, :] # Shape: (1, embedding_dim)
        return cls_embedding # Return as tensor

    def get_combined_embedding(self, sequence, smiles):
        """
        Generates and returns embeddings for a single protein sequence and a single SMILES string.
        Input:
            sequence (str): A single protein amino acid sequence.
            smiles (str): A single SMILES string.
        Output:
            prot_embedding (torch.Tensor): Protein embedding tensor.
            mol_embedding (torch.Tensor): Molecule embedding tensor.
        """
        # Ensure sequence is preprocessed before getting embedding
        # sequence_processed = preprocess_protein_sequence(sequence) # Call the actual function
        # prot_embedding = self.get_protein_embedding(sequence_processed)
        
        # Assuming preprocessing happens outside this specific method now
        prot_embedding = self.get_protein_embedding(sequence) 
        mol_embedding = self.get_molecule_embedding(smiles)
        return prot_embedding, mol_embedding

# --- Preprocessing Function ---
# Keep the function with the specific name containing the logic
def preprocess_protein_sequence(sequence_str):
    """
    Applies the same preprocessing to protein sequences as used in BAPULM.
    Replaces U, Z, O, B with X.
    """
    if not isinstance(sequence_str, str):
        # Handle potential non-string inputs gracefully if needed
        # print(f"Warning: preprocess_protein_sequence received non-string input: {type(sequence_str)}. Returning as is.")
        return sequence_str # Or raise an error, or convert if possible
    processed_sequence = re.sub(r"[UZOB]", "X", sequence_str)
    return processed_sequence

# --- Alias for Compatibility ---
# Assign the function to the name expected by the original inference.py
preprocess_function = preprocess_protein_sequence
# --- End Alias ---

