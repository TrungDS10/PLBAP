import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd # For reading a potential CSV input
import json
from tqdm import tqdm # For a progress bar

# --- Configuration ---
# Model identifiers from Hugging Face
# You can choose different variants based on size and performance needs.
ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
# Other ChemBERTa options: "seyonec/ChemBERTa-zinc-base-v1", "seyonec/PubChem10M_ChemBERTa_100k_MTR"

# Input data file (IMPORTANT: Replace with your actual data source)
# This should be a file containing protein sequences, SMILES strings, and affinity values.
# Example: A CSV with columns 'protein_sequence', 'smiles', 'affinity_pkd'
INPUT_DATA_FILE = "binding_affinity_100k_raw.csv" 
OUTPUT_JSON_FILE = "esm_chemberta_embeddings.json"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Batch size for processing (adjust based on your GPU memory and model size)
BATCH_SIZE = 16 # Start with a smaller batch size if you encounter memory issues
# Max sequence lengths for truncation (check model-specific limits if necessary)
MAX_PROT_LEN = 1024 
MAX_SMILES_LEN = 512 

# --- Helper Functions ---

def get_esm_embedding(sequences, model, tokenizer):
    """
    Generates ESM embeddings for a batch of protein sequences.
    Uses mean pooling of the last hidden states, considering the attention mask.
    """
    # Tokenize sequences. `padding="max_length"` ensures all sequences in a batch
    # are padded to the same length (MAX_PROT_LEN or the longest in batch if tokenizer handles it dynamically).
    # `truncation=True` ensures sequences longer than MAX_PROT_LEN are cut.
    inputs = tokenizer(
        sequences, 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_PROT_LEN, 
        return_tensors="pt", 
        return_attention_mask=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()} # Move inputs to the selected device

    # Get model outputs. `torch.no_grad()` is used as we are doing inference
    # and don't need to compute gradients, saving memory and computation.
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract last hidden states (embeddings for each token)
    last_hidden_states = outputs.last_hidden_state # Shape: (batch_size, seq_len, hidden_size)

    # Perform mean pooling, carefully considering the attention mask to exclude padding tokens.
    attention_mask = inputs['attention_mask']
    # Expand attention_mask to match the shape of last_hidden_states for element-wise multiplication
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    # Sum embeddings where attention_mask is 1
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, dim=1)
    # Count non-padding tokens for each sequence to get the correct denominator for mean
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9) # Avoid division by zero for empty sequences
    
    mean_embeddings = sum_embeddings / sum_mask

    return mean_embeddings.cpu().numpy() # Move to CPU and convert to NumPy array for easier handling

def get_chemberta_embedding(smiles_list, model, tokenizer):
    """
    Generates ChemBERTa embeddings for a batch of SMILES strings.
    Typically uses the embedding of the [CLS] token.
    """
    inputs = tokenizer(
        smiles_list, 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_SMILES_LEN, 
        return_tensors="pt",
        return_attention_mask=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # For many BERT-style models, the embedding of the first token ([CLS])
    # is used as the aggregate representation of the sequence.
    cls_embeddings = outputs.last_hidden_state[:, 0, :] # Shape: (batch_size, hidden_size)
    # Alternatively, some models provide a 'pooler_output' which is a processed version of the [CLS] token.
    # If available and preferred for your chosen ChemBERTa model:
    # cls_embeddings = outputs.pooler_output 
    
    return cls_embeddings.cpu().numpy()

# --- Main Script ---
def main():
    # 1. Load pre-trained models and tokenizers
    print(f"Loading ESM model: {ESM_MODEL_NAME}...")
    esm_tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    esm_model = AutoModel.from_pretrained(ESM_MODEL_NAME).to(DEVICE).eval() # .eval() sets model to evaluation mode

    print(f"Loading ChemBERTa model: {CHEMBERTA_MODEL_NAME}...")
    chemberta_tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_NAME)
    chemberta_model = AutoModel.from_pretrained(CHEMBERTA_MODEL_NAME).to(DEVICE).eval()

    # 2. Load your base dataset
    # This section needs to be adapted based on how your data is stored.
    # The example below assumes a CSV file.
    try:
        df = pd.read_csv(INPUT_DATA_FILE)
        # Ensure your CSV has columns with these exact names or modify accordingly
        protein_sequences = df['protein_sequence'].tolist()
        ligand_smiles = df['smiles'].tolist()
        affinities = df['affinity_pkd'].tolist() # Or whatever your affinity column is named
        print(f"Successfully loaded {len(df)} entries from {INPUT_DATA_FILE}")
    except FileNotFoundError:
        print(f"Error: Input data file '{INPUT_DATA_FILE}' not found.")
        print("Please create this file with columns like 'protein_sequence', 'smiles', 'affinity_pkd'.")
        print("Using dummy data for demonstration purposes only.")
        # Dummy data for demonstration if the file is not found
        protein_sequences = ["MAGAASPCLLPLLALWGPDP", "MKVLWAALLAVLLSACSGH", "MTMKYLMKLSSE", "MERRRLKQQVEE"] * 20 
        ligand_smiles = ["CCO", "CNC(=O)CC", "CC(=O)Oc1ccccc1C(=O)OH", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"] * 20
        affinities = [7.5, 8.1, 6.3, 9.0] * 20
        print(f"Proceeding with {len(protein_sequences)} dummy entries.")
    except KeyError as e:
        print(f"Error: Missing expected column in '{INPUT_DATA_FILE}': {e}")
        print("Please ensure your CSV has 'protein_sequence', 'smiles', and 'affinity_pkd' (or similar) columns.")
        return # Exit if essential columns are missing

    # 3. Process data in batches and generate embeddings
    all_embeddings_data = []
    num_entries = len(protein_sequences)

    # Using tqdm for a progress bar
    for i in tqdm(range(0, num_entries, BATCH_SIZE), desc="Processing batches"):
        batch_protein_seqs = protein_sequences[i:i+BATCH_SIZE]
        batch_ligand_smiles = ligand_smiles[i:i+BATCH_SIZE]
        batch_affinities = affinities[i:i+BATCH_SIZE]

        # Generate embeddings for the current batch
        prot_embeds_batch = get_esm_embedding(batch_protein_seqs, esm_model, esm_tokenizer)
        mol_embeds_batch = get_chemberta_embedding(batch_ligand_smiles, chemberta_model, chemberta_tokenizer)

        # Store the results for this batch
        # The structure should match what BAPULM's BindingAffinityDataset expects
        for j in range(len(batch_protein_seqs)): # Iterate through items in the current batch
            all_embeddings_data.append({
                "prot_embedding": prot_embeds_batch[j].tolist(), # Convert NumPy array to list for JSON
                "mol_embedding": mol_embeds_batch[j].tolist(),   # Convert NumPy array to list for JSON
                "affinity": float(batch_affinities[j])           # Ensure affinity is a float
            })

    # 4. Save the combined embeddings data to a JSON file
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(all_embeddings_data, f, indent=4) # Using indent for better readability of the JSON

    print(f"\nSuccessfully generated and saved embeddings to {OUTPUT_JSON_FILE}")
    print(f"Total entries processed: {len(all_embeddings_data)}")
    if all_embeddings_data:
        print(f"Sample protein embedding dimension: {len(all_embeddings_data[0]['prot_embedding'])}")
        print(f"Sample molecule embedding dimension: {len(all_embeddings_data[0]['mol_embedding'])}")

if __name__ == "__main__":
    main()
