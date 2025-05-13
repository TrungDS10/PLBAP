import torch
import torch.nn as nn

class BAPULM(nn.Module):
    def __init__(self):
        super(BAPULM, self).__init__()
        
        # --- UPDATED INPUT DIMENSIONS ---
        # Protein embedding input dimension is 640 (from ESM)
        self.prot_linear = nn.Linear(640, 512) 
        # Molecule embedding input dimension is NOW 384 (from DeepChem/ChemBERTa-77M-MTR)
        self.mol_linear = nn.Linear(384, 512) 
        # --- END UPDATED INPUT DIMENSIONS ---

        # The rest of the network expects the concatenated dimension (512 + 512 = 1024)
        self.norm = nn.BatchNorm1d(1024, eps=0.001, momentum=0.1, affine=True)
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(1024, 768)
        self.linear2 = nn.Linear(768, 512)
        self.linear3 = nn.Linear(512, 32)
        self.final_linear = nn.Linear(32, 1)

    def forward(self, prot, mol):
        # This part remains the same structurally
        prot_output = torch.relu(self.prot_linear(prot)) # Expects 640-dim input
        mol_output = torch.relu(self.mol_linear(mol))   # NOW expects 384-dim input
        
        combined_output = torch.cat((prot_output, mol_output), dim=1) # Shape: (batch_size, 1024)
        
        # The rest of the forward pass remains the same
        combined_output = self.norm(combined_output)
        combined_output = self.dropout(combined_output)
        x = torch.relu(self.linear1(combined_output))
        x = torch.relu(self.linear2(x))
        x = self.dropout(x) # Original placement
        x = torch.relu(self.linear3(x))
        output = self.final_linear(x)
        return output