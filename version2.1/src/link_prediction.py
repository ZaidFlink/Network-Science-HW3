import os
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import random
from train_model import load_processed_data
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../", "output")

class GAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAE, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 2 * out_channels)
        self.conv2 = pyg_nn.GCNConv(2 * out_channels, out_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)
    
    def decode(self, z, edge_label_index):
        value = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        return value.view(-1)
    
    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, data.edge_index)
    
def compute_auc(pos_scores, neg_scores):
    """Compute Area Under the ROC Curve (AUC) from positive and negative scores."""
    y_true = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]).numpy()
    y_pred = torch.cat([pos_scores, neg_scores]).sigmoid().cpu().numpy()
    return roc_auc_score(y_true, y_pred)

def manual_link_split(data, test_ratio=0.2):
    """Manually split links for prediction task."""
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    
    # Create a random permutation of edge indices
    perm = torch.randperm(num_edges // 2)
    
    # Get the number of test edges
    num_test = int(num_edges // 2 * test_ratio)
    
    # Select test edges
    test_edge_idx = perm[:num_test]
    train_edge_idx = perm[num_test:]
    
    # Due to undirected graph, each edge appears twice (in both directions)
    # Adjust indices to get both directions
    train_edges = edge_index[:, torch.cat([train_edge_idx, train_edge_idx + num_edges // 2])]
    test_edges = edge_index[:, torch.cat([test_edge_idx, test_edge_idx + num_edges // 2])]
    
    # Create negative samples for testing
    num_nodes = data.num_nodes
    neg_edges = []
    
    # Generate negative edges (at least as many as positive test edges)
    while len(neg_edges) < test_edges.size(1):
        # Random node pairs
        neg_edge = torch.randint(0, num_nodes, (2,), device=edge_index.device)
        # Check if this edge exists in train or test sets
        if not ((neg_edge[0] == edge_index[0]) & (neg_edge[1] == edge_index[1])).any():
            neg_edges.append(neg_edge)
    
    neg_edge_index = torch.stack(neg_edges, dim=1)
    
    # Create new data objects for train and test
    train_data = Data(
        x=data.x,
        edge_index=train_edges,
        y=data.y
    )
    
    # For test data, we need positive and negative edges
    test_pos_edge_index = test_edges
    test_neg_edge_index = neg_edge_index
    
    return train_data, test_pos_edge_index, test_neg_edge_index

def train_link_prediction(model, data, runs=10, epochs=200):
    auc_scores = []
    
    for run in range(runs):
        torch.manual_seed(run)
        np.random.seed(run)
        random.seed(run)
        
        # Manual splitting approach
        train_data, test_pos_edge_index, test_neg_edge_index = manual_link_split(data)
        
        # Move data to device
        train_data = train_data.to(device)
        test_pos_edge_index = test_pos_edge_index.to(device)
        test_neg_edge_index = test_neg_edge_index.to(device)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Generate training negative edges for this epoch
            neg_edge_index = generate_negative_edges(train_data.edge_index, train_data.num_nodes, 
                                                    train_data.edge_index.size(1))
            
            z = model.encode(train_data.x, train_data.edge_index)
            
            # Compute loss for positive edges
            pos_out = model.decode(z, train_data.edge_index)
            pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean()
            
            # Compute loss for negative edges
            neg_out = model.decode(z, neg_edge_index)
            neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
            
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            z = model.encode(train_data.x, train_data.edge_index)
            
            # Get scores for positive test edges
            pos_scores = model.decode(z, test_pos_edge_index)
            
            # Get scores for negative test edges
            neg_scores = model.decode(z, test_neg_edge_index)
            
            # Compute AUC
            auc = compute_auc(pos_scores, neg_scores)
            auc_scores.append(auc)
            print(f"Run {run+1}/{runs}, AUC: {auc:.4f}")

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    return mean_auc, std_auc

def generate_negative_edges(edge_index, num_nodes, num_samples):
    """Generate negative edges (edges that don't exist in the graph)."""
    neg_edges = []
    existing_edges = set([(edge_index[0, i].item(), edge_index[1, i].item()) 
                         for i in range(edge_index.size(1))])
    
    while len(neg_edges) < num_samples:
        # Random node pair
        i, j = random.randint(0, num_nodes-1), random.randint(0, num_nodes-1)
        if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
            neg_edges.append([i, j])
            existing_edges.add((i, j))
    
    return torch.tensor(neg_edges, dtype=torch.long, device=edge_index.device).t()


if __name__ == "__main__":
    datasets = ["citeseer", "cora", "pubmed"]
    md_content = "# Link Prediction Performance\n\n"

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        data, train_mask, test_mask = load_processed_data(dataset)
        
        data = data.to(device)
        input_dim = data.x.shape[1]

        # Try multiple link prediction models
        models = {
            "GAE": GAE(input_dim, 16),
        }
        
        md_content += f"## Dataset: {dataset}\n\n"
        md_content += "| Model | AUC | Std Dev |\n"
        md_content += "|-------|------|----------|\n"
        
        for model_name, model in models.items():
            print(f"Training {model_name} on {dataset}...")
            mean_auc, std_auc = train_link_prediction(model, data)
            md_content += f"| {model_name} | {mean_auc:.4f} | {std_auc:.4f} |\n"
            print(f"{model_name} on {dataset}: AUC: {mean_auc:.4f}, Std Dev: {std_auc:.4f}")
        
        md_content += "\n"

    output_path = os.path.join(OUTPUT_DIR, "link_prediction_output.md")
    with open(output_path, "w") as f:
        f.write(md_content)

    print(f"\nResults have been saved to {output_path}")
        
        
            

    