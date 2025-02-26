import os 
import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import numpy as np
import pickle
import random


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../", "output")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_processed_data(dataset_name, output_dir=OUTPUT_DIR):
    file_path = os.path.join(output_dir, f"{dataset_name}_processed.txt")
    with open(file_path, "r") as f:
        content = f.read().split("\n\n")
    
    data_dict = {}
    for section in content:
        if section.strip():
            key, value = section.split(":\n", 1)
            if key == "graph":
                edges = eval(value)
                # Convert to edge_index format and ensure indices are within bounds
                edge_index = torch.tensor([[e[0] for e in edges], 
                                         [e[1] for e in edges]], dtype=torch.long)
                
                # Get number of nodes from features
                num_nodes = None
                if "features" in data_dict:
                    num_nodes = data_dict["features"].shape[0]
                
                # Validate edge indices
                if num_nodes is not None:
                    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                    edge_index = edge_index[:, mask]
                
                data_dict["edge_index"] = edge_index
            elif key == "features":
                features = torch.tensor(eval(value), dtype=torch.float)
                data_dict["features"] = features
                
                # If we already have edge_index, validate it against number of nodes
                if "edge_index" in data_dict:
                    num_nodes = features.shape[0]
                    edge_index = data_dict["edge_index"]
                    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                    data_dict["edge_index"] = edge_index[:, mask]
            elif key == "labels":
                labels = torch.tensor(eval(value), dtype=torch.long)
                data_dict["labels"] = labels
            elif key == "train_mask":
                train_mask = torch.tensor(eval(value), dtype=torch.bool)
                data_dict["train_mask"] = train_mask
            elif key == "test_mask":
                test_mask = torch.tensor(eval(value), dtype=torch.bool)
                data_dict["test_mask"] = test_mask

    # Create PyG Data object
    data = Data(x=data_dict["features"], 
                edge_index=data_dict["edge_index"], 
                y=data_dict["labels"])
    
    # Move data to device
    data = data.to(device)
    train_mask = data_dict["train_mask"].to(device)
    test_mask = data_dict["test_mask"].to(device)
    
    return data, data_dict["train_mask"], data_dict["test_mask"]

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()

        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAT, self).__init__()
        
        self.conv1 = pyg_nn.GATConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GATConv(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
                
class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        
        self.conv1 = pyg_nn.SAGEConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.SAGEConv(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
def train_and_evaluate(model, data, train_mask, test_mask, runs=10,epochs=200):
    # Set random seeds for reproducibility
    torch.manual_seed(runs)
    np.random.seed(runs)
    random.seed(runs)
    
    accuracies = []
    for run in range(runs):
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        pred = model(data).argmax(dim=1)
        acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    var_acc = np.var(accuracies)
    return mean_acc, var_acc


if __name__ == "__main__":
    datasets = ["citeseer", "cora", "pubmed"]
    models = {"GCN": GCN, "GAT": GAT, "GraphSAGE": GraphSAGE}
    
    # Create markdown content
    md_content = "# Graph Neural Network Models Performance\n\n"
    
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        data, train_mask, test_mask = load_processed_data(dataset)
        input_dim = data.x.shape[1]
        output_dim = len(torch.unique(data.y))
        
        md_content += f"## Dataset: {dataset}\n\n"
        md_content += "| Model | Accuracy | Variance |\n"
        md_content += "|-------|----------|----------|\n"
        
        print(f"Dataset: {dataset}")
        for model_name, model_class in models.items():
            model = model_class(input_dim, 16, output_dim)
            mean_acc, var_acc = train_and_evaluate(model, data, train_mask, test_mask)
            print(f"  {model_name} - Accuracy: {mean_acc:.4f}, Variance: {var_acc:.4f}")
            
            # Add results to markdown table
            md_content += f"| {model_name} | {mean_acc:.4f} | {var_acc:.4f} |\n"
        
        md_content += "\n"
    
    # Save markdown file
    output_path = os.path.join(OUTPUT_DIR, "train_models_output.md")
    
    with open(output_path, "w") as f:
        f.write(md_content)
    
    print(f"\nResults have been saved to {output_path}")


            
