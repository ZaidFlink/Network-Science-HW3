import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def save_data(data, output_path):
    """
    Save processed data to a text file for later use.
    Convert complex objects to string representations with improved formatting.
    """
    # Convert data to string representation
    edges = list(data["graph"].edges())
    output_str = {
        "graph": "[\n    " + ",\n    ".join(str(edge) for edge in edges) + "\n]",
        "features": "[\n    " + ",\n    ".join(str(row) for row in data["features"].tolist()) + "\n]",
        "labels": str(data["labels"].tolist()),
        "train_mask": str(data["train_mask"].tolist()),
        "test_mask": str(data["test_mask"].tolist())
    }
    
    with open(output_path, 'w') as f:
        for key, value in output_str.items():
            f.write(f"{key}:\n{value}\n\n")

def load_data(dataset_name, root_dir=None, output_dir=None):
    """
    Load graph, features, labels, and train/test split from dataset files.
    Save processed data to output directory.
    """
    # Set default paths relative to script location
    if root_dir is None:
        root_dir = os.path.join(SCRIPT_DIR, '..', 'real-node-label')
    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, '..', 'output')
    
    dataset_path = os.path.join(root_dir, dataset_name)
    
    # Load graph structure
    graph = load_pickle(os.path.join(dataset_path, f"ind.{dataset_name}.graph"))
    G = nx.from_dict_of_lists(graph)
    
    # Load features
    x = load_pickle(os.path.join(dataset_path, f"ind.{dataset_name}.x"))
    tx = load_pickle(os.path.join(dataset_path, f"ind.{dataset_name}.tx"))
    allx = load_pickle(os.path.join(dataset_path, f"ind.{dataset_name}.allx"))
    
    features = sp.vstack((allx, tx)).tolil()
    features = torch.FloatTensor(np.array(features.todense()))
    
    # Load labels
    y = load_pickle(os.path.join(dataset_path, f"ind.{dataset_name}.y"))
    ty = load_pickle(os.path.join(dataset_path, f"ind.{dataset_name}.ty"))
    ally = load_pickle(os.path.join(dataset_path, f"ind.{dataset_name}.ally"))
    
    labels = np.vstack((ally, ty))
    labels = torch.LongTensor(labels.argmax(axis=1))
    
    # Load test index
    test_idx = np.loadtxt(os.path.join(dataset_path, f"ind.{dataset_name}.test.index"), dtype=int)
    test_mask = torch.BoolTensor([i in test_idx for i in range(labels.shape[0])])
    train_mask = ~test_mask
    
    data = {
        "graph": G,
        "features": features,
        "labels": labels,
        "train_mask": train_mask,
        "test_mask": test_mask
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_processed.txt")  # Changed extension to .txt
    save_data(data, output_file)
    
    return data

if __name__ == "__main__":
    dataset_names = ["citeseer", "cora", "pubmed"]
    for dataset in dataset_names:
        data = load_data(dataset)
        print(f"{dataset} dataset processed and saved to version2.1/output/{dataset}_processed.txt")  # Updated message
