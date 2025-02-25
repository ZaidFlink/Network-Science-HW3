from data_utils.data_loader import CoraDataLoader
from models.gcn import GCNModel
import torch
import numpy as np
import json
from datetime import datetime
import os

def run_gcn_experiment(data, n_runs=10):
    """Run GCN experiment multiple times"""
    results = []
    
    print(f"Running GCN experiment ({n_runs} runs)...")
    
    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")
        
        # Initialize model
        model = GCNModel(
            nfeat=data['features'].shape[1],
            nclass=len(np.unique(data['labels'])),
            nhid=16,
            dropout=0.5,
            lr=0.01,
            weight_decay=5e-4,
            epochs=200
        )
        
        # Train and evaluate
        run_results = model.fit_predict(data)
        results.append(run_results)
    
    # Calculate statistics
    metrics = ['train_acc', 'val_acc', 'test_acc', 'train_f1', 'val_f1', 'test_f1']
    final_results = {}
    
    for metric in metrics:
        values = [r[metric] for r in results]
        final_results[f'{metric}_mean'] = float(np.mean(values))
        final_results[f'{metric}_std'] = float(np.std(values))
    
    print("\nOverall Results:")
    print(f"Test Accuracy: {final_results['test_acc_mean']:.4f} ± {final_results['test_acc_std']:.4f}")
    print(f"Test F1 Score: {final_results['test_f1_mean']:.4f} ± {final_results['test_f1_std']:.4f}")
    
    return final_results

def main():
    print("Loading Cora dataset...")
    try:
        loader = CoraDataLoader()
        data = loader.load_data()
        print("Data loaded successfully!")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Run experiment
        results = run_gcn_experiment(data, n_runs=10)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"gcn_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 