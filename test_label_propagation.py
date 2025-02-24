from data_utils.data_loader import CoraDataLoader
from models.label_propagation import run_label_propagation_experiment
import os
import json
from datetime import datetime

def main():
    print("Loading Cora dataset...")
    try:
        loader = CoraDataLoader()
        data = loader.load_data()
        print("Data loaded successfully!")
        
        # Run experiment
        results = run_label_propagation_experiment(data)
        
        # Save results
        print("\nExperiment completed!")
        print("Summary:")
        print(f"Best Configuration Results:")
        
        # Display parameters based on kernel type
        print(f"Kernel: {results['kernel']}")
        if results['kernel'] == 'rbf' and 'gamma' in results:
            print(f"Gamma: {results['gamma']:.4f}")
        if results['kernel'] == 'knn' and 'n_neighbors' in results:
            print(f"N Neighbors: {results['n_neighbors']}")
            
        # Display performance metrics
        print("\nPerformance Metrics:")
        print(f"Training: Accuracy={results['train_acc']:.4f}, F1={results['train_f1']:.4f}")
        print(f"Validation: Accuracy={results['val_acc']:.4f}, F1={results['val_f1']:.4f}")
        print(f"Test: Accuracy={results['test_acc']:.4f}, F1={results['test_f1']:.4f}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Clean results for saving
        save_results = {k: v for k, v in results.items() 
                       if v is not None and k != 'predictions'}
        
        results_file = os.path.join(results_dir, f"label_propagation_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=4)
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nDebug information:")
        print(f"Current working directory: {os.getcwd()}")
        print("Directory contents:")
        os.system("ls -R")

if __name__ == "__main__":
    main() 