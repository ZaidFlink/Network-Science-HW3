from data_utils.data_loader import CoraDataLoader
import os
import numpy as np

def test_cora_loader():
    # Initialize the data loader
    loader = CoraDataLoader()
    
    # Check if files exist
    base_path = "real-node-label/cora"
    required_files = [
        "ind.cora.x", "ind.cora.y", "ind.cora.graph", "ind.cora.test.index",
        "ind.cora.tx", "ind.cora.ty", "ind.cora.allx", "ind.cora.ally"
    ]
    
    print("Checking for required files...")
    for file in required_files:
        path = os.path.join(base_path, file)
        if os.path.exists(path):
            print(f"✓ Found {file}")
        else:
            print(f"✗ Missing {file}")
    
    print("\nAttempting to load data...")
    try:
        data = loader.load_data()
        
        # Print detailed information about the loaded data
        print("\nDetailed Data Information:")
        print(f"Graph info:")
        print(f"  - Number of nodes: {data['graph'].number_of_nodes()}")
        print(f"  - Number of edges: {data['graph'].number_of_edges()}")
        
        print(f"\nFeatures info:")
        print(f"  - Shape: {data['features'].shape}")
        print(f"  - Type: {type(data['features'])}")
        print(f"  - Number of features: {data['features'].shape[1]}")
        
        print(f"\nLabels info:")
        print(f"  - Shape: {data['labels'].shape}")
        print(f"  - Unique labels: {np.unique(data['labels'])}")
        print(f"  - Number of classes: {len(np.unique(data['labels']))}")
        
        print(f"\nSplit info:")
        print(f"  - Training nodes: {len(data['idx_train'])}")
        print(f"  - Validation nodes: {len(data['idx_val'])}")
        print(f"  - Test nodes: {len(data['idx_test'])}")
        
        # Verify the splits are correct according to the GCN paper
        assert len(data['idx_train']) == 140, "Training set should have 140 nodes"
        assert len(data['idx_val']) == 500, "Validation set should have 500 nodes"
        assert len(data['idx_test']) == 1000, "Test set should have 1000 nodes"
        
        print("\n✓ Data loading successful!")
        print("✓ All consistency checks passed!")
        
    except Exception as e:
        print(f"\n✗ Error loading data: {str(e)}")
        raise e  # This will show the full error traceback

if __name__ == "__main__":
    test_cora_loader() 