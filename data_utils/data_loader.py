import numpy as np
import pickle
import networkx as nx
import scipy.sparse as sp
from pathlib import Path
import os

class CoraDataLoader:
    def __init__(self, data_path=None):
        """
        Initialize the data loader
        
        Args:
            data_path (str): Path to the data directory. If None, will try to find it automatically.
        """
        if data_path is None:
            # Try to find the data directory
            current_dir = Path.cwd()
            possible_paths = [
                current_dir / "real-node-label" / "cora",
                current_dir / "Network-Science-HW3" / "real-node-label" / "cora",
                current_dir.parent / "real-node-label" / "cora"
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.data_path = path
                    print(f"Found data directory at: {path}")
                    break
            else:
                raise FileNotFoundError(
                    "Could not find data directory. Tried: \n" + 
                    "\n".join(str(p) for p in possible_paths)
                )
        else:
            self.data_path = Path(data_path)
            
        # Verify all required files exist
        required_files = [
            "ind.cora.x", "ind.cora.y", "ind.cora.tx", "ind.cora.ty",
            "ind.cora.allx", "ind.cora.ally", "ind.cora.graph", "ind.cora.test.index"
        ]
        
        missing_files = [
            f for f in required_files 
            if not (self.data_path / f).exists()
        ]
        
        if missing_files:
            raise FileNotFoundError(
                "Missing required files: \n" + 
                "\n".join(missing_files)
            )
            
        print("All required files found!")

    def load_data(self):
        """Load Cora dataset"""
        print("Loading Cora dataset...")
        
        try:
            # Load all parts of the dataset
            x = self._load_binary(self.data_path / "ind.cora.x")     # Training features
            tx = self._load_binary(self.data_path / "ind.cora.tx")   # Test features
            allx = self._load_binary(self.data_path / "ind.cora.allx") # All features except test

            y = self._load_binary(self.data_path / "ind.cora.y")     # Training labels
            ty = self._load_binary(self.data_path / "ind.cora.ty")   # Test labels
            ally = self._load_binary(self.data_path / "ind.cora.ally") # All labels except test

            # Load graph
            graph = self._load_binary(self.data_path / "ind.cora.graph")

            # Load test indices
            test_idx_reorder = np.loadtxt(self.data_path / "ind.cora.test.index", dtype=np.int32)
            test_idx_range = np.sort(test_idx_reorder)

            # Combine features
            if sp.issparse(allx):
                features = sp.vstack((allx, tx)).toarray()
            else:
                features = np.vstack((allx, tx))

            # Combine labels
            labels = np.vstack((ally, ty))
            
            # Convert one-hot encoded labels to class indices if needed
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels = np.argmax(labels, axis=1)

            # Fix the ordering of test instances
            idx_test = test_idx_range.tolist()
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+500)

            # Create graph
            G = nx.from_dict_of_lists(graph)

            # Verify data consistency
            print("\nData Consistency Check:")
            print(f"Number of nodes in graph: {G.number_of_nodes()}")
            print(f"Number of feature vectors: {features.shape[0]}")
            print(f"Number of labels: {len(labels)}")
            print(f"Feature dimension: {features.shape[1]}")
            print(f"Number of classes: {len(np.unique(labels))}")
            print(f"Train/Val/Test split: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

            # Additional verification
            if G.number_of_nodes() != features.shape[0]:
                raise ValueError(f"Number of nodes ({G.number_of_nodes()}) doesn't match number of feature vectors ({features.shape[0]})")
            
            if G.number_of_nodes() != len(labels):
                raise ValueError(f"Number of nodes ({G.number_of_nodes()}) doesn't match number of labels ({len(labels)})")

            return {
                'graph': G,
                'features': features,
                'labels': labels,
                'idx_train': idx_train,
                'idx_val': idx_val,
                'idx_test': idx_test
            }

        except Exception as e:
            print(f"\nERROR during data loading: {str(e)}")
            print("\nDetailed error information:")
            import traceback
            traceback.print_exc()
            raise

    def _load_binary(self, file_path):
        """Load binary files"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                print(f"Successfully loaded {file_path}")
                return data
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise 