import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, normalize
import networkx as nx
import scipy.sparse as sp
import warnings
from sklearn.exceptions import ConvergenceWarning

class LabelPropagationModel:
    def __init__(self, kernel='rbf', gamma=None, n_neighbors=None, max_iter=3000):
        """
        Initialize Label Propagation model
        
        Args:
            kernel (str): Kernel type ('rbf' or 'knn')
            gamma (float, optional): Kernel coefficient for rbf kernel
            n_neighbors (int, optional): Number of neighbors for knn kernel
            max_iter (int): Maximum number of iterations
        """
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        
    def fit_predict(self, data):
        """
        Fit the model and predict labels
        
        Args:
            data (dict): Dictionary containing:
                - features: node features matrix
                - labels: node labels
                - idx_train: training set indices
                - idx_val: validation set indices
                - idx_test: test set indices
        
        Returns:
            dict: Dictionary containing performance metrics
        """
        features = data['features']
        labels = data['labels']
        idx_train = data['idx_train']
        idx_val = data['idx_val']
        idx_test = data['idx_test']
        
        # Preprocess features
        features_scaled = self.scaler.fit_transform(features)
        # Normalize features to unit length
        features_normalized = normalize(features_scaled, norm='l2')
        
        # Create mask for unlabeled data
        mask = np.ones(len(labels), dtype=bool)
        mask[idx_train] = False
        
        # Prepare labels for semi-supervised learning
        labels_train = labels.copy()
        labels_train[mask] = -1
        
        # Set up model parameters
        model_params = {
            'kernel': self.kernel,
            'max_iter': self.max_iter,
            'tol': 1e-6,
            'n_jobs': -1
        }
        
        if self.kernel == 'rbf':
            if self.gamma is None:
                # Set gamma to 1/n_features if not specified
                model_params['gamma'] = 1.0 / features.shape[1]
            else:
                model_params['gamma'] = self.gamma
        elif self.kernel == 'knn':
            if self.n_neighbors is None:
                # Set n_neighbors to sqrt(n_samples) if not specified
                model_params['n_neighbors'] = int(np.sqrt(features.shape[0]))
            else:
                model_params['n_neighbors'] = self.n_neighbors
        
        print(f"Fitting Label Propagation model with params: {model_params}")
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                
                model = LabelPropagation(**model_params)
                model.fit(features_normalized, labels_train)
                predictions = model.predict(features_normalized)
                
                if len(w) > 0:
                    print("Warning: Model did not converge. Trying with more iterations...")
                    model_params['max_iter'] = self.max_iter * 2
                    model = LabelPropagation(**model_params)
                    model.fit(features_normalized, labels_train)
                    predictions = model.predict(features_normalized)
            
            # Calculate metrics
            train_acc = accuracy_score(labels[idx_train], predictions[idx_train])
            val_acc = accuracy_score(labels[idx_val], predictions[idx_val])
            test_acc = accuracy_score(labels[idx_test], predictions[idx_test])
            
            # Calculate F1 scores
            train_f1 = f1_score(labels[idx_train], predictions[idx_train], average='macro')
            val_f1 = f1_score(labels[idx_val], predictions[idx_val], average='macro')
            test_f1 = f1_score(labels[idx_test], predictions[idx_test], average='macro')
            
            print(f"Accuracies: Train={train_acc:.4f}, Val={val_acc:.4f}, Test={test_acc:.4f}")
            print(f"F1 Scores: Train={train_f1:.4f}, Val={val_f1:.4f}, Test={test_f1:.4f}")
            
            return {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'test_f1': test_f1,
                'predictions': predictions,
                'converged': model.n_iter_ < model.max_iter
            }
            
        except Exception as e:
            print(f"Error during model fitting: {str(e)}")
            return None

def run_label_propagation_experiment(data):
    """Run Label Propagation experiment with different parameters"""
    results = []
    
    # Try different parameters
    parameter_grid = [
        {'kernel': 'rbf', 'gamma': 0.0001},
        {'kernel': 'rbf', 'gamma': 0.001},
        {'kernel': 'rbf', 'gamma': 0.01},
        {'kernel': 'knn', 'n_neighbors': int(np.sqrt(data['features'].shape[0]))},  # sqrt(n)
        {'kernel': 'knn', 'n_neighbors': 10},
        {'kernel': 'knn', 'n_neighbors': 20}
    ]
    
    for params in parameter_grid:
        print(f"\nTrying parameters: {params}")
        model = LabelPropagationModel(**params)
        run_results = model.fit_predict(data)
        
        if run_results is not None:
            results.append({
                **params,
                'train_acc': run_results['train_acc'],
                'val_acc': run_results['val_acc'],
                'test_acc': run_results['test_acc'],
                'train_f1': run_results['train_f1'],
                'val_f1': run_results['val_f1'],
                'test_f1': run_results['test_f1'],
                'converged': run_results['converged']
            })
    
    if not results:
        raise Exception("No successful experiments")
    
    # Find best configuration based on validation F1 score
    best_result = max(results, key=lambda x: x['val_f1'])
    
    print("\nBest configuration:")
    for k, v in best_result.items():
        if v is not None and k != 'predictions':
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
    
    return best_result 