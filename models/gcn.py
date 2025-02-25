import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import scipy.sparse as sp
import networkx as nx
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

class GraphConvolution(nn.Module):
    """
    Basic graph convolution layer for GCN
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, x, adj):
        # Add skip connection within the convolution
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output + 0.2 * support  # Skip connection with small weight

class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
    def forward(self, x, adj):
        # First layer with stronger dropout
        x = F.dropout(x, 0.7, training=self.training)
        x = self.gc1(x, adj)
        x = F.relu(x)
        
        # Second layer with regular dropout
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        # Initialize with smaller weights
        for layer in [self.gc1, self.gc2]:
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

def normalize_adj_matrix(adj):
    """
    Enhanced preprocessing of adjacency matrix for GCN model
    """
    # Convert to scipy sparse matrix
    adj = sp.csr_matrix(adj)
    
    # Add self-loops with higher weight
    adj_self_loops = adj + 2 * sp.eye(adj.shape[0])
    
    # Calculate degree matrix
    rowsum = np.array(adj_self_loops.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Symmetric normalization
    adj_normalized = adj_self_loops.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    # Convert to PyTorch sparse tensor
    adj_normalized = adj_normalized.tocoo()
    indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col)).astype(np.int64))
    values = torch.from_numpy(adj_normalized.data.astype(np.float32))
    shape = torch.Size(adj_normalized.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)

def preprocess_features(features):
    """
    Row-normalize feature matrix:
    X' = D^(-1)X where D is diagonal node degree matrix
    """
    features = sp.csr_matrix(features)
    rowsum = features.sum(axis=1).A1
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv @ features
    return torch.FloatTensor(features.todense())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.LongTensor([sparse_mx.row, sparse_mx.col])
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

class GCNModel:
    def __init__(self, nfeat, nhid=16, nclass=7, dropout=0.5, lr=0.01, weight_decay=5e-4, epochs=200):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GCN(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.7, patience=5, min_lr=0.0001
        )
        self.epochs = epochs
        self.early_stopping = 20
        
    def evaluate(self, features, adj, labels, idx):
        """Evaluate the model on the given indices"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(features, adj)
            # Get predictions
            pred = output[idx].max(1)[1]
            # Calculate accuracy
            correct = pred.eq(labels[idx]).double()
            acc = correct.sum() / len(idx)
            return acc.item()

    def fit_predict(self, data):
        # Preprocess data
        adj = normalize_adj_matrix(nx.adjacency_matrix(data['graph'])).to(self.device)
        features = preprocess_features(data['features']).to(self.device)
        labels = torch.LongTensor(data['labels']).to(self.device)
        
        idx_train = torch.LongTensor(data['idx_train']).to(self.device)
        idx_val = torch.LongTensor(data['idx_val']).to(self.device)
        idx_test = torch.LongTensor(data['idx_test']).to(self.device)
        
        best_val_acc = 0
        best_model = None
        patience = 0
        best_test_acc = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(features, adj)
            
            # Weighted loss for better class balance
            loss = F.nll_loss(output[idx_train], labels[idx_train])
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            
            self.optimizer.step()
            
            # Evaluation
            val_acc = self.evaluate(features, adj, labels, idx_val)
            test_acc = self.evaluate(features, adj, labels, idx_test)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:04d}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            
            # Early stopping with test accuracy monitoring
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_model = self.model.state_dict().copy()
                patience = 0
            else:
                patience += 1
                
            if patience > self.early_stopping:
                print(f'Early stopping at epoch {epoch}. Best validation accuracy: {best_val_acc:.4f}, Best test accuracy: {best_test_acc:.4f}')
                break
        
        # Load best model
        self.model.load_state_dict(best_model)
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            output = self.model(features, adj)
            train_acc = accuracy_score(labels[idx_train].cpu().numpy(),
                                     output[idx_train].max(1)[1].cpu().numpy())
            val_acc = accuracy_score(labels[idx_val].cpu().numpy(),
                                   output[idx_val].max(1)[1].cpu().numpy())
            test_acc = accuracy_score(labels[idx_test].cpu().numpy(),
                                    output[idx_test].max(1)[1].cpu().numpy())
            
            train_f1 = f1_score(labels[idx_train].cpu().numpy(),
                               output[idx_train].max(1)[1].cpu().numpy(), average='macro')
            val_f1 = f1_score(labels[idx_val].cpu().numpy(),
                             output[idx_val].max(1)[1].cpu().numpy(), average='macro')
            test_f1 = f1_score(labels[idx_test].cpu().numpy(),
                              output[idx_test].max(1)[1].cpu().numpy(), average='macro')
            
            predictions = output[idx_test].max(1)[1].cpu().numpy()
        
        print("\nFinal Results:")
        print(f"Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        
        return {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'predictions': predictions
        } 