# Node Classification and Link Prediction Tasks Progress Tracker

## Project Overview
1. Compare performance of multiple node classification algorithms on citation network datasets
2. Compare performance of link prediction algorithms on the same datasets
3. Optional: Implement recent algorithms (bonus)
4. Optional: Include ogbn-arxiv dataset (bonus)

## Datasets to Process
- [ ] Citeseer
- [ ] Cora
- [ ] PubMed
- [ ] (Bonus) ogbn-arxiv

## Required Steps

### 1. Data Preparation
- [ ] Download and load datasets
  - [ ] Citeseer dataset
  - [ ] Cora dataset
  - [ ] PubMed dataset
  - [ ] (Bonus) ogbn-arxiv dataset
- [ ] Implement train/test split for node classification
  - [ ] 20 nodes per class for training
  - [ ] 500 nodes for validation
  - [ ] 1000 nodes for testing
- [ ] Implement edge splitting for link prediction
  - [ ] Randomly remove 20% of edges
  - [ ] Ensure graph remains connected
  - [ ] Create negative edge samples

### 2. Node Classification Algorithm Implementation (Task 1)
Need to implement at least 2 node classification algorithms:
- [ ] Algorithm 1 (e.g., GCN)
  - [ ] Implementation
  - [ ] Testing
  - [ ] Validation
- [ ] Algorithm 2 (e.g., GraphSAGE)
  - [ ] Implementation
  - [ ] Testing
  - [ ] Validation
- [ ] (Bonus) Recent Algorithm (last 5 years)
  - [ ] Implementation
  - [ ] Testing
  - [ ] Validation

### 3. Link Prediction Algorithm Implementation (Task 2)
Need to implement at least 2 link prediction algorithms:
- [ ] Algorithm 1 (e.g., Node2Vec)
  - [ ] Implementation
  - [ ] Testing
  - [ ] Validation
- [ ] Algorithm 2 (e.g., GAE)
  - [ ] Implementation
  - [ ] Testing
  - [ ] Validation
- [ ] (Bonus) Recent Algorithm (last 5 years)
  - [ ] Implementation
  - [ ] Testing
  - [ ] Validation

### 4. Experimental Setup
- [ ] Node Classification Setup
  - [ ] Set up 10-fold cross-validation framework
  - [ ] Implement accuracy metrics
  - [ ] Create logging mechanism
- [ ] Link Prediction Setup
  - [ ] Edge splitting mechanism
  - [ ] AUC calculation
  - [ ] Negative sampling strategy

### 5. Experiments to Run
For each dataset:
- [ ] Node Classification Task
  - [ ] Run Algorithm 1 (10 runs)
    - [ ] Calculate average accuracy
    - [ ] Calculate variance
  - [ ] Run Algorithm 2 (10 runs)
    - [ ] Calculate average accuracy
    - [ ] Calculate variance
  - [ ] (Bonus) Run Recent Algorithm

- [ ] Link Prediction Task
  - [ ] Run Algorithm 1 (10 runs)
    - [ ] Calculate average AUC
    - [ ] Calculate variance
  - [ ] Run Algorithm 2 (10 runs)
    - [ ] Calculate average AUC
    - [ ] Calculate variance
  - [ ] (Bonus) Run Recent Algorithm

### 6. Results Analysis
- [ ] Node Classification Results
  - [ ] Compile accuracy results
  - [ ] Calculate variances
  - [ ] Create comparison tables/graphs
- [ ] Link Prediction Results
  - [ ] Compile AUC results
  - [ ] Calculate variances
  - [ ] Create comparison tables/graphs
- [ ] (Bonus) Analysis with ogbn-arxiv dataset
- [ ] (Bonus) Analysis of recent algorithms

### 7. Documentation
- [ ] Document methodology for both tasks
- [ ] Document implementation details
- [ ] Create results summary
- [ ] Write conclusions and comparisons
- [ ] Document bonus implementations (if applicable)

## Technical Requirements
- Python environment setup
- Required libraries:
  - [ ] PyTorch/TensorFlow
  - [ ] NetworkX
  - [ ] NumPy
  - [ ] Scikit-learn
  - [ ] Pandas
  - [ ] OGB (for ogbn-arxiv)

## Notes
- Follow GCN paper split methodology for node classification
- Random 20% edge removal for link prediction
- Can choose to use or ignore feature matrices depending on algorithm
- Need to ensure reproducibility of results
- Must maintain consistent splits across algorithms for fair comparison
- Bonus opportunities:
  - Implement algorithms from last 5 years (+10% per task)
  - Include ogbn-arxiv dataset (+5% per task) 