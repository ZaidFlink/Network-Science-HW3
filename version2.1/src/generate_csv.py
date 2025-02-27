import re
import os
import pandas as pd

def extract_link_prediction_data(file_path):
    """
    Extract link prediction data from the markdown file.
    Returns a dataframe with dataset, model, run, auc, mean_auc, and std_dev.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Initialize lists to store data
    data = []
    
    # Find all dataset sections
    dataset_sections = re.split(r'## Dataset: ', content)[1:]
    
    for section in dataset_sections:
        # Extract dataset name
        dataset_name = section.strip().split('\n')[0].strip()
        
        # Find all model sections in this dataset
        model_sections = re.split(r'### (\w+) Results', section)[1:]
        
        # Process each model section
        for i in range(0, len(model_sections), 2):
            if i+1 >= len(model_sections):
                break
                
            model_name = model_sections[i].strip()
            model_content = model_sections[i+1]
            
            # Extract individual run results
            run_pattern = r'\| (\d+) \| (0\.\d+) \|'
            runs = re.findall(run_pattern, model_content)
            
            for run_num, auc in runs:
                data.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'run': int(run_num),
                    'auc': float(auc)
                })
            
            # Extract summary statistics
            summary_pattern = r'\| ' + model_name + r' \| (0\.\d+) \| (0\.\d+) \|'
            summary = re.search(summary_pattern, model_content)
            
            if summary:
                mean_auc, std_dev = summary.groups()
                # Add summary row with run = 0 to indicate it's a summary
                data.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'run': 0,  # 0 indicates summary row
                    'auc': float(mean_auc),
                    'std_dev': float(std_dev)
                })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    return df

def extract_train_models_data(file_path):
    """
    Extract training models data from the markdown file.
    Returns a dataframe with dataset, model, accuracy, and variance.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Initialize list to store data
    data = []
    
    # Find all dataset sections
    dataset_sections = re.split(r'## Dataset: ', content)[1:]
    
    for section in dataset_sections:
        # Extract dataset name
        dataset_name = section.strip().split('\n')[0].strip()
        
        # Extract model data
        model_pattern = r'\| (\w+) \| (0\.\d+) \| (0\.\d+) \|'
        models = re.findall(model_pattern, section)
        
        for model_name, accuracy, variance in models:
            data.append({
                'dataset': dataset_name,
                'model': model_name,
                'accuracy': float(accuracy),
                'variance': float(variance)
            })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    return df

def main():
    # Create output directory if it doesn't exist
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(SCRIPT_DIR, '../', 'output')
    OUTPUT_DIR = os.path.join(INPUT_DIR, 'csv')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process link prediction data
    link_prediction_path = os.path.join(INPUT_DIR, 'link_prediction_output.md')
    link_df = extract_link_prediction_data(link_prediction_path)
    
    # Save all link prediction data
    link_df.to_csv(f'{OUTPUT_DIR}/link_prediction_all.csv', index=False)
    
    # Save summary link prediction data (only rows with run=0)
    link_summary_df = link_df[link_df['run'] == 0].drop(columns=['run'])
    link_summary_df.to_csv(f'{OUTPUT_DIR}/link_prediction_summary.csv', index=False)
    
    # Save individual run data (excluding summary rows)
    link_runs_df = link_df[link_df['run'] > 0]
    link_runs_df.to_csv(f'{OUTPUT_DIR}/link_prediction_runs.csv', index=False)
    
    # Process training models data
    train_models_path = os.path.join(INPUT_DIR, 'train_models_output.md')
    train_df = extract_train_models_data(train_models_path)
    
    # Save training models data
    train_df.to_csv(f'{OUTPUT_DIR}/train_models.csv', index=False)
    
    print(f"CSV files created successfully in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
