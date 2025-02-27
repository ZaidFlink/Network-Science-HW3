import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style parameters
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set up directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '../', 'output')
CSV_DIR = os.path.join(INPUT_DIR, 'csv')
PLOT_DIR = os.path.join(INPUT_DIR, 'plots')

# Create output directory for plots
os.makedirs(PLOT_DIR, exist_ok=True)

# Load CSV files
node_class_data = pd.read_csv(os.path.join(CSV_DIR, 'train_models.csv'))
link_pred_summary = pd.read_csv(os.path.join(CSV_DIR, 'link_prediction_summary.csv'))
link_pred_runs = pd.read_csv(os.path.join(CSV_DIR, 'link_prediction_runs.csv'))

def plot_node_classification_accuracy():
    """
    Plot 1: Node Classification Accuracy Comparison as a grouped bar chart
    """
    plt.figure(figsize=(12, 7))
    
    # Create the grouped bar chart
    ax = sns.barplot(
        x='dataset', 
        y='accuracy', 
        hue='model', 
        data=node_class_data,
        palette="colorblind"
    )
    
    # Customize the plot
    plt.title('Node Classification: Accuracy Comparison Across Datasets', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Accuracy Score', fontsize=14)
    plt.ylim(0, 1.0)  # Set y-axis limits between 0 and 1 for accuracy
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.legend(title='Model', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'node_classification_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Node classification accuracy plot created.")

def plot_link_prediction_auc():
    """
    Plot 2: Link Prediction AUC Scores as a grouped bar chart
    """
    plt.figure(figsize=(14, 7))
    
    # Create the grouped bar chart
    ax = sns.barplot(
        x='dataset', 
        y='auc', 
        hue='model', 
        data=link_pred_summary,
        palette="colorblind"
    )
    
    # Customize the plot
    plt.title('Link Prediction: AUC Scores Comparison Across Datasets', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('AUC Score', fontsize=14)
    plt.ylim(0.8, 1.0)  # Set y-axis limits to focus on the range of our data
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.legend(title='Model', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'link_prediction_auc.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Link prediction AUC scores plot created.")

def plot_link_prediction_boxplot():
    """
    Plot 3: Link Prediction AUC Score Distribution as box plots
    """
    # Create separate plots for each dataset to avoid crowding
    for dataset in link_pred_runs['dataset'].unique():
        plt.figure(figsize=(14, 8))
        
        # Filter data for the current dataset
        dataset_data = link_pred_runs[link_pred_runs['dataset'] == dataset]
        
        # Create the box plot
        ax = sns.boxplot(
            x='model', 
            y='auc', 
            data=dataset_data,
            palette="colorblind"
        )
        
        # Add individual data points
        sns.swarmplot(
            x='model', 
            y='auc', 
            data=dataset_data, 
            color='.25',
            size=7,
            alpha=0.5
        )
        
        # Customize the plot
        plt.title(f'Link Prediction: AUC Score Distribution for {dataset.capitalize()}', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('AUC Score', fontsize=14)
        
        # Set y-axis limits based on data range to better show differences
        min_val = dataset_data['auc'].min() - 0.01
        max_val = dataset_data['auc'].max() + 0.01
        plt.ylim(min_val, max_val)
        
        # Add a horizontal line at the mean for each model
        for i, model in enumerate(dataset_data['model'].unique()):
            model_mean = dataset_data[dataset_data['model'] == model]['auc'].mean()
            ax.text(i, model_mean + 0.002, f'Mean: {model_mean:.3f}', 
                    horizontalalignment='center', size='medium', color='darkblue', weight='semibold')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(PLOT_DIR, f'link_prediction_boxplot_{dataset}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Link prediction AUC distribution box plots created.")

def create_combined_comparison_plot():
    """
    Creates a combined plot showing both node classification accuracy and link prediction AUC
    on the same datasets for a comprehensive comparison
    """
    plt.figure(figsize=(15, 10))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot node classification on first subplot
    sns.barplot(
        x='dataset', 
        y='accuracy', 
        hue='model', 
        data=node_class_data,
        palette="colorblind",
        ax=ax1
    )
    
    # Customize first subplot
    ax1.set_title('Node Classification: Accuracy Comparison', fontsize=16)
    ax1.set_xlabel('Dataset', fontsize=14)
    ax1.set_ylabel('Accuracy Score', fontsize=14)
    ax1.set_ylim(0, 1.0)
    ax1.legend(title='Model', fontsize=12, title_fontsize=14)
    
    # Plot link prediction on second subplot
    sns.barplot(
        x='dataset', 
        y='auc', 
        hue='model', 
        data=link_pred_summary,
        palette="colorblind",
        ax=ax2
    )
    
    # Customize second subplot
    ax2.set_title('Link Prediction: AUC Scores Comparison', fontsize=16)
    ax2.set_xlabel('Dataset', fontsize=14)
    ax2.set_ylabel('AUC Score', fontsize=14)
    ax2.set_ylim(0.8, 1.0)
    ax2.legend(title='Model', fontsize=12, title_fontsize=14)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'combined_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Combined comparison plot created.")

def plot_model_performance_by_dataset():
    """
    Creates radar charts showing model performance across all datasets
    """
    # Prepare data for both node classification and link prediction
    node_class_pivot = node_class_data.pivot(index='model', columns='dataset', values='accuracy')
    link_pred_pivot = link_pred_summary.pivot(index='model', columns='dataset', values='auc')
    
    # Create radar chart for node classification
    create_radar_chart(node_class_pivot, 'Node Classification Accuracy by Dataset', 'node_classification_radar')
    
    # Create radar chart for link prediction
    create_radar_chart(link_pred_pivot, 'Link Prediction AUC by Dataset', 'link_prediction_radar')

def create_radar_chart(pivot_data, title, filename):
    """
    Helper function to create radar charts
    """
    # Set data
    categories = pivot_data.columns
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one line per model and fill area
    for i, model in enumerate(pivot_data.index):
        values = pivot_data.loc[model].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw y-axis labels (min to max values with step)
    if 'Accuracy' in title:
        plt.yticks(np.arange(0, 1.1, 0.2), size=10)
        plt.ylim(0, 1)
    else:  # AUC values
        plt.yticks(np.arange(0.8, 1.01, 0.05), size=10)
        plt.ylim(0.8, 1)
    
    # Add title and legend
    plt.title(title, size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{title} radar chart created.")

def main():
    # Generate all plots
    plot_node_classification_accuracy()
    plot_link_prediction_auc()
    plot_link_prediction_boxplot()
    create_combined_comparison_plot()
    plot_model_performance_by_dataset()
    
    print(f"All plots have been saved to: {PLOT_DIR}")
    
    # Note about missing plots
    print("\nNote: The following plots could not be generated with the current data:")
    print("1. Model Performance Over Training Epochs - Requires epoch-by-epoch data")
    print("2. Feature Importance Analysis - Requires feature importance data from the models")
    print("\nTo generate these plots, you would need to modify your model training code to:")
    print("- Save AUC scores after each epoch during training")
    print("- Extract feature importance weights from trained models")

if __name__ == "__main__":
    main() 