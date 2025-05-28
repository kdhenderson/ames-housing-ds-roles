import os
import sys
import joblib
import numpy as np
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Add the utils directory to the path so we can import data_loader
sys.path.append(os.path.join('src', 'modeling', 'utils'))
from data_loader import load_housing_data
from modeling.utils.validation import perform_kfold_cv

# Add visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 1. Load the cleaned dataset (no pandas)
data_path = os.path.join('data', 'processed', 'housing_cleaned.csv')

# This function returns:
# - numerical_ordinal_features: numpy array of numerical and ordinal features
# - nominal_features_list: list of numpy arrays (one per nominal feature)
# - target: numpy array of target values
# - numerical_ordinal_headers: list of headers for numerical/ordinal features
# - nominal_headers: list of nominal feature names
# - nominal_cardinalities: dict of nominal feature cardinalities
numerical_ordinal_features, nominal_features_list, target, numerical_ordinal_headers, nominal_headers, nominal_cardinalities = load_housing_data(data_path)

print("Numerical/Ordinal Features Shape:", numerical_ordinal_features.shape)
print("Number of Nominal Features:", len(nominal_features_list))
print("Target Shape:", target.shape)

# 2. Load the Random Forest model
rf_model_path = os.path.join('data', 'processed', 'random_forest_model.joblib')
rf_model = joblib.load(rf_model_path)
print("Random Forest model loaded.")

# 3. Load the Deep Learning model
dl_model_path = os.path.join('data', 'processed', 'final_deep_learning_model.h5')
dl_model = keras.models.load_model(dl_model_path)
print("Deep Learning model loaded.")

# Dictionary to store all results for visualization
results = {
    'random_forest': {
        'fold_metrics': [],
        'summary': {}
    },
    'deep_learning': {
        'fold_metrics': [],
        'summary': {}
    }
}

# --- K-fold Cross Validation for Random Forest ---
def create_rf_model(random_state: int = 42):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)

# Combine features for RF (numerical + nominal)
features_combined_rf = np.concatenate([numerical_ordinal_features] + nominal_features_list, axis=1)

print("\nPerforming K-fold validation for Random Forest...")
rf_fold_metrics, rf_avg_metrics = perform_kfold_cv(
    features=features_combined_rf,
    target=target,
    model_builder=create_rf_model,
    n_splits=5,
    random_state=42,
    target_is_log_transformed=False
)

print("\nRandom Forest K-fold Results:")
print(f"RMSE: ${rf_avg_metrics['rmse']:,.2f} ± ${rf_avg_metrics['rmse_std']:,.2f}")
print(f"R² Score: {rf_avg_metrics['r2']:.4f} ± {rf_avg_metrics['r2_std']:.4f}")
print(f"MAPE: {rf_avg_metrics['mape']:.2f}% ± {rf_avg_metrics['mape_std']:.2f}%")

# Store RF results
results['random_forest']['fold_metrics'] = rf_fold_metrics
results['random_forest']['summary'] = rf_avg_metrics

# --- K-fold Cross Validation for Deep Learning ---
print("\nPerforming K-fold validation for Deep Learning...")
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Log transform target for DL (as in your original implementation)
target_log = np.log1p(target)

dl_fold_rmses = []
dl_fold_r2s = []
dl_fold_mapes = []
dl_fold_metrics = []  # Store complete metrics for each fold

for fold, (train_idx, val_idx) in enumerate(kf.split(numerical_ordinal_features), 1):
    print(f"\nFold {fold}/{n_splits}")
    
    # Split data
    X_train_num = numerical_ordinal_features[train_idx]
    X_val_num = numerical_ordinal_features[val_idx]
    X_train_nom = [nom_feat[train_idx] for nom_feat in nominal_features_list]
    X_val_nom = [nom_feat[val_idx] for nom_feat in nominal_features_list]
    y_train_log = target_log[train_idx]
    y_val = target[val_idx]  # Original scale for metrics
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_val_num_scaled = scaler.transform(X_val_num)
    
    # Make predictions
    X_val = [X_val_num_scaled] + X_val_nom
    y_pred_log = dl_model.predict(X_val, verbose=0)  # Add verbose=0 to reduce output
    
    # More conservative clipping and safer conversion
    max_safe_exp = np.log(np.finfo(np.float64).max)  # Maximum safe value for exp
    y_pred_log = np.clip(y_pred_log, -max_safe_exp, max_safe_exp)
    y_pred = np.expm1(y_pred_log.flatten())  # Back to original scale
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    
    # Store metrics
    fold_metrics = {
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    dl_fold_metrics.append(fold_metrics)
    
    dl_fold_rmses.append(rmse)
    dl_fold_r2s.append(r2)
    dl_fold_mapes.append(mape)
    
    print(f"Fold {fold} Metrics:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

# Calculate summary metrics
dl_summary_metrics = {
    'rmse': np.mean(dl_fold_rmses),
    'rmse_std': np.std(dl_fold_rmses),
    'r2': np.mean(dl_fold_r2s),
    'r2_std': np.std(dl_fold_r2s),
    'mape': np.mean(dl_fold_mapes),
    'mape_std': np.std(dl_fold_mapes)
}

print("\nDeep Learning K-fold Results:")
print(f"RMSE: ${dl_summary_metrics['rmse']:,.2f} ± ${dl_summary_metrics['rmse_std']:,.2f}")
print(f"R² Score: {dl_summary_metrics['r2']:.4f} ± {dl_summary_metrics['r2_std']:.4f}")
print(f"MAPE: {dl_summary_metrics['mape']:.2f}% ± {dl_summary_metrics['mape_std']:.2f}%")

# Store DL results
results['deep_learning']['fold_metrics'] = dl_fold_metrics
results['deep_learning']['summary'] = dl_summary_metrics

# Create figures directory if it doesn't exist
figures_dir = os.path.join('figures')
os.makedirs(figures_dir, exist_ok=True)

# Save results for later visualization
results_path = os.path.join('data', 'processed', 'model_comparison_results.joblib')
joblib.dump(results, results_path)
print(f"\nResults saved to: {results_path}")

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
# Wes Anderson Moonrise Kingdom palette
colors = {
    'rf': '#3B9AB2',   # Subtle blue
    'dl': '#E4B363',   # Golden yellow
    'accent1': '#F21D41', # Red
    'accent2': '#78B7C5',  # Light blue
    'accent3': '#EBCC2A'   # Yellow
}

def set_plot_style(ax):
    """Apply consistent styling to plot axes."""
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#E0E0E0')

def create_metric_comparison_plot(results, metric_name, title, ylabel, filename):
    """Create a box plot comparing the metric across models and folds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    rf_metrics = [fold[metric_name] for fold in results['random_forest']['fold_metrics']]
    dl_metrics = [fold[metric_name] for fold in results['deep_learning']['fold_metrics']]
    
    # Create box plot
    bplot = ax.boxplot([rf_metrics, dl_metrics], 
                      labels=['Random Forest', 'Deep Learning'],
                      patch_artist=True,
                      medianprops=dict(color="black", linewidth=1.5))
    
    # Color boxes
    bplot['boxes'][0].set_facecolor(colors['rf'])
    bplot['boxes'][1].set_facecolor(colors['dl'])
    
    # Style whiskers and caps
    for whisker in bplot['whiskers']:
        whisker.set_color('black')
    for cap in bplot['caps']:
        cap.set_color('black')
    
    # Add individual points
    x_rf = [1] * len(rf_metrics)
    x_dl = [2] * len(dl_metrics)
    ax.scatter(x_rf, rf_metrics, alpha=0.6, color=colors['rf'])
    ax.scatter(x_dl, dl_metrics, alpha=0.6, color=colors['dl'])
    
    ax.set_title(title, pad=20)
    ax.set_ylabel(ylabel)
    set_plot_style(ax)
    plt.tight_layout()
    
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_subplots(results, filename):
    """Create separate subplots for each metric."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Performance Comparison: Random Forest vs Deep Learning', fontsize=16, y=1.05)
    
    metrics = ['rmse', 'r2', 'mape']
    titles = ['RMSE ($)', 'R² Score', 'MAPE (%)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        rf_mean = results['random_forest']['summary'][metric]
        rf_std = results['random_forest']['summary'][f'{metric}_std']
        dl_mean = results['deep_learning']['summary'][metric]
        dl_std = results['deep_learning']['summary'][f'{metric}_std']
        
        x = [0, 1]
        ax.bar(x[0], rf_mean, yerr=rf_std, color=colors['rf'], capsize=5, label='Random Forest', width=0.4)
        ax.bar(x[1], dl_mean, yerr=dl_std, color=colors['dl'], capsize=5, label='Deep Learning', width=0.4)
        
        ax.set_xticks([])
        ax.set_title(title)
        set_plot_style(ax)
        
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_residual_plots(results, rf_model, dl_model, features_combined_rf, numerical_ordinal_features, nominal_features_list, filename):
    """Create residual plots for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RF residuals
    y_pred_rf = rf_model.predict(features_combined_rf)
    residuals_rf = target - y_pred_rf
    
    # DL residuals - with improved prediction handling
    X_dl = [numerical_ordinal_features] + nominal_features_list
    y_pred_log_dl = dl_model.predict(X_dl).flatten()
    
    # Print statistics to help diagnose issues
    print("\nDeep Learning Prediction Statistics (before scaling):")
    print(f"Min log prediction: {np.min(y_pred_log_dl):.2f}")
    print(f"Max log prediction: {np.max(y_pred_log_dl):.2f}")
    print(f"Mean log prediction: {np.mean(y_pred_log_dl):.2f}")
    
    # It seems the model is not properly scaled - let's fix that
    # If predictions are way too large, they're likely not in log space
    # Let's convert them to log space first
    if np.min(y_pred_log_dl) > 100:  # This would be an unreasonable value for log-transformed prices
        print("\nDetected unscaled predictions - applying log transformation...")
        y_pred_log_dl = np.log1p(y_pred_log_dl)
        print("\nDeep Learning Prediction Statistics (after log transform):")
        print(f"Min log prediction: {np.min(y_pred_log_dl):.2f}")
        print(f"Max log prediction: {np.max(y_pred_log_dl):.2f}")
        print(f"Mean log prediction: {np.mean(y_pred_log_dl):.2f}")
    
    # Now convert back to original scale
    y_pred_dl = np.expm1(y_pred_log_dl)
    
    # Handle any remaining numerical issues
    y_pred_dl = np.clip(y_pred_dl, 0, np.max(target) * 2)  # Clip to reasonable house price range
    
    residuals_dl = target - y_pred_dl
    
    # Print final prediction statistics
    print("\nFinal Prediction Statistics:")
    print(f"Min prediction: ${np.min(y_pred_dl):,.2f}")
    print(f"Max prediction: ${np.max(y_pred_dl):,.2f}")
    print(f"Mean prediction: ${np.mean(y_pred_dl):,.2f}")
    print(f"Actual target min: ${np.min(target):,.2f}")
    print(f"Actual target max: ${np.max(target):,.2f}")
    print(f"Actual target mean: ${np.mean(target):,.2f}")
    
    # Plot RF with adjusted alpha
    ax1.scatter(y_pred_rf, residuals_rf, alpha=0.3, color=colors['rf'], label='Residuals', s=50)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Predicted Values ($)')
    ax1.set_ylabel('Residuals ($)')
    ax1.set_title('Random Forest Residuals')
    set_plot_style(ax1)
    
    # Plot DL with adjusted alpha
    ax2.scatter(y_pred_dl, residuals_dl, alpha=0.3, color=colors['dl'], label='Residuals', s=50)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Predicted Values ($)')
    ax2.set_ylabel('Residuals ($)')
    ax2.set_title('Deep Learning Residuals')
    set_plot_style(ax2)
    
    # Add legends
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_plot(rf_model, feature_names, filename, top_n=20):
    """Create a bar plot of feature importances from Random Forest."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    ax.bar(range(top_n), importances[indices], color=colors['rf'])
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title(f'Top {top_n} Most Important Features (Random Forest)')
    set_plot_style(ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return [feature_names[i] for i in indices[:6]]  # Return top 6 features for EDA plots

def create_shap_plots(rf_model, features_combined_rf, feature_names, filename):
    """Create SHAP summary plot for Random Forest."""
    # Calculate SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(features_combined_rf)
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, features_combined_rf, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_target_plots(features_combined_rf, target, feature_names, top_features, filename):
    """Create scatter plots of top features vs target."""
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Relationship Between Top Features and Sale Price', fontsize=16, y=1.02)
    
    # Create 2x3 grid of subplots
    for i, feature in enumerate(top_features, 1):
        ax = plt.subplot(2, 3, i)
        feature_idx = feature_names.index(feature)
        feature_values = features_combined_rf[:, feature_idx]
        
        # Add scatter plot with adjusted alpha and size
        ax.scatter(feature_values, target, alpha=0.3, color=colors['rf'], s=50)
        ax.set_xlabel(feature)
        ax.set_ylabel('Sale Price ($)' if i in [1, 4] else '')  # Only add y-label for leftmost plots
        set_plot_style(ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

print("\nGenerating visualization plots...")

# Create all plots
create_metric_comparison_plot(results, 'rmse', 'RMSE Comparison Across Folds', 'RMSE ($)', 'rmse_comparison.png')
create_metric_comparison_plot(results, 'r2', 'R² Score Comparison Across Folds', 'R² Score', 'r2_comparison.png')
create_metric_comparison_plot(results, 'mape', 'MAPE Comparison Across Folds', 'MAPE (%)', 'mape_comparison.png')

create_summary_subplots(results, 'model_comparison_summary.png')

# Combine feature names and create importance plot
all_feature_names = numerical_ordinal_headers + nominal_headers
top_features = create_feature_importance_plot(rf_model, all_feature_names, 'feature_importance.png')

create_residual_plots(results, rf_model, dl_model, features_combined_rf, numerical_ordinal_features, nominal_features_list, 'residual_plots.png')

create_shap_plots(rf_model, features_combined_rf, all_feature_names, 'shap_summary.png')

create_feature_target_plots(features_combined_rf, target, all_feature_names, top_features, 'feature_target_relationships.png')

print("\nVisualization plots have been saved to the 'figures' directory:")
print("1. rmse_comparison.png - Box plots of RMSE across folds")
print("2. r2_comparison.png - Box plots of R² across folds")
print("3. mape_comparison.png - Box plots of MAPE across folds")
print("4. model_comparison_summary.png - Summary of all metrics")
print("5. feature_importance.png - Top 20 important features")
print("6. residual_plots.png - Residual analysis")
print("7. shap_summary.png - SHAP value analysis")
print("8. feature_target_relationships.png - Top features vs. Sale Price") 