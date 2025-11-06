import glob
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
import argparse
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

from feature_extractor import DUNEEventParser, DUNEFeatureEngineer
 
class BaselineXGBoostModel:
    """
    Baseline XGBoost model for DUNE job completion time prediction
    """
    
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.feature_importance = None
        
    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str], 
                     target_col: str = 'total_wall_time') -> Tuple:
        """
        Prepare features and target for training
        """
        # Remove rows with missing target
        df_clean = df[df[target_col].notna()].copy()
        
        # Remove outliers (jobs that took too long - likely failed or stuck)
        q99 = df_clean[target_col].quantile(0.99)
        df_clean = df_clean[df_clean[target_col] <= q99]
        
        # Remove failed jobs for initial baseline
        df_clean = df_clean[df_clean['job_success'] == 1]
        
        print(f"Data after cleaning: {len(df_clean)} jobs")
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Train XGBoost model
        """
        print("Training XGBoost model...")
        
        # Define model parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = xgb.XGBRegressor(**params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'train_mape': np.mean(np.abs((y_train - train_pred) / y_train)) * 100,
            'val_mape': np.mean(np.abs((y_val - val_pred) / y_val)) * 100
        }
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== Training Results ===")
        print(f"Train MAE: {metrics['train_mae']:.2f} seconds ({metrics['train_mae']/60:.2f} minutes)")
        print(f"Val MAE: {metrics['val_mae']:.2f} seconds ({metrics['val_mae']/60:.2f} minutes)")
        print(f"Val RMSE: {metrics['val_rmse']:.2f} seconds ({metrics['val_rmse']/60:.2f} minutes)")
        print(f"Val R²: {metrics['val_r2']:.4f}")
        print(f"Val MAPE: {metrics['val_mape']:.2f}%")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def recommend_sites(self, new_job_features: pd.DataFrame, 
                       available_sites: List[str], top_k: int = 5) -> pd.DataFrame:
        """
        Recommend top-k sites for a new job
        
        Args:
            new_job_features: DataFrame with job characteristics (without site info)
            available_sites: List of available site names
            top_k: Number of sites to recommend
        
        Returns:
            DataFrame with site recommendations and predicted times
        """
        recommendations = []
        
        for site in available_sites:
            # Create a copy of features with this site
            job_with_site = new_job_features.copy()
            
            # This is a simplified version - you'd need to properly encode the site
            # and add site-specific features based on your feature engineering
            
            # Predict completion time for this site
            pred_time = self.model.predict(job_with_site)[0]
            
            recommendations.append({
                'site_name': site,
                'predicted_time_seconds': pred_time,
                'predicted_time_minutes': pred_time / 60,
                'predicted_time_hours': pred_time / 3600
            })
        
        # Sort by predicted time (fastest first)
        recs_df = pd.DataFrame(recommendations).sort_values('predicted_time_seconds')
        
        return recs_df.head(top_k)
    
    def plot_feature_importance(self, top_n: int = 20):
        """
        Plot feature importance
        """
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Plot predicted vs actual values
        """
        predictions = self.predict(X_val)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(y_val / 3600, predictions / 3600, alpha=0.5)
        axes[0].plot([y_val.min()/3600, y_val.max()/3600], 
                     [y_val.min()/3600, y_val.max()/3600], 
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Time (hours)')
        axes[0].set_ylabel('Predicted Time (hours)')
        axes[0].set_title('Predicted vs Actual Job Completion Time')
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = (y_val - predictions) / 3600
        axes[1].scatter(predictions / 3600, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Time (hours)')
        axes[1].set_ylabel('Residual (hours)')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path: str = 'baseline_xgboost_model.pkl'):
        """
        Save trained model
        """
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = 'baseline_xgboost_model.pkl'):
        """
        Load trained model
        """
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.feature_cols = saved_data['feature_cols']
        self.feature_importance = saved_data['feature_importance']
        print(f"Model loaded from {path}")

def resolve_json_list(input_arg: str):
    """
    Accepts:
      - a directory (reads *.json in it),
      - a single .json file,
      - or a glob pattern (e.g., path/to/*.json).
    Returns a sorted list of json file paths.
    """
    if os.path.isdir(input_arg):
        pattern = os.path.join(input_arg, "*.json")
        files = glob.glob(pattern)
    elif input_arg.endswith(".json"):
        files = glob.glob(input_arg)  # allow something like dir/file*.json too
        if not files:
            files = [input_arg] if os.path.exists(input_arg) else []
    else:
        # treat anything else as a glob pattern
        files = glob.glob(input_arg)

    return sorted(f for f in files if f.lower().endswith(".json"))


def main():
    """
    Complete baseline model training pipeline
    """
    parser = argparse.ArgumentParser(description="Training AI agent.")
    parser.add_argument('--input', type=str, default="test.json", help="Model")
    parser.add_argument('--output', type=str, default="test.pkl", help="Model")
    args = parser.parse_args()
 
    base, _ = os.path.splitext(args.input)
    json_path = base +'.json'
    
    parser = DUNEEventParser()

    # Step 1: Parse JSON data
    print("=" * 60)
    print("STEP 1: Parsing JSON Event Data")
    print("=" * 60)

    json_files = resolve_json_list(args.input)
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found for input: {args.input}"
        )

    print(f"Found {len(json_files)} JSON file(s).")
    for p in json_files:
        print(f"  - {p}")

    dfs = []
    for jp in json_files:
        try:
            df = parser.load_and_process_workflow(jp)
            if df is None or (hasattr(df, "empty") and df.empty):
                print(f"[WARN] No rows parsed from: {jp}")
                continue
            df["_source_json"] = os.path.basename(jp)  # optional provenance
            dfs.append(df)
            print(f"[OK] Parsed {len(df)} rows from {jp}")
        except Exception as e:
            print(f"[ERROR] Failed to parse {jp}: {e}")

    if not dfs:
        raise RuntimeError("No valid dataframes parsed from any JSON input.")

    # Union of columns across files; differing schemas are handled by concat
    jobs_df = pd.concat(dfs, ignore_index=True, sort=False)

    print(f"\nData shape (combined): {jobs_df.shape}")
    print(f"Columns ({len(jobs_df.columns)}): {jobs_df.columns.tolist()}")
    print("\nSample data:")
    print(jobs_df.head())
    
    # Step 2: Feature Engineering
    print("\n" + "="*60)
    print("STEP 2: Feature Engineering")
    print("="*60)
    
    feature_engineer = DUNEFeatureEngineer()
    jobs_df, feature_cols = feature_engineer.fit_transform(jobs_df)
    
    print(f"\nFeatures created: {len(feature_cols)}")
    print(f"Feature list: {feature_cols}")
    
    # Step 3: Train Model
    print("\n" + "="*60)
    print("STEP 3: Training Baseline XGBoost Model")
    print("="*60)
    
    model = BaselineXGBoostModel()
    model.feature_cols = feature_cols
    
    X_train, X_val, y_train, y_val = model.prepare_data(
        jobs_df, feature_cols, target_col='total_wall_time'
    )
    
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    # Step 4: Visualizations
    print("\n" + "="*60)
    print("STEP 4: Generating Visualizations")
    print("="*60)
    
    model.plot_feature_importance(top_n=20)
    model.plot_predictions(X_val, y_val)
    
    # Step 5: Save Model
    print("\n" + "="*60)
    print("STEP 5: Saving Model")
    print("="*60)
    
    model.save_model(args.output)
    
    # Step 6: Example Prediction
    print("\n" + "="*60)
    print("STEP 6: Example Site Recommendation")
    print("="*60)
    
    # Get unique sites from data
    available_sites = jobs_df['site_name'].dropna().unique()[:10]
    print(f"\nAvailable sites: {list(available_sites)}")
    
    # Take a sample job and get recommendations
    sample_job = X_val.iloc[0:1]
    
    # This is simplified - in practice you'd need to properly handle the site encoding
    print("\nTop 5 recommended sites for sample job:")
    # recommendations = model.recommend_sites(sample_job, available_sites, top_k=5)
    # print(recommendations)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    return model, feature_engineer, jobs_df, metrics

if __name__ == "__main__":
    model, feature_engineer, jobs_df, metrics = main()
