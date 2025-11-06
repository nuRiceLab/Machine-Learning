import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

class DUNEEventParser:
    """
    Parser for DUNE workflow JSON event data
    """
    
    def __init__(self):
        self.event_types = [
            "JOB_SUBMITTED", "JOB_STARTED", "FILE_ALLOCATED", 
            "JOB_PROCESSING", "FILE_PROCESSED", "JOB_OUTPUTTING",
            "FILE_CREATED", "JOB_FINISHED"
        ]
    
    def load_json_data(self, json_path: str) -> Dict:
        """
        Load JSON file containing workflow events
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    def parse_events_to_dataframe(self, json_data: Dict) -> pd.DataFrame:
        """
        Convert nested JSON events into a flat DataFrame
        """
        all_events = []
        
        for event_type in self.event_types:
            if event_type in json_data:
                events = json_data[event_type]
                for event in events:
                    event['event_type'] = event_type
                    all_events.append(event)
        
        df = pd.DataFrame(all_events)
        
        # Convert event_time to datetime
        df['event_time'] = pd.to_datetime(df['event_time'])
        
        return df
    
    def aggregate_job_timeline(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate events by jobsub_id to create complete job timeline
        """
        jobs = []
        
        # Group by jobsub_id
        for jobsub_id, group in events_df.groupby('jobsub_id'):
            job_data = {
                'jobsub_id': jobsub_id,
                'workflow_id': group['workflow_id'].iloc[0],
                'stage_id': group['stage_id'].iloc[0],
            }
            
            # Extract timestamps for each event type
            for event_type in self.event_types:
                event_rows = group[group['event_type'] == event_type]
                if not event_rows.empty:
                    job_data[f'{event_type.lower()}_time'] = event_rows['event_time'].min()
                    
                    # Extract site and entry info from relevant events
                    if event_type in ['FILE_PROCESSED', 'JOB_FINISHED']:
                        job_data['site_name'] = event_rows['site_name'].iloc[0]
                        job_data['entry_name'] = event_rows['entry_name'].iloc[0]
                    
                    # Get exit code from JOB_FINISHED
                    if event_type == 'JOB_FINISHED':
                        job_data['jobscript_exit'] = event_rows['jobscript_exit'].iloc[0]
            
            # Count number of files processed
            file_processed_events = group[group['event_type'] == 'FILE_PROCESSED']
            job_data['files_processed_count'] = len(file_processed_events)
            
            # Count number of files created
            file_created_events = group[group['event_type'] == 'FILE_CREATED']
            job_data['files_created_count'] = len(file_created_events)
            
            jobs.append(job_data)
        
        jobs_df = pd.DataFrame(jobs)
        return jobs_df
    
    def compute_job_metrics(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute timing metrics and performance indicators
        """
        df = jobs_df.copy()
        
        # Time differences (in seconds)
        if 'job_submitted_time' in df.columns and 'job_started_time' in df.columns:
            df['queue_time'] = (df['job_started_time'] - df['job_submitted_time']).dt.total_seconds()
        
        if 'job_started_time' in df.columns and 'job_processing_time' in df.columns:
            df['startup_time'] = (df['job_processing_time'] - df['job_started_time']).dt.total_seconds()
        
        if 'job_processing_time' in df.columns and 'file_processed_time' in df.columns:
            df['processing_duration'] = (df['file_processed_time'] - df['job_processing_time']).dt.total_seconds()
        
        if 'file_processed_time' in df.columns and 'job_outputting_time' in df.columns:
            df['output_prep_time'] = (df['job_outputting_time'] - df['file_processed_time']).dt.total_seconds()
        
        if 'job_outputting_time' in df.columns and 'job_finished_time' in df.columns:
            df['output_transfer_time'] = (df['job_finished_time'] - df['job_outputting_time']).dt.total_seconds()
        
        # Total wall time
        if 'job_submitted_time' in df.columns and 'job_finished_time' in df.columns:
            df['total_wall_time'] = (df['job_finished_time'] - df['job_submitted_time']).dt.total_seconds()
        
        # Total execution time (after starting)
        if 'job_started_time' in df.columns and 'job_finished_time' in df.columns:
            df['execution_time'] = (df['job_finished_time'] - df['job_started_time']).dt.total_seconds()
        
        # Success indicator
        df['job_success'] = (df['jobscript_exit'] == 0).astype(int)
        
        # Processing efficiency (files per second)
        if 'processing_duration' in df.columns:
            df['processing_efficiency'] = df['files_processed_count'] / (df['processing_duration'] + 1)
        
        return df
    
    def load_and_process_workflow(self, json_path: str) -> pd.DataFrame:
        """
        Complete pipeline: load JSON -> parse events -> aggregate -> compute metrics
        """
        print(f"Loading data from {json_path}")
        json_data = self.load_json_data(json_path)
        
        print("Parsing events...")
        events_df = self.parse_events_to_dataframe(json_data)
        print(f"  Total events: {len(events_df)}")
        
        print("Aggregating job timelines...")
        jobs_df = self.aggregate_job_timeline(events_df)
        print(f"  Total jobs: {len(jobs_df)}")
        
        print("Computing metrics...")
        jobs_df = self.compute_job_metrics(jobs_df)
        
        return jobs_df


# feature_engineering/dune_features.py

class DUNEFeatureEngineer:
    """
    Feature engineering specifically for DUNE workflow data
    """
    
    def __init__(self):
        self.site_encoders = {}
        self.entry_encoders = {}
        self.site_stats = None
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from submission time
        """
        df = df.copy()
        
        if 'job_submitted_time' in df.columns:
            df['submit_hour'] = df['job_submitted_time'].dt.hour
            df['submit_day_of_week'] = df['job_submitted_time'].dt.dayofweek
            df['submit_day_of_month'] = df['job_submitted_time'].dt.day
            df['submit_month'] = df['job_submitted_time'].dt.month
            df['is_weekend'] = df['submit_day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hours'] = df['submit_hour'].between(9, 17).astype(int)
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['submit_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['submit_hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['submit_day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['submit_day_of_week'] / 7)
        
        return df
    
    def extract_site_features(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """
        Extract site-level aggregated features
        """
        df = df.copy()
        
        if training:
            # Compute site statistics on training data
            site_stats = df.groupby('site_name').agg({
                'total_wall_time': ['mean', 'std', 'median', 'min', 'max'],
                'execution_time': ['mean', 'std', 'median'],
                'queue_time': ['mean', 'std', 'median'],
                'processing_duration': ['mean', 'std'],
                'job_success': ['mean', 'count'],
                'processing_efficiency': ['mean', 'std']
            })
            
            # Flatten column names
            site_stats.columns = ['_'.join(col).strip() for col in site_stats.columns.values]
            site_stats = site_stats.reset_index()
            
            # Rename for clarity
            site_stats = site_stats.rename(columns={
                'total_wall_time_mean': 'site_avg_wall_time',
                'total_wall_time_std': 'site_std_wall_time',
                'total_wall_time_median': 'site_med_wall_time',
                'total_wall_time_min': 'site_min_wall_time',
                'total_wall_time_max': 'site_max_wall_time',
                'execution_time_mean': 'site_avg_exec_time',
                'execution_time_std': 'site_std_exec_time',
                'queue_time_mean': 'site_avg_queue_time',
                'queue_time_std': 'site_std_queue_time',
                'job_success_mean': 'site_success_rate',
                'job_success_count': 'site_job_count',
                'processing_efficiency_mean': 'site_avg_efficiency',
                'processing_efficiency_std': 'site_std_efficiency'
            })
            
            self.site_stats = site_stats
        
        # Merge site statistics
        if self.site_stats is not None:
            df = df.merge(self.site_stats, on='site_name', how='left')
            
            # Fill missing values for new sites with global averages
            for col in self.site_stats.columns:
                if col != 'site_name' and col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def extract_entry_features(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """
        Extract entry_name level features
        """
        df = df.copy()
        
        if 'entry_name' not in df.columns:
            return df
        
        if training:
            entry_stats = df.groupby('entry_name').agg({
                'total_wall_time': ['mean', 'std'],
                'job_success': 'mean',
                'processing_efficiency': 'mean'
            })
            
            entry_stats.columns = ['_'.join(col).strip() for col in entry_stats.columns.values]
            entry_stats = entry_stats.reset_index()
            
            entry_stats = entry_stats.rename(columns={
                'total_wall_time_mean': 'entry_avg_wall_time',
                'total_wall_time_std': 'entry_std_wall_time',
                'job_success_mean': 'entry_success_rate',
                'processing_efficiency_mean': 'entry_avg_efficiency'
            })
            
            self.entry_stats = entry_stats
        
        if hasattr(self, 'entry_stats') and self.entry_stats is not None:
            df = df.merge(self.entry_stats, on='entry_name', how='left')
            
            for col in self.entry_stats.columns:
                if col != 'entry_name' and col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def extract_workflow_features(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """
        Extract workflow and stage level features
        """
        df = df.copy()
        
        if training:
            workflow_stats = df.groupby('workflow_id').agg({
                'total_wall_time': ['mean', 'std'],
                'job_success': 'mean'
            })
            
            workflow_stats.columns = ['_'.join(col).strip() for col in workflow_stats.columns.values]
            workflow_stats = workflow_stats.reset_index()
            
            workflow_stats = workflow_stats.rename(columns={
                'total_wall_time_mean': 'workflow_avg_time',
                'total_wall_time_std': 'workflow_std_time',
                'job_success_mean': 'workflow_success_rate'
            })
            
            self.workflow_stats = workflow_stats
        
        if hasattr(self, 'workflow_stats') and self.workflow_stats is not None:
            df = df.merge(self.workflow_stats, on='workflow_id', how='left')
            
            for col in self.workflow_stats.columns:
                if col != 'workflow_id' and col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def compute_site_load(self, df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
        """
        Compute concurrent job load at each site
        """
        df = df.copy()
        df = df.sort_values('job_submitted_time')
        
        site_loads = []
        
        for idx, row in df.iterrows():
            if pd.isna(row['site_name']):
                site_loads.append(0)
                continue
                
            time_window_start = row['job_submitted_time'] - pd.Timedelta(hours=window_hours)
            
            # Count jobs at same site in time window
            concurrent_jobs = df[
                (df['site_name'] == row['site_name']) &
                (df['job_submitted_time'] >= time_window_start) &
                (df['job_submitted_time'] < row['job_submitted_time'])
            ]
            
            site_loads.append(len(concurrent_jobs))
        
        df['site_load_recent'] = site_loads
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables
        """
        df = df.copy()
        
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = ['site_name', 'entry_name']
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            if training:
                encoder = LabelEncoder()
                df[f'{col}_encoded'] = encoder.fit_transform(df[col].fillna('unknown'))
                self.site_encoders[col] = encoder
            else:
                if col in self.site_encoders:
                    encoder = self.site_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].fillna('unknown')
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ 
                        else -1
                    )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features
        """
        df = df.copy()
        
        # Interaction between site and time of day
        if 'site_name_encoded' in df.columns and 'submit_hour' in df.columns:
            df['site_hour_interaction'] = df['site_name_encoded'] * df['submit_hour']
        
        # Interaction between workflow and site
        if 'workflow_id' in df.columns and 'site_name_encoded' in df.columns:
            df['workflow_site_interaction'] = df['workflow_id'] * df['site_name_encoded']
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete feature engineering pipeline (training)
        """
        print("Extracting temporal features...")
        df = self.extract_temporal_features(df)
        
        print("Extracting site features...")
        df = self.extract_site_features(df, training=True)
        
        print("Extracting entry features...")
        df = self.extract_entry_features(df, training=True)
        
        print("Extracting workflow features...")
        df = self.extract_workflow_features(df, training=True)
        
        print("Computing site load...")
        df = self.compute_site_load(df)
        
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df, training=True)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Define feature columns
        feature_cols = [
            # Temporal features
            'submit_hour', 'submit_day_of_week', 'is_weekend', 'is_business_hours',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            
            # Job characteristics
            'workflow_id', 'stage_id', 'files_processed_count', 'files_created_count',
            
            # Site features
            'site_avg_wall_time', 'site_std_wall_time', 'site_med_wall_time',
            'site_avg_exec_time', 'site_avg_queue_time', 'site_success_rate',
            'site_job_count', 'site_avg_efficiency', 'site_load_recent',
            
            # Entry features
            'entry_avg_wall_time', 'entry_success_rate', 'entry_avg_efficiency',
            
            # Workflow features
            'workflow_avg_time', 'workflow_success_rate',
            
            # Encoded features
            'site_name_encoded', 'entry_name_encoded',
            
            # Interactions
            'site_hour_interaction', 'workflow_site_interaction',
            
            # Timing features (if available)
            'queue_time', 'startup_time'
        ]
        
        # Only keep features that exist in the dataframe
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        print(f"Total features: {len(feature_cols)}")
        
        return df, feature_cols
    
    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Transform new data using fitted encoders (inference)
        """
        df = self.extract_temporal_features(df)
        df = self.extract_site_features(df, training=False)
        df = self.extract_entry_features(df, training=False)
        df = self.extract_workflow_features(df, training=False)
        df = self.compute_site_load(df)
        df = self.encode_categorical_features(df, training=False)
        df = self.create_interaction_features(df)
        
        # Return only the required features
        return df[feature_cols]
