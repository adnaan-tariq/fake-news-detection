import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
import logging
from tqdm import tqdm
import json
# import kaggle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.processed_data_dir = self.project_root / "data" / "processed"
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    

    def process_kaggle_dataset(self):
        """Process the Kaggle dataset."""
        logger.info("Processing Kaggle dataset...")
        
        # Read fake and real news files
        fake_df = pd.read_csv(self.raw_data_dir / "Fake.csv")
        true_df = pd.read_csv(self.raw_data_dir / "True.csv")
        
        # Add labels
        fake_df['label'] = 1  # 1 for fake
        true_df['label'] = 0  # 0 for real
        
        # Combine datasets
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Save processed data
        combined_df.to_csv(self.processed_data_dir / "kaggle_processed.csv", index=False)
        logger.info(f"Saved {len(combined_df)} articles from Kaggle dataset")
    
    def process_liar(self):
        """Process LIAR dataset."""
        logger.info("Processing LIAR dataset...")
        
        # Read LIAR dataset
        liar_file = self.raw_data_dir / "liar" / "train.tsv"
        if not liar_file.exists():
            logger.error("LIAR dataset not found!")
            return
        
        # Read TSV file
        df = pd.read_csv(liar_file, sep='\t', header=None)
        
        # Rename columns
        df.columns = [
            'id', 'label', 'statement', 'subject', 'speaker',
            'job_title', 'state_info', 'party_affiliation',
            'barely_true', 'false', 'half_true', 'mostly_true',
            'pants_on_fire', 'venue'
        ]
        
        # Convert labels to binary (0 for true, 1 for false)
        label_map = {
            'true': 0,
            'mostly-true': 0,
            'half-true': 0,
            'barely-true': 1,
            'false': 1,
            'pants-fire': 1
        }
        df['label'] = df['label'].map(label_map)
        
        # Select relevant columns
        df = df[['statement', 'label', 'subject', 'speaker', 'party_affiliation']]
        df.columns = ['text', 'label', 'subject', 'speaker', 'party']
        
        # Save processed data
        df.to_csv(self.processed_data_dir / "liar_processed.csv", index=False)
        logger.info(f"Saved {len(df)} articles from LIAR dataset")
    
    def combine_datasets(self):
        """Combine processed datasets."""
        logger.info("Combining datasets...")
        
        # Read processed datasets
        kaggle_df = pd.read_csv(self.processed_data_dir / "kaggle_processed.csv")
        liar_df = pd.read_csv(self.processed_data_dir / "liar_processed.csv")
        
        # Combine datasets
        combined_df = pd.concat([
            kaggle_df[['text', 'label']],
            liar_df[['text', 'label']]
        ], ignore_index=True)
        
        # Save combined dataset
        combined_df.to_csv(self.processed_data_dir / "combined_dataset.csv", index=False)
        logger.info(f"Combined dataset contains {len(combined_df)} articles")

def main():
    downloader = DatasetDownloader()
    
    # Process datasets
    downloader.process_kaggle_dataset()
    downloader.process_liar()
    
    # Combine datasets
    downloader.combine_datasets()
    
    logger.info("Dataset preparation completed!")

if __name__ == "__main__":
    main() 