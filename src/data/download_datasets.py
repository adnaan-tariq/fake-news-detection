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
    
    # def download_kaggle_dataset(self):
    #     """Download dataset from Kaggle."""
    #     logger.info("Downloading dataset from Kaggle...")
        
    #     # Kaggle dataset ID
    #     dataset_id = "clmentbisaillon/fake-and-real-news-dataset"
        
    #     try:
    #         kaggle.api.dataset_download_files(
    #             dataset_id,
    #             path=self.raw_data_dir,
    #             unzip=True
    #         )
    #         logger.info("Successfully downloaded dataset from Kaggle")
    #     except Exception as e:
    #         logger.error(f"Error downloading from Kaggle: {str(e)}")
    #         logger.info("Please download the dataset manually from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    
    # def download_liar(self):
    #     """Download LIAR dataset."""
    #     logger.info("Downloading LIAR dataset...")
        
    #     # URL for LIAR dataset
    #     url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
    #     output_path = self.raw_data_dir / "liar_dataset.zip"
        
    #     if not output_path.exists():
    #         try:
    #             response = requests.get(url, stream=True)
    #             total_size = int(response.headers.get('content-length', 0))
                
    #             with open(output_path, 'wb') as f, tqdm(
    #                 desc="Downloading LIAR dataset",
    #                 total=total_size,
    #                 unit='iB',
    #                 unit_scale=True
    #             ) as pbar:
    #                 for data in response.iter_content(chunk_size=1024):
    #                     size = f.write(data)
    #                     pbar.update(size)
                
    #             # Extract the zip file
    #             with zipfile.ZipFile(output_path, 'r') as zip_ref:
    #                 zip_ref.extractall(self.raw_data_dir / "liar")
    #         except Exception as e:
    #             logger.error(f"Error downloading LIAR dataset: {str(e)}")
    #             logger.info("Please download the LIAR dataset manually from: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
    
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
    
    # Download datasets
    # downloader.download_kaggle_dataset()
    # downloader.download_liar()
    
    # Process datasets
    downloader.process_kaggle_dataset()
    downloader.process_liar()
    
    # Combine datasets
    downloader.combine_datasets()
    
    logger.info("Dataset preparation completed!")

if __name__ == "__main__":
    main() 