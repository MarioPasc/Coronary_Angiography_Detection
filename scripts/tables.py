import pandas as pd
import os
import logging
import yaml

# Set up logging
os.makedirs('./logs', exist_ok=True)
logging.basicConfig(filename='./logs/visualization.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting plots generation process.")

CONFIG_PATH = "./scripts/config.yaml"

def load_config(yaml_path: str) -> dict:
    """
    Loads configuration parameters from a YAML file into a dictionary.
    
    Args
    -----
    yaml_path : str
        Path to the YAML configuration file.
    
    Returns
    -----
    dict
        Configuration parameters loaded from YAML.
    """
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error("Error loading config.yaml file; Are you executing the script from the root folder? If so, check this .py and change CONFIG_PATH.")

# Load the CONFIG variable
CONFIG = load_config(CONFIG_PATH)

def analyze_augmentations_csv_only(augmentations: list):
    paths = ['CADICA_Augmented_Images/processed_train.csv',
             'CADICA_Augmented_Images/processed_val.csv',
             'CADICA_Holdout_Info/test.csv']
    csv = [os.path.join(CONFIG["OUTPUT_PATH"], path) for path in paths]

    # Load the CSV files
    train_df = pd.read_csv(csv[0])
    val_df = pd.read_csv(csv[1])
    test_df = pd.read_csv(csv[2])
    
    # Initialize results dictionary
    results = {aug: {"Train": 0, "Val": 0, "Test": 0} for aug in augmentations + ["Not-Augmented"]}

    # Function to count augmentation types in a dataframe
    def count_augmentations(df):
        counts = {aug: 0 for aug in augmentations + ["Not-Augmented"]}
        for aug in augmentations:
            counts[aug] = df[df["Frame_path"].str.contains(aug)].shape[0]
        counts["Not-Augmented"] = df[~df["Frame_path"].str.contains("|".join(augmentations))].shape[0]
        return counts

    # Count augmentations in each dataset
    train_counts = count_augmentations(train_df)
    val_counts = count_augmentations(val_df)
    test_counts = count_augmentations(test_df)

    # Fill the results dictionary
    total_train = len(train_df)
    total_val = len(val_df)
    total_test = len(test_df)

    for aug in results:
        results[aug]["Train"] = (train_counts[aug], train_counts[aug] / total_train * 100)
        results[aug]["Val"] = (val_counts[aug], val_counts[aug] / total_val * 100)
        results[aug]["Test"] = (test_counts[aug], test_counts[aug] / total_test * 100)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame({
        "Augmentation": results.keys(),
        "Train Count": [v["Train"][0] for v in results.values()],
        "Train %": [v["Train"][1] for v in results.values()],
        "Val Count": [v["Val"][0] for v in results.values()],
        "Val %": [v["Val"][1] for v in results.values()],
        "Test Count": [v["Test"][0] for v in results.values()],
        "Test %": [v["Test"][1] for v in results.values()]
    })

    # Save the results to a CSV file
    results_df.to_csv(os.path.join(CONFIG["OUTPUT_PATH"], "augmentation_analysis.csv"), index=False)

    return results_df

if __name__ == "__main__":
    # Load and analyze the uploaded CSVs
    augmentation_types = ["brightness", "contrast", "xray_noise", "elastic", "translation"]

    # Run the updated analysis to generate only the CSV file
    analyze_augmentations_csv_only(
        augmentation_types
    )
