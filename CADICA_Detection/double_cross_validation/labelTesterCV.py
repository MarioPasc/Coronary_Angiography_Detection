import os
import shutil
import logging
import pandas as pd
import yaml
from typing import List
from CADICA_Detection.external.ultralytics.ultralytics import YOLO
import torch.multiprocessing
import torch

class LabelTester:
    def __init__(self, train_df: pd.DataFrame, 
                 val_df: pd.DataFrame, 
                 test_df: pd.DataFrame, 
                 yolo_dataset_path: str, 
                 output_base_dir: str, 
                 model_path: str):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.yolo_dataset_path = yolo_dataset_path
        self.output_base_dir = output_base_dir
        self.model_path = model_path

    def create_label_datasets(self) -> List[str]:
        """
        Creates symlinked YOLO-compatible datasets for each label found in the 
        training, validation, and test DataFrames. Returns a list of all unique labels.
        """
        data_frames = {"train": self.train_df, "val": self.val_df, "test": self.test_df}
        labels = set()

        for df in data_frames.values():
            for label_list in df["LesionLabel"]:
                split_labels = label_list.split(",")
                split_labels = [label.strip().replace("'", "") for label in split_labels]
                labels.update(split_labels)

        labels = list(labels)
        logging.info(f"Unique labels: {labels}")

        for label in labels:
            label_dir = os.path.join(self.output_base_dir, f"CADICA_{label}")
            for split, df in data_frames.items():
                split_images_dir = os.path.join(label_dir, "images", split)
                split_labels_dir = os.path.join(label_dir, "labels", split)
                os.makedirs(split_images_dir, exist_ok=True)
                os.makedirs(split_labels_dir, exist_ok=True)

                for _, row in df.iterrows():
                    frame_path = row["Frame_path"]
                    filename = os.path.basename(frame_path)

                    # Check if this label is present in the current row
                    if label in row["LesionLabel"].split(","):
                        image_src = os.path.join(self.yolo_dataset_path, "images", split, filename)
                        label_filename = os.path.splitext(filename)[0] + ".txt"
                        label_src = os.path.join(self.yolo_dataset_path, "labels", split, label_filename)

                        image_dst = os.path.join(split_images_dir, filename)
                        label_dst = os.path.join(split_labels_dir, label_filename)

                        if os.path.exists(image_src) and not os.path.exists(image_dst):
                            os.symlink(image_src, image_dst)
                        if os.path.exists(label_src) and not os.path.exists(label_dst):
                            os.symlink(label_src, label_dst)

        return labels

    def create_config_files(self, labels: List[str]) -> None:
        """
        Creates a YAML config file for each label, to be used by YOLO for detection.
        """
        for label in labels:
            config = {
                "path": os.path.join(self.output_base_dir, f"CADICA_{label}"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: label},
            }
            config_path = os.path.join(self.output_base_dir, f"config_{label}.yaml")
            with open(config_path, "w") as file:
                yaml.dump(config, file)
            logging.info(f"Created config file for label '{label}' at {config_path}")

    def run_validation_on_labels(self, labels: List[str]) -> pd.DataFrame:
        """
        Runs validation on each label, returns a DataFrame with precision, recall, 
        mAP50, mAP50-95 for train/val/test splits.
        
        At the end, removes all label-specific directories and config files.
        """
        torch.multiprocessing.set_start_method("spawn", force=True)

        results = []
        splits = ["train", "val", "test"]

        for label in labels:
            logging.info(f"Validating model on label '{label}' dataset.")
            config_path = os.path.join(self.output_base_dir, f"config_{label}.yaml")
            model = YOLO(model=self.model_path, task="detect", verbose=True)

            label_results = {"Label": label}
            for split in splits:
                logging.info(f"Validating split '{split}' for label '{label}'.")
                val = model.val(
                    data=config_path,
                    imgsz=512,
                    batch=16,
                    iou=0.5,
                    plots=False,
                    split=split,
                    workers=0,
                    half=True,
                    device="cuda:0",
                )
                torch.cuda.empty_cache()
                label_results[f"{split}/precision"] = val.box.mp
                label_results[f"{split}/recall"] = val.box.mr
                label_results[f"{split}/mAP50"] = val.box.map50
                label_results[f"{split}/mAP50-95"] = val.box.map

            results.append(label_results)

        # Once we've completed validation for all labels, clean up
        for label in labels:
            # Remove the label-specific directory
            label_dir = os.path.join(self.output_base_dir, f"CADICA_{label}")
            if os.path.exists(label_dir):
                logging.info(f"Removing directory: {label_dir}")
                shutil.rmtree(label_dir, ignore_errors=True)

            # Remove the label-specific config file
            config_file = os.path.join(self.output_base_dir, f"config_{label}.yaml")
            if os.path.exists(config_file):
                logging.info(f"Removing config file: {config_file}")
                os.remove(config_file)

        return pd.DataFrame(results)
