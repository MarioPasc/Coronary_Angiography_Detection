# Test the performance of an obtained model with the sets.

from ultralytics import YOLO
import pandas as pd

if __name__ == "__main__":
    config_path = './CADICA_Detection/model/config_labels.yaml'
    model_path_train_best = './weights/best.pt'
    model_path_train_last = './weights/last.pt'
    model_path_baseline = '/home/mariopasc/Python/Results/Coronariografias/Baseline_train_val/train_and_validate_ateroesclerosis/weights/best.pt'

    save_path = '/home/mariopasc/Python/Results/Coronariografias/Baseline_train_val/train_and_validate_ateroesclerosis/weights'    
    split = "test"

    # Initialize a list to store results
    results = []

    for split in ['train', 'test', 'val']:
        print(f"Inference on split {split}")
        model = YOLO(model=model_path_train_best, task="detect", verbose=True)
        val = model.val(data="./config.yaml", imgsz=512, batch=8, iou=0.6, plots=True, split=split)
        
        # Collect the split name and mAP50-95 value
        results.append({"Set": split, "mAP50-95": val.box.map})

    # Convert the results into a DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv('./results_validation_on_sets.csv', index=False)