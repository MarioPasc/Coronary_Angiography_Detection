from external.ultralytics import YOLO
import pandas as pd

if __name__ == "__main__":
    # Define a dictionary with symbolic names as keys and model paths as values
    model_paths = {
        "Patient_Video-based": '/home/mariopasc/Python/Projects/Coronary_Angiography_Detection/models/iteration_2.pt',
        "Patient-based": "/home/mariopasc/Python/Results/Coronariografias/Difference_performance/detect/holdout_patients/weights/best.pt"
    }
    
    save_path = "/home/mariopasc/Python/Results/Coronariografias/Difference_performance"
    
    # Initialize a list to store results
    results = []

    # Loop through each model name and path in the dictionary
    for model_name, model_path in model_paths.items():
        # Loop through each split
        for split in ['train', 'val', 'test']:
            print(f"Inference on split {split} with model {model_name}")
            
            # Initialize the YOLO model
            model = YOLO(model=model_path, task="detect", verbose=True)
            
            # Perform validation and get results
            val = model.val(data="./CADICA_Detection/model/config.yaml", imgsz=640, batch=8, iou=0.6, plots=True, split=split, workers=0)
            
            # Collect the model name, split name, and mAP50-95 value
            results.append({
                "Model": model_name,
                "Set": split,
                "mAP50-95": val.box.map
            })

    # Convert the results into a DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv('./results_validation_on_sets.csv', index=False)
