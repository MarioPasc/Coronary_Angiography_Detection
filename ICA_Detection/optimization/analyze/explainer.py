
from YOLOv8_Explainer import yolov8_heatmap, display_images

model = yolov8_heatmap(
    weight="/home/mpascual/misc/x2go_shared/kfold_results/kfold_results/yolov8m/gpsampler/out/fold_1/weights/best.pt", 
        conf_threshold=0.4,  
        method = "EigenCAM", 
        layer=[10, 12, 14, 16, 18, -3],
        ratio=0.02,
        show_box=True,
        renormalize=False,
)

imagelist = model(
    img_path="/home/mpascual/research/datasets/angio/tasks/stenosis_detection/images/cadica_p42_v5_00010.png", 
    )

