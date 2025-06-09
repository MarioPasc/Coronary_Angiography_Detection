from ICA_Detection.explainer import build_explainer, display_images

# one line: choose model by name, point to weights

weight = "/home/mpascual/misc/x2go_shared/kfold_results/kfold_results/yolov8l/gpsampler/out/fold_1/weights/best.pt"
image = "/home/mpascual/research/datasets/angio/tasks/stenosis_detection/images/cadica_p42_v5_00022.png"

cam = build_explainer(
    "DCA_YOLOv8",
    weight=weight,
    method="eigencam",
    conf_threshold=0.3,
)

imgs = cam(image)          # or a directory
display_images(imgs)
