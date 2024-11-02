# Ideas for visualization in the dataset:
# 1. Samples per set (training/validation/test). Include a subplot with 2 plots: 
#       I. Set-based visualization: barplot; X-axis: "training" "validation" "test". Y-axis: counts; bars: "Lesion" "No Lesion" "Total"
#       II. Label-based visualization: barplot; X-axis: Diagnostic labels, Y-axis: counts; bars: "Training" "Validation" "Test"
# 2. Undersampling/Augmentation types used.
#       I. 4 images from a specific patient that show each augmentation image generated
#       II. Augmented images per label: Comparison between original, undersampled, and augmented
# 3. Simulated Annealing visualization: One scatterplot per adjusted hyperparameter, mark the best result. X-Axis: hyperparameter values; Y-Axis: fitness
# 4. Goal-oriented sensibility study of hyperparameters. 
#       I. lineplot for variation of mAP@50-95 per epoch for every hyperparameter value tried. X-axis: epoch, Y-axis: mAP@50-95, legend: Hyperparameter values
#       II. lineplot for variation of mAP@50-95 per hyperparameter value. X-axis: hyperparameter value, Y-axis: mAP@50-95
# 5. Results comparison: Lineplot of mAP@50-95 performance per model. X-axis: epoch, Y-axis: mAP@50-95, one data series per model (baseline, Simulated Annealing, iteration 1, iteration 2)
