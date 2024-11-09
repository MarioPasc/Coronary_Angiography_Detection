#!/usr/bin/env python3

# This script executes the hyperparameter tuning we applied in the paper. 
# This hyperparameter optimization is grounded in a goal-oriented hyperparameter sensitivity analysis. 
# Using the configuration provided by the Simulated Annealing algorithm (via Ultralyitcs' .tune() function) as our base model, 
# we iteratively adjust one hyperparameter at a time. For each parameter, we combine its adjusted value with the base configuration, 
# then train and validate the model to observe changes in mAP@50-95. If performance improves, we continue adjusting in the same 
# direction until reaching a local maximum, after which we fine-tune values around this maximum to identify the best setting for each hyperparameter.
# Once optimal values are found for all 10 hyperparameters, we merge them into a final configuration for our YOLO model. 
# We then train and validate this optimized model on an external dataset, achieving improved results.

from CADICA_Detection.model.optimization import HyperparameterTuning

def fixed_values():    
    lr0 = 1.0e-05
    lrf = 0.00829
    momentum = 0.70064
    weight_decay = 0.00048
    warmup_epochs = 3.66787
    warmup_momentum = 0.78696
    warmup_bias_lr = 0.1
    box = 8.57719
    cls = 0.68361
    dfl = 1.19862

    hyperparameters = {
        'lr0': [5.0e-6, 7.5e-6, lr0, 5.0e-5, 7.5e-5],
        'lrf': [0.004, 0.00614, lrf, 0.012, 0.016],
        'momentum': [0.68, 0.69, momentum, 0.71, 0.72],
        'weight_decay': [0.00024, 0.00036, weight_decay, 0.0006, 0.00072],
        'warmup_epochs': [3.4, 3.5, warmup_epochs, 3.6, 3.7],
        'warmup_momentum': [0.69, 0.74, warmup_momentum, 0.83, 0.88],
        'warmup_bias_lr': [0.05, 0.075, warmup_bias_lr, 0.15, 0.175],
        'box': [7.8, 8.18, box, 8.98, 9.38],
        'cls': [0.61, 0.65, cls, 0.72, 0.75],
        'dfl': [1.05, 1.12, dfl, 1.27, 1.34]
    }

    tuner = HyperparameterTuning(output='output_tuning', model='yolov8l.pt', config_yaml_path='config.yaml', yaml_params_path='args.yaml')
    tuner.tune_hyperparameters(hyperparameters, epochs_per_iteration=100, random=False)

if __name__ == "__main__":
    fixed_values()
