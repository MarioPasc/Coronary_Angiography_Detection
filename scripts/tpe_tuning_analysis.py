"""
Visualizations:
    1. Slice plot. Hyperparameter value in the X-axis, Objective Value in the Y-axis. Trial could be a colormap. 
    2. Hyperparameter importance. It could accompany the visualization 1 in some way. We can measure the hyperparameter importance with
       this optuna.importance submodule: https://optuna.readthedocs.io/en/stable/reference/importance.html 
    3. Epoch evolution of trials against objective value. X-axis is epoch, Y-axis is Objective Value. We can include trials that are 
       completed and pruned or only completed trials. Every trial that is not the best one can be marked grey. 
    4. Trial against Objective Value. X-axis is trial, Y-axis is the Objective Value. We can put this visualization next to plot 3.    
    5. Contour plots of the latent hyperparameter space. We can create a dataframe with each hyperparameter on one column, and add the Objective
       Value obtained for the trial corresponding to that specific se of hyperparameters. We could apply dimensionality reduction to the 
       hyperparameters, keeping the explained varianze high, and plot a contour plot of PC0 vs PC1, where the colour is the mAP@50-95 obtained. 
    6. Pairplot with Contours. We could also see the pariwise distribution of hyperparameters in a pairplot, and add the contour regions based on
       the Objective Value.

"""

PATH = "/home/mariopasc/Python/Results/Coronariografias/patient-based/TPE_Sampler/optuna_study.db"

