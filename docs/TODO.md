# TODO

1. We must choose a resolution-free format to standarize the bbox coordinates before choosing any model. 

2. Write a script that:
   - Given (input):
      1. Folderpath to a dataset
      2. Set (list) of models to train and validate on the dataset
      3. Number of splits (K)
      4. Folderpath to an output folder
   - Using the original dataset, which will have two subfolders, images/ and labels/, in whose the files will be named the same but with different terminations (e.g. cadica_p1_v12_00004 is the name, and it will either have .png for the image or .txt for the label). You must create, using symbolic links, N folds, following a Stratified K-Fold Cross Validation methodology. Each fold will have a training set and a testing set. Depending on the number of folds, due to the fact that it is a stratified k-fold, the % of the files that will be assigned to each split (train/test) per fold will vary depending on K. (e.g. if `K==3`, then we will have 75% train and 25% test, but, if `K==2`, we will have 50% train 50% test per fold)
   - Now, when the folds are generated, we are going to start training and testing each given model. !!! Qué hacemos con las labels ...

3. Revisar módulo Swim-Transformer para la selección de frames
4. Revisar la disponibilidad de Efficient-Net, MAMBA-YOLO para realizar detección

--- 
### Completado
---

- [X] Revisar el resultado del preprocesamiento FSE comparado con el artículo
