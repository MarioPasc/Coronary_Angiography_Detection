# TODO

- Code a script that automatically generates dataset-wise splits, performs training and validation, removes the splits created, and moves onto the next dataset. 

**Prioridad**:

- Montar el proyecto en picasso?
- Ejecución del conjunto de datos baseline con los 3 modelos

**Futuro**:

- Revisar módulo Swim-Transformer para la selección de frames
- Revisar la disponibilidad de Efficient-Net, MAMBA-YOLO para realizar detección
- La idea principal es entrenar un modelo de segmentación con el conjunto ARCADE, con la finalidad de realizar una segmentación de las arterias. Este modelo entrenado con ARCADE luego podrá generar máscaras para las arterias en CADICA o KEMEROVO. Luego, se propondrá un modelo de detección que realice detección en la imagen cruda, y la máscara de segmentación extraída. Se comprobará dónde se realiza la detección mejor.

**Completado:**

- [X] Revisar el resultado del preprocesamiento FSE comparado con el artículo
- [X] Arreglar el problema de mantener el formato Pascal VOC después de estandarizar el tamaño de la imagen
