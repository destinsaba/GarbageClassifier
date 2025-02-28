# Garbage Classifier
Group 5: Cole Cathcart, Destin Saba

Neural network solution to classify garbage into appropriate waste bin based on an image and a short description.

### Instructions to run
Please find a saved copy of our model here: https://drive.google.com/file/d/1w0Rk6Ehz0GbM5Law7gbiNL1icizd1ZQc/view?usp=sharing

In order to run the jupyter notebook you will need to download the saved model from the above link or generate a new one with the python file (more instructions are in the notebook). The python file should be runnable as-is, though you may need to change the paths if not running on TALC

### Repo contents
This repo contains all of the code required to train, test and analyze our solution model:
* **garbageClassifier.py**: Contains the model and data processing classes, and functions to train and evaluate the model
* **predictionResults.ipynb**: A notebook which loads a previously-trained model and runs evaluation only, along with metrics and visualizations
* **build.slurm**: The script used for submitting the model to be trained on TALC

### Experimental setup
Below is outlined the structure of our solution and how we conducted design and tuning for the problem:
* **Solution design**: We used classes where applicable to make our code more reusable and easier to iterate upon. Our solution includes a class to load the dataset, preprocess the images with transforms and normalization, and preprocess the image filenames into tokenized words with DistilBert. We also define a multi-modal neural network that combines an image and text classifier, and functions to train and evaluate the model.
* **Model**: Our model class makes use of transfer learning to create a multi-modal classifier. Pre-trained models from DistilBert and resnet are loaded and have their final layers unfrozen for fine-tuning. In addition to normalization, activation and dropout layers, the model makes use of a gate layer which learns to weight the text and image features.
* **Training and tuning**: The model was trained on GPUs on the TALC cluster using the provided dataset. Our model trains for a preset number of epochs, saving the model at epochs with the lowest validation loss. The final model is saved to a file for future use. Hyperparameter tuning was done manually due to the long training times making gridsearch unfeasible. We experimented with many hyperparameters as well as different schedulers and optimizers, but eventually narrowed our focus to iterative testing of batch size, learning rate and dropout rate.
* **Evaluation**: There are evaluation functions in both the python file and jupyter notebook for flexibility. To aid in analysis, we retrieve several metrics in addition to the test accuracy to provide more insight into the strengths and weaknesses of the model and provide visualizations of performance.
