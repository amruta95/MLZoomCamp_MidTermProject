# MLBookCamp_MidTermProject

# Overview
Nowadays, arguing without committing a fallacy is one of the main requirements of an ideal debate.But even when debating rules are strictly enforced and fallacious arguments punished, arguers often lapse into attacking the opponent by an ad-hominem argument.
When challenged by genuine back-and-forth argumentation, humans do better in both generating and evaluating arguments (Mercier and Sperber, 2011). The dialogical perspective on argumentation has been reflected in argumentation theory prominently by the pragma-dialectic model of argumentation


Change My View (CMV) is an online platform for ‘good-faith’ argumentation.A user posts a submission (also called original post(er); OP) and other participants provide arguments to change the OP’s view, forming a typical tree-form Web discussion. This project is on ad-hominem predictions of dialogues in the online interactions by users. This is done as part of a Machine Learning course held by Mr. Alexey Grigorev. This is the [dataset](https://panda.uni-paderborn.de/pluginfile.php/1752026/mod_resource/content/1/test-data-prepared.json) used for this project.

# Problem Description
Nowadays, many people are vocal with their ideas about various news, articles or posts online. There is a high possibility of the online interactions between users to turn into ad-hominem(rude) conversations. Since, people have unique opinions there can be a difference of viewpoints between them. In such scenarios they tend to write abusive or rude comments which in turn changes the conversation into a toxic one. In this project, we are predicting the outcome of dialogues, if they are ad-hominem(rude) or not.

# Folder Contents
The folder contains the following files:

- MidTerm_Project_Script.ipynb :
  This script contains the code for the following,
  - Data Cleaning and pre-processing.
  - EDA : Dialogue Modelling, Feature Analysis, Target features analysis, Important features extraction
  - Training of 6 classification models.
  - Hyperparameter Tuning of all classification models(with output).
  - Predictions with Results.
  
  
- train.py : to train the final model
- predict.py : to load the model 
- eval.py : To evaluate the F1 score on validation and training data.
- requirements.txt : Contains all the required libraries to run the project.
- train-data-prepared : Split dataset for training.
- val-data-prepared : split dataset for validation.

# Steps to run the code
- Clone the folder to your local device. <br>
  Command : git clone https://github.com/amruta95/MLBookCamp_MidTermProject.git
- Else, you can download the **MLBoot** zip folder to your local device.Extract the contents of the folder.
- Open the python notebook, “MidTerm_Project_Script.ipynb” and run all the cells one-by-one (some training instances can take time) . 
- A **predictions_(nameofclassifier).json** file will be generated in the parent folder which will contain all the predictions for the desired
test file.
- Then open this complete folder in PyCharm or Visual Studio Code.  
- Run _eval.py_ in the terminal to check the prediction f1 scores for both the training and validation files using the following commands:<br>


  $ python eval.py -t (path-to-ground-truth-file) -p (path-to-predictions-file)

    Example:<br>
    $ python eval.py -t train-data-prepared.json -p predictions_svm.json

- Similarily, run the eval script on validation data using the following commands :<br>
  $ python eval.py -t (path-to-ground-truth-file) -p (path-to-predictions-file)

    Example:<br>
    $ python eval.py -t val-data-prepared.json -p predictions_svm.json
  
# Results 

## Prediction Scores 
After training on all classifiers, I found out that the best performing model is SVM with a f1 score of 0.70 on the validation dataset.
To check for overfitting, we also had a look at the f1 score on the training dataset which is 0.74 which means both the f1 scores are quite similar so there is no overfitting on the training dataset.

