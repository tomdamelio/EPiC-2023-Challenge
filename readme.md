# Emotion Physiology and Experience Collaboration (EPiC) Shared Task Repository
This repository contains the work done as part of the Emotion Physiology and Experience Collaboration (EPiC) shared task competition. The goal of the competition is to develop models and methods for understanding and predicting moment-to-moment ratings of emotion using measures of physiology.

## Repository Structure

`./preprocess`: This folder contains files for preprocessing and generating continuous features from the raw data. The scripts in this directory are responsible for cleaning, transforming, and preparing the data for the modeling stage.

`./test_scripts`: This folder contains the final scripts for making predictions using the preprocessed data. These scripts load the preprocessed data, train the models, and generate predictions for the four scenarios proposed in the challenge.

`./results`: This folder contains the final predictions for each of the four scenarios proposed in the challenge. The predictions are stored as separate files, one for each scenario, in a format specified by the competition guidelines.

`./src`: This folder contains other files, notebooks, and results related to the work done for the challenge. However, these files are not used in the general pipeline of the predictive models. This directory is mainly used for exploratory analysis, additional experiments, and other research-related activities.
