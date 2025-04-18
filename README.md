# Hello, Penguins
 Machine learning experiments with the Palmer Penguins dataset


<img src="images/penguins.png" alt="Palmer Penguins illustration" width="400">
<br/>

*Illustration by @allison_horst*


## Welcome
Welcome to the “Hello Penguins” repository, a collection of machine learning experiments with the Palmer Penguins dataset. 

Inspired by the “Hello, World!” programming tradition, this repository is a series of small experiments to illustrate foundational machine learning concepts. Each experiment includes evaluation metrics and visuals to verify the model predictions make sense and are explainable.

Software engineering concepts are used to ensure the code is testable and reproducible.

To learn more about the dataset, checkout the the official [Palmer Penguins GitHub repo](https://github.com/allisonhorst/palmerpenguins/blob/main/README.md).

## Training Approach and Technology 
MLflow is used for model training and evaluation instead of notebooks. 

Training happens locally and the experiment results are shared in an [MLflow portfolio](https://hello-penguins-981081086626.us-central1.run.app/#/) that is hosted with Google Cloud Run. The goal is to have the portfolio highly available, but there may be times when it is offline.  The portfolio Docker container files are in the [docker-portfolio](/docker-portfolio/) directory. 

## Pre-Training Checks
* Consider data [bias](/pretraining/bias.md)


## Acknowledgements and Sources

### This repo builds on many foundations:

* Allison Horst’s [Palmer Penguins repo](https://github.com/allisonhorst/palmerpenguins/blob/main/README.md) 
    - Data downloaded 3/16/2025 ```curl -o data/penguins.csv https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv```
* Lynn Langit’s mentorship and amazing resources for [learning cloud](https://github.com/lynnlangit/learning-cloud) 
* Santiago Valdarrama’s [ML School repo](https://github.com/svpino/ml.school)

