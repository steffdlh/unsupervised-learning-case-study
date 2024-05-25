# unsupervised-learning-case-study
 
## Getting started
Please install all requirements from the requirements.txt at first. Ensure compatibility with your python version. As many of the libraries used have direct ties to CUDA and are therefore challenging to run efficiently in a venv, I did not build this as a poetry project.

## Data
Get the data from arxiv papers published and place them in the folder unsupervised_learning_case_study/data folder as a json.

## Training the model
Run the module unsupervised_learning_case_study.pipeline.train_model first by running:
```
python -m unsupervised_learning_case_study.pipeline.train_model
```

## Categorizing the papers
Run the module unsupervised_learning_case_study.pipeline.categorize_paper next to categorize the papers.
```
python -m unsupervised_learning_case_study.pipeline.categorize_paper
```

## Create visualizations
Create the visualizations using unsupervised_learning_case_study.pipeline.viz.
```
python -m unsupervised_learning_case_study.pipeline.viz
```

## Notes on locally stored data
Each of the pipeline steps will store different data locally (corpus, embeddings, model, visualizations,...).
These enable you to e.g. not retrain the model, when you want to classify a new (updated?) set of papers. Just run only the categorization and the visualization part, so you do not need to wait for a lengthy training process.