# Classification algorithms evaluated against published clustering methods for automated flow cytometry data analysis

## Running analysis

- Evaluation data used in the study is unavailable due to patient confidentiality.
- To use your own input data, a pickled pandas Dataframe is required in the format where each row is an FC event, each column is a marker and additional columns ```label``` and ```sample_name``` indicate the event label and the name of the sample the event is from, respectively.

#### benchmarking
- ```benchmarking.py``` begins the pipeline to run clustering methods on the input data and match outputs to labelled populations using the Hungarian assignment algorithm, then evaluate performance.
- FLOCK requires installation from [here](http://sourceforge.net/projects/immportflock/files/FLOCK_flowCAP-I_code/).
- flowgrid requires installation from [here](https://github.com/holab-hku/FlowGrid).
- ```helper_match_evaluate_multiple.R``` was made available by [Weber and Robinson, 2016](https://onlinelibrary.wiley.com/doi/10.1002/cyto.a.23030).

#### classification
- ```classification.py``` will run Decision Tree, Random Forest and XGBoost classifiers to train and evaluate on input data.

## Results

#### figures
- ```figures.py``` allows recreation of the figures in the publication, except for; Fig 6, Fig 7 and Table 2 which would require the publication of confidential patient data.
- ```figs``` is the folder where the figures will be saved to.

#### data
- Data from the study which is required by ```figure.py``` to recreate publication figures.
- ```confusion_matrix``` contains the confusion matrices generated from each trained classification model.
- ```f1.pickle``` contains the results of the evaluation of the classification methods.
- ```output``` contains the results and confusion matrices generated from each benchmarked clustering method.

