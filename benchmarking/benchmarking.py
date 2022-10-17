import pickle
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
import plot_scatter as plots


df = pd.read_pickle("PATH TO DATA")

# select 10 random samples to perform benchmarking on
u_samples = df['sample_name'].unique()
classes = df['label'].unique()

N_REPETITIONS = 5  # number of times to run each method
N_SAMPLES = 10  # number of samples to test
PATH_TO_INPUT = ".../input_data"
PATH_TO_OUTPUT = PATH_TO_INPUT.replace("input_data", "output")
PATH_TO_SCRIPT = PATH_TO_INPUT.replace("/input_data", "")

np.random.seed(0)
samples_to_test = np.random.choice(u_samples, N_SAMPLES)

for sample in samples_to_test:
	# save processed data to csv to path to R methods
	s_df = df[df['sample_name'] == sample]  # get data for that sample
	s_df = s_df.drop(columns=['sample_name', 'Time'])

	samp_name = sample[0:4]

	s_df.to_csv(PATH_TO_INPUT + "/" + samp_name + ".csv", index=False)

# call R script to run the methods on each csv, perform hungarian matching algorithm, save output and time to csv.
os.system("Rscript " + "process_benchmarking.R " + str(N_REPETITIONS) + " " + '"' + str(PATH_TO_INPUT) + '"')

# in python, calculate f1 and plot data
output_files = os.listdir(PATH_TO_OUTPUT)
input_files = os.listdir(PATH_TO_INPUT)

"""
process time taken by each method into dataframe
"""

time_files = [f for f in output_files if f.split("_")[0] == "time"]  # get txt files which are results for time.
time_results_df = pd.DataFrame()
for tf in time_files:
	method = tf.split("_")[1]
	sample = tf.split("_")[2]
	rep = tf.split("_")[3].split(".")[0]

	with open(PATH_TO_OUTPUT + "/" + tf) as f:
		time = f.readlines()

	d = {'method': method, 'sample': sample, 'repetition': rep, 'time': time}
	time_results_df = pd.concat([time_results_df, pd.DataFrame(data=d, index=[0])])

# make plots
predictions_files = [f for f in output_files if f.split("_")[0] == "predictions" and "pickle" not in f]  # get files which are predictions
labels_files = [f for f in input_files if f.split(".")[1] == "csv"]  # get files which are labels (input to algorithms)

[plots.plot_predictions(f, PATH_TO_OUTPUT, PATH_TO_INPUT) for f in predictions_files]  # plot predictions
[plots.plot_labels(f, PATH_TO_INPUT) for f in labels_files]  # plot labels

# generate accuracy results
# loop over each prediction run and calculate results, and add to df, add time taken while doing so
results_df = pd.DataFrame()

# if a predictions file has been created, then methods have run successfully.
for f in predictions_files:

	method = f.split("_")[1]
	sample = f.split("_")[2]
	rep = f.split("_")[3].split(".")[0]

	# if method fails, write the error to the time file
	samp_time = time_results_df[(time_results_df['method'] == method) &
									(time_results_df['sample'] == sample) &
									(time_results_df['repetition'] == rep)]['time'].values[0]

	if "ERROR" not in samp_time:

		# load original data to get labels
		df_labels = pd.read_csv(PATH_TO_INPUT + "/" + sample + ".csv")
		labels = df_labels["label"].values
		df_predictions = pd.read_csv(PATH_TO_OUTPUT + "/" + f)
		# convert nan to str
		df_predictions.fillna('nan', inplace=True)
		preds = df_predictions.iloc[:, 0].values

		# performance metrics

		"""
		pass in 'labels' param, metrics are only calculated for true class events. ie. not nan's which
		indicate method didn't make prediction.

		nan's are still used to make calculations for metrics, but nan's always count as incorrect, but don't 
		affect averages for metrics.

		this means that when labels arg been passed into cm's can't derive true metrics from them.

		so save two types of cm:
		- one not including nan's, eg. for figures
		- one including nan's eg. used for calculations
		"""

		# overall
		f1_micro = f1_score(labels, preds, average='micro', labels=classes)
		f1_macro = f1_score(labels, preds, average='macro', labels=classes)
		f1_weighted = f1_score(labels, preds, average='weighted', labels=classes)
		acc = accuracy_score(labels, preds)

		p_micro = precision_score(labels, preds, average='micro', labels=classes)
		p_macro = precision_score(labels, preds, average='macro', labels=classes)
		p_weighted = precision_score(labels, preds, average='weighted', labels=classes)
		
		r_micro = recall_score(labels, preds, average='micro', labels=classes)
		r_macro = recall_score(labels, preds, average='macro', labels=classes)
		r_weighted = recall_score(labels, preds, average='weighted', labels=classes)

		# per class
		f1_class = f1_score(labels, preds, average=None, labels=classes)
		p_class = precision_score(labels, preds, average=None, labels=classes)
		r_class = recall_score(labels, preds, average=None, labels=classes)

		# confusion matrices
		cm_norm_none_nonan = confusion_matrix(labels, preds, labels=classes, normalize=None)
		cm_norm_true_nonan = confusion_matrix(labels, preds, labels=classes, normalize='true')
		disp_norm_none_nonan = ConfusionMatrixDisplay(confusion_matrix=cm_norm_none_nonan, display_labels=classes)
		disp_norm_true_nonan = ConfusionMatrixDisplay(confusion_matrix=cm_norm_true_nonan, display_labels=classes)

		cm_norm_none = confusion_matrix(labels, preds, normalize=None)
		cm_norm_true = confusion_matrix(labels, preds, normalize='true')
		disp_norm_none = ConfusionMatrixDisplay(confusion_matrix=cm_norm_none, display_labels=unique_labels(labels, preds))
		disp_norm_true = ConfusionMatrixDisplay(confusion_matrix=cm_norm_true, display_labels=unique_labels(labels, preds))

		cm = {
			'cm_norm_none_nonan': cm_norm_none_nonan,
			'cm_norm_true_nonan': cm_norm_true_nonan,
			'disp_norm_none_nonan': disp_norm_none_nonan,
			'disp_norm_true_nonan': disp_norm_true_nonan,
			'cm_norm_none': cm_norm_none,
			'cm_norm_true': cm_norm_true,
			'disp_norm_none': disp_norm_none,
			'disp_norm_true': disp_norm_true,
			# record what is label order for cm's with nan, as 'labels' arg not passed
			'label_order_with_nan': unique_labels(labels, preds)
		}

		with open(PATH_TO_OUTPUT + "/" + f[:-4] + "_cm.pickle", 'wb') as handle:
			pickle.dump(cm, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# add results to df.
		# get time taken form df for that run
		time = float(time_results_df[(time_results_df['method'] == method) &
								(time_results_df['sample'] == sample) &
								(time_results_df['repetition'] == rep)]['time'].values[0])

		d = {'method': [method], 'sample': [sample], 'rep': [rep], 'f1_micro': [f1_micro], 'f1_macro': [f1_macro],
				'f1_weighted': [f1_weighted], 'acc': [acc], 'p_micro': [p_micro], 'p_macro': [p_macro], 'p_weighted': [p_weighted],
				'r_micro': [r_micro], 'r_macro': [r_macro], 'r_weighted': [r_weighted], 'time': [time], 'success': ["y"]}

		# loop over metrics for each class, and add to dict.
		for name, metric in zip(['f1_', 'p_', 'r_'], [f1_class, p_class, r_class]):
			for m, c in zip(metric, classes):
				d[name + c] = [m]

		results_df = pd.concat([results_df, pd.DataFrame(data=d, index=[0])])

# if methods haven't run successfully then errors are written to "time file", read this df and check for errors to add to results df
# get errors from time_df
for idx, row in time_results_df.iterrows():
	method = row['method']
	sample = row['sample']
	rep = row['repetition']
	time = row['time']

	time_str = time.split("-")[0]
	# if error then add to results df as error
	if time_str == "ERROR":
		error_msg = time
		d = {'method': [method], 'sample': [sample], 'rep': [rep], 'f1_micro': [np.NaN], 'f1_macro': [np.NaN],
				'f1_weighted': [np.NaN], 'acc': [np.NaN], 'p_micro': [np.NaN], 'p_macro': [np.NaN], 'p_weighted': [np.NaN],
				'r_micro': [np.Nan], 'r_macro': [np.Nan], 'r_weighted': [np.NaN],'time': [np.NaN], 'success': [error_msg]}

		# loop over metrics for each class, and add to dict.
		for name, metric in zip(['f1_', 'p_', 'r_'], [f1_class, p_class, r_class]):
			for m, c in zip(metric, classes):
				d[name + c] = [np.Nan]

		results_df = pd.concat([results_df, pd.DataFrame(data=d, index=[0])])

with open(PATH_TO_OUTPUT + "/" + "results_df.pickle", 'wb') as handle:
	pickle.dump(results_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
