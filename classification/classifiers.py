import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import pickle
import seaborn as sns
import xgboost as xgb
import time


def record_val_counts(current_df, d_sets, model, split):
	# count values for each label for each dataset
	# return updated results_df

	new_df = current_df
	for s_k, s_v in d_sets.items():
		unique, counts = np.unique(s_v, return_counts=True)
		val_counts = {'split': split, 'set': s_k, 'model': model}
		for c in zip(unique, counts):
			val_counts[inv_label_encoder[c[0]] + "_counts"] = c[1]

		new_df = pd.concat([new_df, pd.DataFrame(data=val_counts, index=[0])])

	return new_df


def get_x_y(idx_dict, df, u_samples):
	"""
	get data from sample idx and return the actual X and y

	:param idx_dict: the dictionary of idxs to get data from, keys are the data it's used for, vals are idxs
	:param df: the dataframe
	:param u_samples: list of unique samples
	:return: x_y_dict dict of all samples
	"""
	x_y_dict = {}  # dict to hold data

	for d_set, idx in idx_dict.items():

		samps_names = [u_samples[i] for i in idx]  # get names of samps idxs
		df_for_samps = df[df['sample_name'].isin(samps_names)]  # get data for those samples

		d_set_str = d_set.split("_")[0]  # get string of purpose to use as dict key ie. 'test', 'train', 'val'

		# get x data
		x_y_dict['X_' + d_set_str] = df_for_samps.drop(columns=['label', 'sample_name', 'Time']).values

		# get y data
		y = df_for_samps['label']
		x_y_dict['y_' + d_set_str] = y.replace(label_encoder)

	return x_y_dict


JOBS = 50

results_df = pd.DataFrame()
val_counts_df = pd.DataFrame()  # count train/ test instances in each split

df = pd.read_pickle("PATH TO DATA")

unique_samples = df['sample_name'].unique()

# reset idx otherwise causes problems when splitting
df.reset_index(drop=True, inplace=True)

# easier to use dict as labels are simple
label_encoder = {'cd4pos': 0, 'dualpos': 1, 'dualneg': 2, 'cd8pos': 3, 'gdt': 4, 'kappa': 5, 'lambd': 6, 'nk': 7, 'notlymph': 8}
inv_label_encoder = dict((v, k) for k, v in label_encoder.items())

with open("pre_decoded_feat_names.pickle", 'wb') as handle:
	pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

# run 10 k fold CV on DT, RF and xgboost
# but split on samples, not
# run RF and DT first
rkf = KFold(n_splits=10, random_state=0, shuffle=True)

# get indexes of test and train data for k_fold cv
models_data_idx = 10 * []

# split on samples
for train_sample_idx, test_sample_idx in rkf.split(unique_samples):
	models_data_idx.append({'train_index': train_sample_idx, 'test_index': test_sample_idx})

with open("rf_dt_idx.pickle", 'wb') as handle:
	pickle.dump(models_data_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

split = 0  # count splits in data for results_df

# train all models
for idx in models_data_idx:
	
	# extract X_train, X_test, y_train, y_test from samples
	x_y_dict = get_x_y(idx, df, unique_samples)
	X_train, X_test, y_train, y_test = x_y_dict['X_train'], x_y_dict['X_test'], x_y_dict['y_train'], x_y_dict['y_test']

	sets = {'train': y_train, 'test': y_test}

	# count train/ test instances in each split
	val_counts_df = record_val_counts(val_counts_df, sets, 'sklearn', split)

	for balanced in ['balanced', None]:

		# DT
		clf_dt = DecisionTreeClassifier(class_weight=balanced)

		start = time.time()
		clf_dt.fit(X_train, y_train)
		dt_train_time = time.time() - start

		start = time.time()
		y_pred = clf_dt.predict(X_test)
		dt_pred_time = time.time() - start

		# performance metrics
		dt_f1_micro = f1_score(y_test, y_pred, average='micro')
		dt_f1_macro = f1_score(y_test, y_pred, average='macro')
		dt_f1_weighted = f1_score(y_test, y_pred, average='weighted')
		dt_acc = accuracy_score(y_test, y_pred)

		# confusion matrices
		cm_norm_none_dt = confusion_matrix(y_test, y_pred, labels=clf_dt.classes_, normalize=None)
		cm_norm_true_dt = confusion_matrix(y_test, y_pred, labels=clf_dt.classes_, normalize='true')
		disp_norm_none_dt = ConfusionMatrixDisplay(confusion_matrix=cm_norm_none_dt, display_labels=[inv_label_encoder[c] for c in clf_dt.classes_])
		disp_norm_true_dt = ConfusionMatrixDisplay(confusion_matrix=cm_norm_true_dt, display_labels=[inv_label_encoder[c] for c in clf_dt.classes_])

		# RF
		clf_rf = RandomForestClassifier(class_weight=balanced, random_state=0, n_jobs=JOBS)

		start = time.time()
		clf_rf.fit(X_train, y_train)
		rf_train_time = time.time() - start

		start = time.time()
		y_pred = clf_rf.predict(X_test)
		rf_pred_time = time.time() - start

		# performance metrics
		rf_f1_micro = f1_score(y_test, y_pred, average='micro')
		rf_f1_macro = f1_score(y_test, y_pred, average='macro')
		rf_f1_weighted = f1_score(y_test, y_pred, average='weighted')
		rf_acc = accuracy_score(y_test, y_pred)

		# confusion matrices
		cm_norm_none_rf = confusion_matrix(y_test, y_pred, labels=clf_rf.classes_, normalize=None)
		cm_norm_true_rf = confusion_matrix(y_test, y_pred, labels=clf_rf.classes_, normalize='true')
		disp_norm_none_rf = ConfusionMatrixDisplay(confusion_matrix=cm_norm_none_rf, display_labels=[inv_label_encoder[c] for c in clf_rf.classes_])
		disp_norm_true_rf = ConfusionMatrixDisplay(confusion_matrix=cm_norm_true_rf, display_labels=[inv_label_encoder[c] for c in clf_rf.classes_])

		# add results to df
		d = {'f1_micro': [dt_f1_micro, rf_f1_micro], 'f1_macro': [dt_f1_macro, rf_f1_macro], 'f1_weighted': [dt_f1_weighted, rf_f1_weighted],
			'acc': [dt_acc, rf_acc], 'gridsearch_time': ['NA', 'NA'], 'train_time': [dt_train_time, rf_train_time], 'pred_time': [dt_pred_time, rf_pred_time],
			'model': ['DT', 'RF'], 'split': [split, split],	'balanced': [balanced, balanced], 'gamma': ['NA', 'NA'], 'learning_rate': ['NA', 'NA'],
			'max_depth': ['NA', 'NA'], 'reg_lambda': ['NA', 'NA']}
		results_df = pd.concat([results_df, pd.DataFrame(data=d)])

		# save model
		save_name_to_model = {'dt': clf_dt, 'rf': clf_rf}

		for model in save_name_to_model.keys():
			path = ".../models/"
			with open(path + str(model) + "_" + str(balanced) + "_" + str(split) + '.pickle', 'wb') as handle:
				pickle.dump(save_name_to_model[model], handle, protocol=pickle.HIGHEST_PROTOCOL)

		# save confusion matrices
		save_name_to_cm = {'dt': {
			'cm_norm_none': cm_norm_none_dt,
			'cm_norm_true' : cm_norm_true_dt,
			'disp_norm_none': disp_norm_none_dt,
			'disp_norm_true': disp_norm_true_dt
		},
			'rf': {
			'cm_norm_none': cm_norm_none_rf,
			'cm_norm_true': cm_norm_true_rf,
			'disp_norm_none': disp_norm_none_rf,
			'disp_norm_true': disp_norm_true_rf,
		}}

		for cm in save_name_to_cm.keys():
			path = '.../conf_matrix/'
			with open(path + str(cm) + "_" + str(balanced) + "_" + str(split) + "_cm" + '.pickle', 'wb') as handle:
				pickle.dump(save_name_to_cm[cm], handle, protocol=pickle.HIGHEST_PROTOCOL)

		print('DT- RF: splits complete: ' + str(split) + " / " + str(10) + " | Balanced : " + str(balanced))

	# save sklearn train data
	path = '.../data/'
	with open(path + "sklearn_train_" + str(split) + '.pickle', 'wb') as handle:
		pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

	split += 1  # increment split counter

# get indexes of test and train data for k_fold cv
models_data_idx = 10 * []
# use stratified k fold to generate validation set and "rest" (test and train index)
# means validation is always 1/10th of dataset
rkf = KFold(n_splits=10, shuffle=False)
for rest_sample_index, val_sample_index in rkf.split(unique_samples):

	# split "rest" into test and train set
	# of what is left, 0.8 used for training (huge dataset so that is fine for testing)
	train_sample_index, test_sample_index = train_test_split(rest_sample_index, test_size=0.2, random_state=0)

	models_data_idx.append({'train_index': train_sample_index, 'test_index': test_sample_index, 'val_index': val_sample_index})

with open(".../xgb_idx.pickle", 'wb') as handle:
	pickle.dump(models_data_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

split = 0  # count splits in data for results_df

for idx in models_data_idx:

	# XGBoost needs encoded data as inputs
	# extract X_train, X_test, y_train, y_test from samples
	x_y_dict = get_x_y(idx, df, unique_samples)
	X_train, X_test, X_val, y_train, y_test, y_val = x_y_dict['X_train'], x_y_dict['X_test'], x_y_dict['X_val'],\
														x_y_dict['y_train'], x_y_dict['y_test'], x_y_dict['y_val']

	sets = {'train': y_train, 'test': y_test, 'val': y_val}

	# count train/ test instances in each split
	val_counts_df = record_val_counts(val_counts_df, sets, 'xgboost', split)

	for balanced in ['balanced', None]:

		# train model
		xgb_model = xgb.XGBClassifier(objective='multi:softmax',
									  n_jobs=JOBS,
									  random_state=0,
									  eval_metric='mlogloss',
									  use_label_encoder=False)

		param_grid_xgb = {'max_depth': list(range(5, 14)),  # set
							'learning_rate': [x / 1000 for x in range(1, 100)],
							'gamma': [x / 1000 for x in range(1, 100)],  # create range 0.001, 1
							'reg_lambda': list(range(0, 100))}

		start = time.time()
		clf_xgb = RandomizedSearchCV(xgb_model,
								param_grid_xgb,
								n_iter=10,
								verbose=1,
								scoring='f1_weighted',
								n_jobs=JOBS)
		xgb_gridsearch_time = time.time() - start

		if balanced == None:
			sample_weights = None
		else:
			sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

		start = time.time()
		clf_xgb.fit(X_train,
					y_train,
					verbose=150,
					eval_set=[(X_val, y_val)],
					sample_weight=sample_weights,
					early_stopping_rounds=10)
		xgb_train_time = time.time() - start

		start = time.time()
		y_pred = clf_xgb.predict(X_test)
		xgb_pred_time = time.time() - start

		# performance metrics
		xgb_f1_micro = f1_score(y_test, y_pred, average='micro')
		xgb_f1_macro = f1_score(y_test, y_pred, average='macro')
		xgb_f1_weighted = f1_score(y_test, y_pred, average='weighted')
		xgb_acc = accuracy_score(y_test, y_pred)

		# confusion matrices
		cm_norm_none_xgb = confusion_matrix(y_test, y_pred, labels=clf_xgb.classes_, normalize=None)
		cm_norm_true_xgb = confusion_matrix(y_test, y_pred, labels=clf_xgb.classes_, normalize='true')
		disp_norm_none_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_norm_none_xgb, display_labels=[inv_label_encoder[c] for c in clf_xgb.classes_])
		disp_norm_true_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_norm_true_xgb, display_labels=[inv_label_encoder[c] for c in clf_xgb.classes_])

		xgb_cm =  {
			'cm_norm_none': cm_norm_none_xgb,
			'cm_norm_true': cm_norm_true_xgb,
			'disp_norm_none': disp_norm_none_xgb,
			'disp_norm_true': disp_norm_true_xgb,
		}

		# add results to df.
		d = {'f1_micro': [xgb_f1_micro], 'f1_macro': [xgb_f1_macro], 'f1_weighted': [xgb_f1_weighted], 'acc': [xgb_acc], 
			'gridsearch_time' : [xgb_gridsearch_time], 'train_time': [xgb_train_time], 'pred_time': [xgb_pred_time],
			'model': ['xgboost'], 'split': [split], 'balanced': [balanced],'gamma': [clf_xgb.best_params_['gamma']], 'learning_rate': [clf_xgb.best_params_['learning_rate']],
			'max_depth': [clf_xgb.best_params_['max_depth']], 'reg_lambda': [clf_xgb.best_params_['reg_lambda']]}
		results_df = pd.concat([results_df, pd.DataFrame(data=d)])

		# save model
		path = '.../models/'
		with open(path + "xgb_" + str(balanced) + "_" + str(split) + '.pickle', 'wb') as handle:
			pickle.dump(clf_xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# save conf matrix
		path = '.../conf_matrix/'
		with open(path + "xgb_" + str(balanced) + "_" + str(split) + "_cm" + '.pickle', 'wb') as handle:
			pickle.dump(xgb_cm, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print('Boosting: splits complete: ' + str(split) + " / " + str(10) + " | Balanced : " + str(balanced))

	# save xgb train data
	path = '.../data/'
	with open(path + "xgb_train_" + str(split) + '.pickle', 'wb') as handle:
		pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

	split += 1  # increment split counter

with open(".../f1.pickle", 'wb') as handle:
	pickle.dump(results_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(".../val_counts_df.pickle", 'wb') as handle:
	pickle.dump(val_counts_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
