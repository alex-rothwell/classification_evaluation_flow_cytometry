import pickle
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from adjustText import adjust_text
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm
from statsmodels.formula.api import ols


def save_fig_types(fig, save_dir, name):

	try:
		fig.savefig(save_dir + name + ".png", dpi=300)

	except:
		fig = fig.get_figure()
		fig.savefig(save_dir + name + ".png", dpi=300)


def tp_fp_fn_tn_from_cm(cm):
	# calculate true pos, false pos, false neg, true neg from confusion matrices
	# https://stackoverflow.com/questions/47899463/how-to-extract-false-positive-false-negative-from-a-confusion-matrix-of-multicl

	n_classes = len(cm)

	tp = np.diag(cm)

	fp = []
	for i in range(n_classes):  # loop over classes
		fp.append(sum(cm[:, i]) - cm[i, i])

	fn = []
	for i in range(n_classes):
		fn.append(sum(cm[i, :]) - cm[i, i])

	tn = []
	for i in range(n_classes):
		temp = np.delete(cm, i, 0)  # delete ith row
		temp = np.delete(temp, i, 1)  # delete ith column
		tn.append(sum(sum(temp)))

	return tp, fp, fn, tn


def conf_matrix_labels(mean, std):
	"""
	function for np. vectorise which returns labels for conf_matrices
	:param mean: array
	:param std: array
	:return: array of labels
	"""
	return str(round(mean, 2)) + "\n" + "(" + u"\u00B1" + " " + str(round(std, 2)) + ")"


"""
paths
"""
cm_bm_path = r"D:\Git\classification_evaluation_flow_cytometry\data\output"  # confusion matrices for benchmarking
f1_bm_path = r"D:\Git\classification_evaluation_flow_cytometry\data\output\results_df.pickle"  # result df for benchmarking
cm_cfs_path = r"D:\Git\classification_evaluation_flow_cytometry\data\conf_matrix"  # confusion matrices for clf's
f1_cfs_path = r"D:\Git\classification_evaluation_flow_cytometry\data\f1.pickle"  # result df for clf's

fig_dir = r"D:\Git\classification_evaluation_flow_cytometry\figures\figs\\"  # dir to save figs to

"""
plot configs
"""

# color palette
color_p = 'mako'

# set style
sns.set_context("paper")

sns.set(font_scale=2)

# Make the background white, and specify the specific font family
sns.set_style("white", {"font.serif": "Helvetica"})

plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['axes.linewidth'] = 2.5

"""
clean
"""

bm_df = pd.read_pickle(f1_bm_path)
clf_df = pd.read_pickle(f1_cfs_path)

bm_df['method'].replace({'flowmeans': 'flowMeans',
					'flowgrid': 'FlowGrid',
					'flowsom': 'flowSOM',
					'cytometree': 'cytometree',
					'samspectral': 'samSPECTRAL',
					'rclusterpp': 'Rclusterpp',
					'flock': 'FLOCK'}, inplace=True)

bm_df.rename(columns={'time': 'pred_time'}, inplace=True)

clf_df.rename(columns={'model': 'method'}, inplace=True)

clf_df['method'] = clf_df['method'].replace({'DT': 'CART',
							'RF': 'Random Forest',
							'xgboost': 'XGBoost'})

comb_df = pd.concat([bm_df, clf_df])

comb_df['balanced'] = comb_df['balanced'].replace({'balanced': 'Balanced Training Weights', None: 'No Balancing'})

# ordered methods by highest mean f1 macro
all_f1_ordered_methods = comb_df.groupby(by='method').mean().sort_values(by='f1_macro').index.to_list()

# keep only clfs
clf_f1_ordered_methods = [m for m in all_f1_ordered_methods if m in ['XGBoost', 'CART', 'Random Forest']]

# change hue of clf and clustering methods
comb_df_method_hue = comb_df.copy()
comb_df_method_hue.loc[~comb_df_method_hue['method'].isin(clf_f1_ordered_methods), 'balanced'] = 'Clustering Method'
comb_df_method_hue['balanced'] = \
	comb_df_method_hue['balanced'].replace({'No Balancing': 'CLF - No Balancing',
											'Balanced Training Weights': 'CLF - Balanced Training Weights'})

"""
f1 - fig 1
"""

f1_plot = sns.catplot(data=comb_df_method_hue,
					kind="bar",
					x="method",
					y="f1_macro",
					ci="sd",
					palette=sns.color_palette('mako', 3),
					hue="balanced",
					height=9,
					aspect=2.5,
					legend=False,  # easier to do in matplotlib below
					linewidth=2.5,
					edgecolor='black',
					capsize=.15,
					errwidth=2.5,
					errcolor='0',
					order=all_f1_ordered_methods
					)
f1_plot.despine(left=False)
f1_plot.set_axis_labels("", "F1")
plt.legend(loc="best",
					fancybox=False,
					framealpha=1,  # opaque
					facecolor='w',
					edgecolor='0')
plt.tight_layout()
save_fig_types(f1_plot, fig_dir, "f1_bar")

"""
time taken - fig 3
"""

# prediction time
pred_plot = sns.catplot(data=comb_df_method_hue,
					kind="bar",
					x="method",
					y="pred_time",
					ci="sd",
					palette=sns.color_palette('mako', 3),
					hue="balanced",
					height=9,
					aspect=2.5,
					legend=False,  # easier to do in matplotlib below
					linewidth=2.5,
					edgecolor='black',
					capsize=.15,
					errwidth=2.5,
					errcolor='0',
					order=all_f1_ordered_methods
					)
pred_plot.despine(left=False)
pred_plot.set_axis_labels("", "Prediction time (s)")
plt.legend(loc="best",
					fancybox=False,
					framealpha=1,  # opaque
					facecolor='w',
					edgecolor='0')
plt.ylim(bottom=0)  # don't allow error bars off screen
plt.tight_layout()
save_fig_types(pred_plot, fig_dir, "pred_bar")

"""
training time - fig 2
"""

train_plot = sns.catplot(data=comb_df[comb_df['method'].isin(["CART", "Random Forest", "XGBoost"])],
					kind="bar",
					x="method",
					y="train_time",
					ci="sd",
					palette=sns.color_palette(color_p, 2),
					hue="balanced",
					height=9,
					aspect=1,
					legend=False,  # easier to do in matplotlib below
					linewidth=2.5,
					edgecolor='black',
					capsize=.15,
					errwidth=2.5,
					errcolor='0',
					order=clf_f1_ordered_methods
					)
train_plot.despine(left=False)
train_plot.set_axis_labels("", "Training time (s)")
plt.legend(loc="best",
					fancybox=False,
					framealpha=1,  # opaque
					facecolor='w',
					edgecolor='0')
plt.tight_layout()
save_fig_types(train_plot, fig_dir, "train_bar")

"""
scatter f1 vs prediction time - fig 4
"""
# add new column indicating if classification or clustering
grouped_comb_df = comb_df.groupby(["method", "balanced"]).mean()

method_type = []
for idx, row in grouped_comb_df.iterrows():
	method = idx[0]
	if method in clf_f1_ordered_methods:
		method_type.append('Classification algorithm')
	else:
		method_type.append('Clustering method')

grouped_comb_df['Method Type'] = method_type

dims = (9, 9)
fig, ax = plt.subplots(figsize=dims)
f1_time_scatter = sns.scatterplot(data=grouped_comb_df,
								x="f1_macro",
								y="pred_time",
								ax=ax,
								palette=sns.color_palette(color_p, 2),
								color="black",
								style="Method Type")
f1_time_scatter.spines.right.set_visible(False)
f1_time_scatter.spines.top.set_visible(False)
f1_time_scatter.set_xlabel("F1 score")
f1_time_scatter.set_ylabel("Prediction time (s)")

# add name of method as annotation to each point
annotations = []

for idx, row in comb_df.groupby(["method", "balanced"]).mean().iterrows():

	if idx[0] in ["CART", "Random Forest", "XGBoost"]:
		# if clf then add whether balanced or not
		annotations.append(plt.text(x=row["f1_macro"],
								y=row["pred_time"],
								s=idx[0] + "-" + idx[1],
								fontsize=11))

	else:
		# else don't add if balanced or not
		annotations.append(plt.text(x=row["f1_macro"],
								y=row["pred_time"],
								s=idx[0],
								fontsize=11))

plt.xlim(right=1)  # show all of f1
adjust_text(annotations, arrowprops={'arrowstyle': '-', 'color': 'black'})

plt.tight_layout()
plt.legend(loc="best",
					fancybox=False,
					framealpha=1,  # opaque
					facecolor='w',
					edgecolor='0',
					fontsize=15)  # reduce font size
save_fig_types(f1_time_scatter, fig_dir, "f1_pred_scatter")

"""
calculate recall and precision

classifiers - needs calculating from confusion matrices
benchmarking - calculated at the time
"""
# classifiers

os.chdir(cm_cfs_path)

cm_cfs_files = os.listdir(".")

prf1_df = pd.DataFrame()

for cm_pickle in cm_cfs_files:

	method = cm_pickle.split("_")[0]

	if method == 'dt':
		method = 'CART'
	elif method == 'rf':
		method = 'Random Forest'
	elif method == 'xgb':
		method = 'XGBoost'

	balanced = cm_pickle.split("_")[1]
	if balanced == 'None':
		balanced = 'No Balancing'
	elif balanced == 'balanced':
		balanced = 'Balanced Training Weights'

	split = int(cm_pickle.split("_")[2])

	with open(cm_pickle, 'rb') as handle:
		cm = pickle.load(handle)

	cm = cm['cm_norm_none']

	tp, fp, fn, tn = tp_fp_fn_tn_from_cm(cm)

	precision = np.nan_to_num(tp / (tp + fp))
	recall = np.nan_to_num(tp / (tp + fn))

	precision_macro = np.mean(precision)
	recall_macro = np.mean(recall)

	# get from clf df
	f1_macro = comb_df[(comb_df['method'] == method) &
						(comb_df['balanced'] == balanced) &
						(comb_df['split'] == split)]['f1_macro'].values[0]

	prf1_df = pd.concat([prf1_df,
							pd.DataFrame(data={'method': method,
												'balanced': balanced,
												'split': split,
												'sample': "-",
												'rep': "-",
												'precision': precision_macro,
												'recall': recall_macro,
												'f1_macro': f1_macro}, index=[0])])

# benchmarking
prf1_df = pd.concat([prf1_df, pd.DataFrame(data={'method': bm_df['method'],
												'balanced': "No Balancing",
												'split': "-",
												'sample': bm_df['sample'],
												'rep': bm_df['rep'],
												'precision': bm_df['p_macro'],
												'recall': bm_df['r_macro'],
												'f1_macro': bm_df['f1_macro']})])

# if clf method, rename method separating out into balanced and unbalanced, else if clustering method then don't

prf1_df['method_bal'] = prf1_df['method'] + " - " + prf1_df['balanced']
prf1_df['method'] = prf1_df['method_bal'].apply(lambda x: x.split(" - ")[0] if x.split(" - ")[0] not in clf_f1_ordered_methods else x)

# shorten long names, need to do str.replace() to get substrings.
prf1_df['method'] = prf1_df['method'].str.replace('CART', 'CART')
prf1_df['method'] = prf1_df['method'].str.replace('Random Forest', 'RF')
prf1_df['method'] = prf1_df['method'].str.replace('XGBoost', 'XGB')
prf1_df['method'] = prf1_df['method'].str.replace('Balanced Training Weights', 'BTW')
prf1_df['method'] = prf1_df['method'].str.replace('No Balancing', 'NB')

prf1_df_m = pd.melt(prf1_df, id_vars=['method', 'balanced', 'split', 'sample', 'rep'],
					value_vars=['precision', 'recall', 'f1_macro'])

prf1_df_m['variable'].replace({'precision': 'Precision',
								'recall': 'Recall',
								'f1_macro': 'F1'},
								inplace=True)

# ordered methods by highest mean f1 macro
all_f1_ordered_methods_prf1 = prf1_df.groupby(by='method').mean().sort_values(by='f1_macro').index.to_list()

"""
f1, recall, precision - SF 1
"""
f1_plot = sns.catplot(data=prf1_df_m,
					kind="bar",
					x="method",
					y="value",
					ci="sd",
					palette=sns.color_palette(color_p, 3),
					hue="variable",
					height=9,
					aspect=1.5,
					legend=False,  # easier to do in matplotlib below
					linewidth=2.5,
					edgecolor='black',
					capsize=.15,
					errwidth=2.5,
					errcolor='0',
					order=all_f1_ordered_methods_prf1
					)
f1_plot.despine(left=False)
f1_plot.set_axis_labels("", "")
plt.legend(loc="best",
					fancybox=False,
					framealpha=1,  # opaque
					facecolor='w',
					edgecolor='0')
f1_plot.set_xticklabels(rotation=90)
plt.tight_layout()
save_fig_types(f1_plot, fig_dir, "f1_prec_rec")

"""
plot confusion matrices
Supplementary Figures and fig 5
"""

# make dictionary of confusion matrices, key for each model, value, list with each conf matrix
conf_matrices = {}  # normalised (for plots)

# not normalised, for when calculating f1, recall, prec per class later
values_conf_matrices = {}

# clfs

os.chdir(cm_cfs_path)

cm_cfs_files = os.listdir(".")

# order labels are in
label_order = ['cd4pos', 'dualpos', 'dualneg', 'cd8pos', 'gdt', 'kappa', 'lambd', 'nk', 'notlymph']

for cm_pickle in cm_cfs_files:

	method = cm_pickle.split("_")[0]

	if method == 'dt':
		method = 'CART'
	elif method == 'rf':
		method = 'Random Forest'
	elif method == 'xgb':
		method = 'XGBoost'

	balanced = cm_pickle.split("_")[1]
	if balanced == 'None':
		balanced = 'No Balancing'
	elif balanced == 'balanced':
		balanced = 'Balanced Training Weights'

	split = int(cm_pickle.split("_")[2])

	with open(cm_pickle, 'rb') as handle:
		cm = pickle.load(handle)

	# if key in dict, append to list, if not, then create list
	if method + "-" + balanced in conf_matrices:
		# add to 3rd dimension of array, so conf_matrices with n 3rd dims
		conf_matrices[method + "-" + balanced] = np.dstack((conf_matrices[method + "-" + balanced], cm['disp_norm_true'].confusion_matrix))

		# get unnormalised values for calculating precision and recall.
		values_conf_matrices[method + "-" + balanced] = np.dstack((values_conf_matrices[method + "-" + balanced], cm['disp_norm_none'].confusion_matrix))
	else:
		conf_matrices[method + "-" + balanced] = cm['disp_norm_true'].confusion_matrix

		values_conf_matrices[method + "-" + balanced] = cm['disp_norm_none'].confusion_matrix

# benchmarking

os.chdir(cm_bm_path)

cm_bm_files = os.listdir(".")

cm_bm_files = [f for f in cm_bm_files if f.endswith("cm.pickle")]

for cm_pickle in cm_bm_files:

	cm_string = cm_pickle.replace("predictions_", "")

	method = cm_string.split("_")[0]

	if method == 'flock':
		method = 'FLOCK'
	elif method == 'flowgrid':
		method = 'FlowGrid'
	elif method == 'flowmeans':
		method = 'flowMeans'
	elif method == 'flowsom':
		method = 'flowSOM'
	elif method == 'rclusterpp':
		method = 'Rclusterpp'
	elif method == 'samspectral':
		method = 'samSPECTRAL'

	sample = cm_string.split("_")[1]
	rep = cm_string.split("_")[2]

	with open(cm_pickle, 'rb') as handle:
		cm = pickle.load(handle)

	# if key in dict, append to list, if not, then create list
	if method in conf_matrices:
		# add to 3rd dimension of array, so conf_matrices with n 3rd dims
		conf_matrices[method] = np.dstack((conf_matrices[method], cm['disp_norm_true_nonan'].confusion_matrix))
	else:
		conf_matrices[method] = cm['disp_norm_true_nonan'].confusion_matrix

# calculate mean + sd cm's by looping over all conf_matrices
mean_sd_cm = {}

# ensure than ticklabels are in the same order as order that conf_matrices are
display_labels = {'cd4pos': 'CD4 +',
					'dualpos': 'Dual +',
					'dualneg': 'Dual -',
					'cd8pos': 'CD8 +',
					'gdt': 'G/D T',
					'kappa': 'Kappa',
					'lambd': 'Lambda',
					'nk': 'NK',
					'notlymph': 'Not Lymph'}

ticklabels = [display_labels[l] for l in label_order]

heatmap_dir = fig_dir + "conf_matrices\\"

for method, cms in conf_matrices.items():
	mean_sd_cm[method] = {'mean': np.mean(cms, axis=2),  # on 3rd axis, i.e. across samples for each element of cm
							'std': np.std(cms, axis=2)}

	# create annotation in format of mean plus minus sd
	labels_func = np.vectorize(conf_matrix_labels)
	mean_sd_labels = labels_func(mean_sd_cm[method]['mean'], mean_sd_cm[method]['std'])

	dims = (11, 11)
	fig, ax = plt.subplots(figsize=dims)
	sns.heatmap(mean_sd_cm[method]['mean'],
				annot=mean_sd_labels,
				fmt='',  # required for non numeric labels
				annot_kws={"fontsize": 13},
				cmap=sns.color_palette("mako_r", as_cmap=True),
				xticklabels=ticklabels,
				yticklabels=ticklabels,
				square=True,
				ax=ax,
				vmin=0,
				vmax=1)
	plt.xlabel("Predicted label")
	plt.ylabel("True label")
	plt.title(method)
	plt.tight_layout()
	save_fig_types(fig, heatmap_dir, method)

"""
Confusion matrix subplot
Figure 5
"""
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(25, 25), sharex=False, sharey=False)

axs = np.array(axs).reshape(-1)

cbar_ax = fig.add_axes([.91, .3, .03, .4])

# loop over calculated cms
for i in range(len(mean_sd_cm.items())):
	model = list(mean_sd_cm.keys())[i]
	cm = mean_sd_cm[model]['mean']
	ax = axs[i]

	# replace with shorthand
	model = model.replace("Random Forest", "RF")
	model = model.replace("XGBoost", "XGB")
	model = model.replace("Balanced Training Weights", "BTW")
	model = model.replace("No Balancing", "NB")

	ax.title.set_text(model)

	sns.heatmap(cm,
				cmap=sns.color_palette("mako_r", as_cmap=True),
				xticklabels=ticklabels,
				yticklabels=ticklabels,
				square=True,
				ax=ax,
				vmin=0,
				vmax=1,
				cbar= i == 0,
				cbar_ax=None if i else cbar_ax)

# remove empty axes
for j in range(len(mean_sd_cm.items()), 16):
	axs[j].axis("off")

fig.tight_layout(rect=[0, 0, .9, 1])
save_fig_types(fig, heatmap_dir, 'subplot')

"""
f1, precision, recall, per class, per algorithm

columns - class
rows - algorithm

Table 1, Supplementary Table 1 and 2.
"""

# benchmarking
per_class_metrics = bm_df.drop(columns=['f1_micro', 'f1_macro', 'f1_weighted', 'acc', 'p_micro',
										'p_macro', 'p_weighted', 'r_micro', 'r_macro', 'r_weighted',
										'pred_time', 'success'])

# loop over confusion matrices dict, calculating metrics per class for clf methods
for method, cms in values_conf_matrices.items():

	# only do calculations if cm is from classifier
	if any(clf in method for clf in clf_f1_ordered_methods):

		# call each CV split 'rep' for ease
		for rep in range(np.shape(cms)[2]):  # the 3rd dimension of cms, is number of splits that were done
			# do actual calculations
			cm_for_rep = cms[:, :, rep]
			d = {'method': method, 'sample': '', 'rep': rep}  # data to add to df

			# loop over the conf matrix and get name of cell to add to df and idx of position.
			for idx, label in enumerate(label_order):

				# https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal

				tp = cm_for_rep[idx, idx]
				fp = cm_for_rep[:, idx].sum() - tp
				fn = cm_for_rep[idx, :].sum() - tp
				tn = cm_for_rep.sum() - (tp + fp + fn)

				precision = tp / (tp + fp)
				recall = tp / (tp + fn)
				f1 = 2 * (precision * recall) / (precision + recall)

				d['p_' + label] = precision
				d['r_' + label] = recall
				d['f1_' + label] = f1

			# add data to df
			per_class_metrics = pd.concat([per_class_metrics, pd.DataFrame(data=d, index=[0])])

method_type = []
for idx, row in per_class_metrics.iterrows():
	if any(clf in row['method'] for clf in clf_f1_ordered_methods):
		method_type.append('Classification algorithm')
	else:
		method_type.append('Clustering method')

per_class_metrics['method_type'] = method_type
per_class_metrics = per_class_metrics.drop(columns=['sample', 'rep'])

# calculate stats
per_class_mean = per_class_metrics.groupby(by=['method', 'method_type'], as_index=False).mean()
per_class_sd = per_class_metrics.groupby(by=['method', 'method_type'], as_index=False).std()

per_class_mean = per_class_mean.set_index('method')
per_class_sd = per_class_sd.set_index('method')

for df in [per_class_mean, per_class_sd]:
	df.loc['Mean: Clustering method'] = df[df['method_type'] == 'Clustering method'].mean(axis=0)
	df.loc['Mean: Classification algorithm'] = df[df['method_type'] == 'Classification algorithm'].mean(axis=0)
	df.loc['Mean: All'] = df.mean(axis=0)

per_class_mean = per_class_mean.drop(columns=['method_type'])
per_class_sd = per_class_sd.drop(columns=['method_type'])

# loop over dataframes, separating out into individual dfs for f1, prec and recall
df_dict = {'f1_mean': pd.DataFrame(index=per_class_mean.index),
			'f1_sd': pd.DataFrame(index=per_class_sd.index),
			'p_mean': pd.DataFrame(index=per_class_mean.index),
			'p_sd': pd.DataFrame(index=per_class_sd.index),
			'r_mean': pd.DataFrame(index=per_class_mean.index),
			'r_sd': pd.DataFrame(index=per_class_sd.index)
			}

for m_sd, df in zip(['mean', 'sd'], [per_class_mean, per_class_sd]):
	for col in df.columns:

		df_type = col.split("_")[0]
		label = col.split("_")[1]

		# get display label from dict
		fancy_label = display_labels[label]

		# add to correct df
		df_dict[df_type + "_" + m_sd][fancy_label] = df[col]

# then calc overall metric column
for df in df_dict.values():
	df['Overall'] = df.mean(axis=1)

# then sort index in order of decreasing f1
sorted_f1_idx_table = list(df_dict['f1_mean'].sort_values(by='Overall', ascending=False).index)
subtotals = ['Mean: Clustering method', 'Mean: Classification algorithm', 'Mean: All']

idx_tables = [i for i in sorted_f1_idx_table if i not in subtotals] + subtotals

# sort indexes on all tables
for k, v in df_dict.items():
	df_dict[k] = v.reindex(index=idx_tables)

# format tables mean + sd
f1_df = df_dict['f1_mean'].applymap(lambda x: str(round(x, 2))) + " (± " + df_dict['f1_sd'].applymap(lambda x: str(round(x, 2))) + ")"
r_df = df_dict['r_mean'].applymap(lambda x: str(round(x, 2))) + " (± " + df_dict['r_sd'].applymap(lambda x: str(round(x, 2))) + ")"
p_df = df_dict['p_mean'].applymap(lambda x: str(round(x, 2))) + " (± " + df_dict['p_sd'].applymap(lambda x: str(round(x, 2))) + ")"

f1_df.to_csv(fig_dir + 'f1_df.csv', encoding="utf-8-sig")
r_df.to_csv(fig_dir + 'r_df.csv', encoding="utf-8-sig")
p_df.to_csv(fig_dir + 'p_df.csv', encoding="utf-8-sig")

"""
Two-way anova
"""

for_anova_df = clf_df[['method', 'f1_macro', 'balanced']]
for_anova_df['balanced'] = for_anova_df['balanced'].replace({None: 'not balanced'})

model = ols('f1_macro ~ C(method) + C(balanced) + C(method):C(balanced)', data=for_anova_df).fit()

anova_results = sm.stats.anova_lm(model, typ=2)
