import matplotlib.pyplot as plt
import seaborn
import pandas as pd

"""
plot predictions/ labels
"""


def plot_predictions(f, PATH_TO_OUTPUT, PATH_TO_INPUT):
	# get data from input
	sample = f.split("_")[2]

	# load original data
	orig_df = pd.read_csv(PATH_TO_INPUT + "/" + sample + ".csv")

	df = orig_df.drop(columns=["label"])  # just leave original data

	# add predictions as label
	predictions = pd.read_csv(PATH_TO_OUTPUT + "/" + f)
	df['label'] = predictions

	save_name = f.split(".")[0]
	plot_scatter(df, save_name, PATH_TO_INPUT)


def plot_labels(f, PATH_TO_INPUT):

	df = pd.read_csv(PATH_TO_INPUT + "/" + f)
	save_name = 'labels_' + f[0:4]
	plot_scatter(df, save_name, PATH_TO_INPUT)


def plot_scatter(df, save_name, PATH_TO_INPUT):

	# plot lst plots of data

	plot_path = PATH_TO_INPUT.replace("input_data", "plots")

	# drop nans
	sample_df = df.dropna()

	# font
	plt.rcParams["font.family"] = "Calibri"

	# fig size
	plt.figure(figsize=(12, 8))

	plt.suptitle(save_name)

	plt.subplot(2, 3, 1)
	# CD45 SSC-A
	gate_1 = sample_df.copy()
	c_1 = gate_1['label'].replace(to_replace={'cd4pos': '#1e84d4',
														'cd8pos': '#1e84d4',
														'kappa': '#1e84d4',
														'lambd': '#1e84d4',
														'gdt': '#1e84d4',
														'nk': '#1e84d4',
														'dualpos': '#1e84d4',
														'dualneg': '#1e84d4',
														'notlymph': '#323234'})
	s = plt.scatter(x=gate_1['CD45 V500-C-A'], y=gate_1['SSC-A'], c=c_1, s=0.1, marker='.', edgecolors=None)
	plt.title("All events")
	plt.xlabel('CD45')
	plt.ylabel('SSC-A')

	plt.subplot(2, 3, 2)
	# CD19 CD3
	# lymphocytes
	gate_2 = sample_df[~sample_df['label'].str.contains('notlymph')]
	c_2 = gate_2['label'].replace(to_replace={'kappa': '#db9e23',
															'lambd': '#db9e23',
															'cd4pos': '#772b14',
															'cd8pos': '#772b14',
															'dualpos': '#772b14',
															'dualneg': '#772b14',
															'gdt': '#FF69B4',
															'nk': '#45b5aa'})
	plt.scatter(x=gate_2['CD19+TCRgd PE-Cy7-A'], y=gate_2['CD3 APC-A'], c=c_2, s=0.1, marker='.', edgecolors=None)
	plt.title("Lymphocytes")
	plt.xlabel('CD19')
	plt.ylabel('CD3')

	plt.subplot(2, 3, 3)
	# FSC-A SSC-A
	# all events
	plt.scatter(x=gate_1['FSC-A'], y=gate_1['SSC-A'], c=c_1, s=0.1, marker='.', edgecolors=None)
	plt.title("All Events")
	plt.xlabel('FSC-A')
	plt.ylabel('SSC-A')

	plt.subplot(2, 3, 4)
	# Lambda/CD8 CD20/CD4
	# t cells
	gate_3 = sample_df[(sample_df['label'] == 'cd4pos') |
						(sample_df['label'] == 'cd8pos') |
						(sample_df['label'] == 'dualpos') |
						(sample_df['label'] == 'dualneg')]
	c_3 = gate_3['label'].replace(['dualpos', 'dualneg', 'cd4pos', 'cd8pos'],
								['#7EC0EE', '#273F87', '#FFA500', '#009900'])
	plt.scatter(x=gate_3['Lambda_CD8 FITC-A'], y=gate_3['CD20+CD4 V450-A'], c=c_3, s=0.1, marker='.', edgecolors=None)
	plt.title("T cells")
	plt.xlabel('Lambda/CD8')
	plt.ylabel('CD20/CD4')

	plt.subplot(2, 3, 5)
	# Kappa/CD56 Lambda/CD8
	# b cells
	gate_4 = sample_df[(sample_df['label'] == 'kappa') |
						(sample_df['label'] == 'lambd')]
	c_4 = gate_4['label'].replace(['lambd', 'kappa'], ['#d94f70', '#1e84d4'])
	plt.scatter(x=gate_4['Kappa_CD56 PE-A'], y=gate_4['Lambda_CD8 FITC-A'], c=c_4, s=0.1, marker='.', edgecolors=None)
	plt.title("B cells")
	plt.xlabel('Kappa/CD56')
	plt.ylabel('Lambda/CD8')

	plt.subplot(2, 3, 6)
	# Kappa/CD56 CD3
	# NOT b cells
	gate_5 = gate_2[(gate_2['label'] == 'nk') |
					(gate_2['label'] == 'gdt') |
					(gate_2['label'] == 'cd4pos') |
					(gate_2['label'] == 'cd8pos') |
					(gate_2['label'] == 'dualpos') |
					(gate_2['label'] == 'dualneg')]
	c_5 = gate_5['label'].replace(to_replace={'cd4pos': '#cccc51',
														'cd8pos': '#cccc51',
														'dualpos': '#cccc51',
														'dualneg': '#cccc51',
														'gdt': '#cccc51',
														'nk': '#864755'})
	plt.scatter(x=gate_5['Kappa_CD56 PE-A'], y=gate_5['CD3 APC-A'], c=c_5, s=0.1, marker='.', edgecolors=None)
	plt.title("NOT B cells")
	plt.xlabel('Kappa/CD56')
	plt.ylabel('CD3')

	plt.tight_layout()
	# room for title
	plt.subplots_adjust(top=0.9)
	plt.savefig(plot_path + "/" + save_name + ".png")
	plt.close()
