import os
import json
import gzip
import numpy as np
import matplotlib.pyplot as plt
from properties import datafolder, resultsdatafolder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

with open(os.path.join(datafolder, "6_optimal_num_topics.json")) as infile:
	optimal_num_topics = json.load(infile)

all_assignees = [5, 10, 15, 20]
project = "FLINK"
model_keys = ['title', 'title_description', 'title_description_labels', 'title_description_topics', 'all']
model_titles = ['Title', 'Title & Desc.', 'Title & Desc. & Labels', 'Title & Desc. & Topics', 'All']

# Produce Figures 4 and 5 of paper
assignees = all_assignees
accuracy = []
fmeasure = []
for num_assignees in all_assignees:
	with gzip.open(os.path.join(datafolder, "5_" + project + "_" + str(num_assignees) + "_assignees" + "_results.json.gz"), 'r') as infile:
		results = json.loads(infile.read().decode('utf-8'))
	yclasses = results["classes"]
	ytest = results["y_test"]
	model_key = 'all'
	num_topics = optimal_num_topics[project + "_" + str(num_assignees) + "_assignees"]
	y_pred = results[str(num_topics)]["SVM"][model_key]["y_pred"]
	y_score = results[str(num_topics)]["SVM"][model_key]["y_pred_proba"]
	label_binarizer = LabelBinarizer().fit(yclasses)
	y_true = label_binarizer.transform(ytest)
	y_pred = label_binarizer.transform(y_pred)
	accuracy.append(accuracy_score(y_true, y_pred))
	fmeasure.append(f1_score(y_true, y_pred, average='micro'))

fig, ax = plt.subplots(figsize=(4.4, 2.8))
ax.plot(assignees, accuracy, 'o-')
ax.set_xticks(np.arange(4, 21))
ax.set_xticklabels(np.arange(4, 21))
ax.set_xlabel("Number of Assignees")
ax.set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(resultsdatafolder, "AccuracyAssignees.eps"))
plt.savefig(os.path.join(resultsdatafolder, "AccuracyAssignees.pdf"))

fig, ax = plt.subplots(figsize=(4.4, 2.8))
ax.plot(assignees, fmeasure, 'o-')
ax.set_xticks(np.arange(4, 21))
ax.set_xticklabels(np.arange(4, 21))
#ax.set_ylim(0.39, 0.6)
#ax.set_yticks(np.arange(0.4, 0.61, 0.05))
#ax.set_yticklabels(["%.2f" %s for s in np.arange(0.4, 0.61, 0.05)])
ax.set_xlabel("Number of Assignees")
ax.set_ylabel("F-measure")
plt.tight_layout()
plt.savefig(os.path.join(resultsdatafolder, "FmeasureAssignees.eps"))
plt.savefig(os.path.join(resultsdatafolder, "FmeasureAssignees.pdf"))
#plt.show()
