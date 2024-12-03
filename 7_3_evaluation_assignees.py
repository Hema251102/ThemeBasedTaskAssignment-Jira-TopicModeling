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

num_assignees = 5
project = "CASSANDRA"
model_key = 'topics'
topics = [4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]

# Produce Figure 6 of paper
accuracy = []
for num_topics in topics:
	with gzip.open(os.path.join(datafolder, "5_" + project + "_" + str(num_assignees) + "_assignees" + "_results.json.gz"), 'r') as infile:
		results = json.loads(infile.read().decode('utf-8'))
	yclasses = results["classes"]
	ytest = results["y_test"]
	
	y_pred = results[str(num_topics)]["SVM"][model_key]["y_pred"]
	y_score = results[str(num_topics)]["SVM"][model_key]["y_pred_proba"]
	label_binarizer = LabelBinarizer().fit(yclasses)
	y_true = label_binarizer.transform(ytest)
	y_pred = label_binarizer.transform(y_pred)
	accuracy.append(accuracy_score(y_true, y_pred))

fig, ax = plt.subplots(figsize=(4.4, 2.8))
ax.plot(topics, accuracy, '-')
#ax.set_xticks(np.arange(4, 21))
#ax.set_xticklabels(np.arange(4, 21))
ax.set_xlabel("Number of Topics")
ax.set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(resultsdatafolder, "TopicsAccuracy.eps"))
plt.savefig(os.path.join(resultsdatafolder, "TopicsAccuracy.pdf"))
#plt.show()
