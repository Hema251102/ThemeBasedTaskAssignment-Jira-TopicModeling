import os
import json
import gzip
import numpy as np
from properties import datafolder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

all_assignees = [5, 10, 15, 20]
num_assignees = 5
num_topics = [4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
projects = ["AMBARI", "ARROW", "CASSANDRA", "CB", "DATALAB", "FLINK", "GEODE", "HDDS", "IGNITE", "IMPALA", "MESOS", "OAK"]

optimal_num_topics = {}
for project in projects:
	for n_assignees in (all_assignees if project == "FLINK" else [num_assignees]):
		with gzip.open(os.path.join(datafolder, "5_" + project + "_" + str(n_assignees) + "_assignees" + "_results.json.gz"), 'r') as infile:
			results = json.loads(infile.read().decode('utf-8'))
		yclasses = results["classes"]
		ytest = results["y_test"]
		accuracies = []
		for n_topics in num_topics:
			y_pred = results[str(n_topics)]["SVM"]["topics"]["y_pred"]
			y_score = results[str(n_topics)]["SVM"]["topics"]["y_pred_proba"]
			label_binarizer = LabelBinarizer().fit(yclasses)
			y_true = label_binarizer.transform(ytest)
			y_pred = label_binarizer.transform(y_pred)
			accuracies.append(accuracy_score(y_true, y_pred))
		optimal_num_topics[project + "_" + str(n_assignees) + "_assignees"] = num_topics[np.argmax(accuracies)]
with open(os.path.join(datafolder, "6_optimal_num_topics.json"), 'w') as outfile:
	json.dump(optimal_num_topics, outfile, indent = 3)
