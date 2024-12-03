import os
import json
import gzip
import numpy as np
from properties import datafolder, resultsdatafolder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

with open(os.path.join(datafolder, "6_optimal_num_topics.json")) as infile:
	optimal_num_topics = json.load(infile)

num_assignees = 5
projects = ["AMBARI", "ARROW", "CASSANDRA", "CB", "DATALAB", "FLINK", "GEODE", "HDDS", "IGNITE", "IMPALA", "MESOS", "OAK"]
model_keys = ['title', 'title_description', 'title_description_labels', 'title_description_topics', 'all']
model_titles = ['Title', 'Title & Desc.', 'Title & Desc. & Labels', 'Title & Desc. & Topics', 'All']

# Produce Tables 3, 4 and 5 of paper
print("Accuracies")
with open(os.path.join(resultsdatafolder, 'Accuracies.txt'), 'w') as outfile:
	for project in projects:
		with gzip.open(os.path.join(datafolder, "5_" + project + "_" + str(num_assignees) + "_assignees" + "_results.json.gz"), 'r') as infile:
			results = json.loads(infile.read().decode('utf-8'))
		yclasses = results["classes"]
		ytest = results["y_test"]
		print(project, end="")
		outfile.write(project)
		for model_key, model in zip(model_keys, model_titles):
			num_topics = optimal_num_topics[project + "_" + str(num_assignees) + "_assignees"]
			y_pred = results[str(num_topics)]["SVM"][model_key]["y_pred"]
			y_score = results[str(num_topics)]["SVM"][model_key]["y_pred_proba"]
			label_binarizer = LabelBinarizer().fit(yclasses)
			y_true = label_binarizer.transform(ytest)
			y_pred = label_binarizer.transform(y_pred)
			print(" & %.2f" %(accuracy_score(y_true, y_pred)), end='')
			outfile.write(" & %.2f" %(accuracy_score(y_true, y_pred)))
		print("\\\\")
		outfile.write("\\\\\n")

print("\nFmeasures")
with open(os.path.join(resultsdatafolder, 'Fmeasures.txt'), 'w') as outfile:
	for project in projects:
		with gzip.open(os.path.join(datafolder, "5_" + project + "_" + str(num_assignees) + "_assignees" + "_results.json.gz"), 'r') as infile:
			results = json.loads(infile.read().decode('utf-8'))
		yclasses = results["classes"]
		ytest = results["y_test"]
		print(project, end="")
		outfile.write(project)
		for model_key, model in zip(model_keys, model_titles):
			num_topics = optimal_num_topics[project + "_" + str(num_assignees) + "_assignees"]
			y_pred = results[str(num_topics)]["SVM"][model_key]["y_pred"]
			y_score = results[str(num_topics)]["SVM"][model_key]["y_pred_proba"]
			label_binarizer = LabelBinarizer().fit(yclasses)
			y_true = label_binarizer.transform(ytest)
			y_pred = label_binarizer.transform(y_pred)
			print(" & %.2f" %(f1_score(y_true, y_pred, average="micro")), end='')
			outfile.write(" & %.2f" %(f1_score(y_true, y_pred, average="micro")))
		print("\\\\")
		outfile.write("\\\\\n")

print("\nAUCs")
with open(os.path.join(resultsdatafolder, 'AUCs.txt'), 'w') as outfile:
	for project in projects:
		with gzip.open(os.path.join(datafolder, "5_" + project + "_" + str(num_assignees) + "_assignees" + "_results.json.gz"), 'r') as infile:
			results = json.loads(infile.read().decode('utf-8'))
		yclasses = results["classes"]
		ytest = results["y_test"]
		print(project, end="")
		outfile.write(project)
		for model_key, model in zip(model_keys, model_titles):
			num_topics = optimal_num_topics[project + "_" + str(num_assignees) + "_assignees"]
			y_pred = results[str(num_topics)]["SVM"][model_key]["y_pred"]
			y_score = results[str(num_topics)]["SVM"][model_key]["y_pred_proba"]
			label_binarizer = LabelBinarizer().fit(yclasses)
			y_true = label_binarizer.transform(ytest)
			y_pred = label_binarizer.transform(y_pred)
			print(" & %.2f" %(roc_auc_score(y_true, y_score, multi_class="ovr", average="micro")), end='')
			outfile.write(" & %.2f" %(roc_auc_score(y_true, y_score, multi_class="ovr", average="micro")))
		print("\\\\")
		outfile.write("\\\\\n")
