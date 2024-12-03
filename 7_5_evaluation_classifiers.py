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
projects = ["AMBARI", "ARROW", "CASSANDRA", "CB", "DATALAB", "FLINK", "GEODE", "HDDS", "IGNITE", "IMPALA", "MESOS", "OAK"]
model_key = ['all']

# Produce Figure 9 of paper
accuracy_svm, accuracy_bayes = [], []
fmeasure_svm, fmeasure_bayes = [], []
auc_svm, auc_bayes = [], []
for project in projects:
	with gzip.open(os.path.join(datafolder, "5_" + project + "_" + str(num_assignees) + "_assignees" + "_results.json.gz"), 'r') as infile:
		results = json.loads(infile.read().decode('utf-8'))
	yclasses = results["classes"]
	ytest = results["y_test"]
	for model in ["SVM", "NaiveBayes"]:
		num_topics = optimal_num_topics[project + "_" + str(num_assignees) + "_assignees"]
		y_pred = results[str(num_topics)][model]['all']["y_pred"]
		y_score = results[str(num_topics)][model]['all']["y_pred_proba"]
		label_binarizer = LabelBinarizer().fit(yclasses)
		y_true = label_binarizer.transform(ytest)
		y_pred = label_binarizer.transform(y_pred)
		if model == 'SVM':
			accuracy_svm.append(accuracy_score(y_true, y_pred))
			fmeasure_svm.append(f1_score(y_true, y_pred, average='micro'))
			auc_svm.append(roc_auc_score(y_true, y_score, multi_class="ovr", average="micro"))
		else:
			accuracy_bayes.append(accuracy_score(y_true, y_pred))
			fmeasure_bayes.append(f1_score(y_true, y_pred, average='micro'))
			auc_bayes.append(roc_auc_score(y_true, y_score, multi_class="ovr", average="micro"))

metrics = ["Accuracy", "F-measure", "AUC"]
svmMeanAccuracy, svmStdAccuracy = np.mean(accuracy_svm), np.std(accuracy_svm)
bayesMeanAccuracy, bayesStdAccuracy = np.mean(accuracy_bayes), np.std(accuracy_bayes)
svmMeanFmeasure, svmStdFmeasure = np.mean(fmeasure_svm), np.std(fmeasure_svm)
bayesMeanFmeasure, bayesStdFmeasure = np.mean(fmeasure_bayes), np.std(fmeasure_bayes)
svmMeanAUC, svmStdAUC = np.mean(auc_svm), np.std(auc_svm)
bayesMeanAUC, bayesStdAUC = np.mean(auc_bayes), np.std(auc_bayes)

index = np.arange(len(metrics))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(4.4, 2.8))
rects1 = plt.bar(index, [svmMeanAccuracy, svmMeanFmeasure, svmMeanAUC], bar_width, label='SVM', yerr = [svmStdAccuracy, svmStdFmeasure, svmStdAUC], capsize=3)
rects2 = plt.bar(index + bar_width, [bayesMeanAccuracy, bayesMeanFmeasure, bayesMeanAUC], bar_width, label='Naïve Bayes', yerr = [bayesStdAccuracy, bayesStdFmeasure, bayesStdAUC], capsize=3)
ax.set_ylim(0, 1)
#ax.set_xlabel('Metrics')
ax.set_ylabel('Value')
ax.set_xticks(index + bar_width/2, metrics)#, rotation = 45, ha='right')
ax.set_xlim(-0.66, 2.9)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(resultsdatafolder, "ClassificationModels.png"))
plt.savefig(os.path.join(resultsdatafolder, "ClassificationModels.pdf"))
#plt.show()
