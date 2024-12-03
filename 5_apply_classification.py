import os
import gzip
import json
import zipfile
import numpy as np
import pandas as pd
from numpy import savetxt
from properties import datafolder, num_topics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def apply_tf_idf(training_corpus, test_corpus):
	cv = CountVectorizer(max_features=1000, min_df=5)
	training_vect = cv.fit_transform(training_corpus.values.astype('U')).toarray()
	test_vect = cv.transform(test_corpus.values.astype('U')).toarray()
	tfidfconverter = TfidfTransformer()
	training_freq = tfidfconverter.fit_transform(training_vect).toarray()
	test_freq = tfidfconverter.transform(test_vect).toarray()
	return training_freq, test_freq

def apply_svc(X_train, y_train, X_test):
	model = CalibratedClassifierCV(LinearSVC())
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	y_pred_proba = model.predict_proba(X_test)
	return y_pred.tolist(), y_pred_proba.tolist(), model.classes_.tolist()

def apply_naive_bayes(X_train, y_train, X_test):
	model = MultinomialNB()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	y_pred_proba = model.predict_proba(X_test)
	return y_pred.tolist(), y_pred_proba.tolist(), model.classes_.tolist()

def combine_predictions(weights, probabilities, classes):
	'''Combines the outputs of the classifiers using different weights'''
	combined_probabilities = sum([np.array(prob) * weight for prob, weight in zip(probabilities, weights)]) / sum(weights)
	indeces = np.argmax(combined_probabilities, axis=1)
	y_pred = [classes[indeces[j]] for j in range(len(indeces))]
	return y_pred, combined_probabilities.tolist()

for f in os.listdir(datafolder):
	if f.startswith("4"):
		_, project_name, num_assignees, _ = f.split("_")
		projectfolder = os.path.join(datafolder, "4_" + project_name + "_" + str(num_assignees) + "_assignees")
		with gzip.open(os.path.join(datafolder, "5_" + project_name + "_" + str(num_assignees) + "_assignees" + "_results.json.gz"), 'w') as outfile:
			results = {}
			y_train = pd.read_csv(os.path.join(projectfolder, "y_train.csv"), sep='\t', encoding='utf-8')
			y_train.drop(y_train.columns[y_train.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
			y_test = pd.read_csv(os.path.join(projectfolder, "y_test.csv"), sep='\t', encoding='utf-8')
			y_test.drop(y_test.columns[y_test.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
			y_train, y_test = y_train["assignee_id"], y_test["assignee_id"]
			for topics_number in num_topics:
				results[topics_number] = {}
				# Read data
				X_train = pd.read_csv(os.path.join(projectfolder, "X_train_" + str(topics_number) + "_topics.csv"), sep='\t', encoding='utf-8')
				X_train_distribution = pd.read_csv(os.path.join(projectfolder, "X_train_distribution_" + str(topics_number) + "_topics.csv"), sep='\t', encoding='utf-8')
				X_test = pd.read_csv(os.path.join(projectfolder, "X_test_" + str(topics_number) + "_topics.csv"), sep='\t', encoding='utf-8')
				X_test_distribution = pd.read_csv(os.path.join(projectfolder, "X_test_distribution_" + str(topics_number) + "_topics.csv"), sep='\t', encoding='utf-8')
				# Drop unnamed columns
				X_train.drop(X_train.columns[X_train.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
				X_train_distribution.drop(X_train_distribution.columns[X_train_distribution.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
				X_test.drop(X_test.columns[X_test.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
				X_test_distribution.drop(X_test_distribution.columns[X_test_distribution.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
				# Process labels and top terms
				X_train['labels'] = [re.sub('\[|\]|\'|', '', re.sub('\,|', '', labels)) for labels in X_train['labels']]
				X_train['top_terms'] = [re.sub('\[|\]|\'|', '', re.sub('\,|', '', top_terms)) for top_terms in X_train['top_terms']]
				X_train['labels_top_terms'] = X_train['labels'] + ' ' + X_train['top_terms']
				X_test['labels'] = [re.sub('\[|\]|\'|', '', re.sub('\,|', '', labels)) for labels in X_test['labels']]
				X_test['top_terms'] = [re.sub('\[|\]|\'|', '', re.sub('\,|', '', top_terms)) for top_terms in X_test['top_terms']]
				X_test['labels_top_terms'] = X_test['labels'] + ' ' + X_test['top_terms']
				# Split to different datasets
				X_train_title, X_test_title = apply_tf_idf(X_train['title'], X_test['title'])
				X_train_description, X_test_description = apply_tf_idf(X_train['description'], X_test['description'])
				X_train_labels, X_test_labels = apply_tf_idf(X_train['labels'], X_test['labels'])
				X_train_top_terms, X_test_top_terms = apply_tf_idf(X_train['top_terms'], X_test['top_terms'])
				X_train_labels_top_terms, X_test_labels_top_terms = apply_tf_idf(X_train['labels_top_terms'], X_test['labels_top_terms'])
				X_train_topics, X_test_topics = X_train_distribution, X_test_distribution
				X_train_other, X_test_other = X_train[['priority_id', 'type_id']], X_test[['priority_id', 'type_id']]
				# Classify using classifiers
				for classifier_name, classifier_function in [("SVM", apply_svc), ("NaiveBayes", apply_naive_bayes)]:
					results[topics_number][classifier_name] = {}
					y_pred, y_pred_proba, classes = classifier_function(X_train_title, y_train, X_test_title)
					results[topics_number][classifier_name]["title"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba, classes = classifier_function(X_train_description, y_train, X_test_description)
					results[topics_number][classifier_name]["description"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba, classes = classifier_function(X_train_labels, y_train, X_test_labels)
					results[topics_number][classifier_name]["labels"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba, classes = classifier_function(X_train_top_terms, y_train, X_test_top_terms)
					results[topics_number][classifier_name]["top_terms"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba, classes = classifier_function(X_train_labels_top_terms, y_train, X_test_labels_top_terms)
					results[topics_number][classifier_name]["labels_top_terms"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba, classes = classifier_function(X_train_topics, y_train, X_test_topics)
					results[topics_number][classifier_name]["topics"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba, classes = classifier_function(X_train_other, y_train, X_test_other)
					results[topics_number][classifier_name]["other"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					# Create combined models - choose the desired weights, using the following form: [title, description, labels, labels_terms, topics, other]
					probabilities = [results[topics_number][classifier_name]["title"]["y_pred_proba"], results[topics_number][classifier_name]["description"]["y_pred_proba"], \
									results[topics_number][classifier_name]["labels"]["y_pred_proba"], results[topics_number][classifier_name]["labels_top_terms"]["y_pred_proba"], \
									results[topics_number][classifier_name]["topics"]["y_pred_proba"], results[topics_number][classifier_name]["other"]["y_pred_proba"]]
					weights = [0.6, 0.7, 0.5, 0.5, 0.5, 0.1]
					y_pred, y_pred_proba = combine_predictions(weights[0:2], probabilities[0:2], classes)
					results[topics_number][classifier_name]["title_description"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba = combine_predictions(weights[0:3], probabilities[0:3], classes)
					results[topics_number][classifier_name]["title_description_labels"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba = combine_predictions(weights[0:2] + [weights[4]], probabilities[0:2] + [probabilities[4]], classes)
					results[topics_number][classifier_name]["title_description_labels_top_terms"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba = combine_predictions(weights[0:2] + [weights[5]], probabilities[0:2] + [probabilities[5]], classes)
					results[topics_number][classifier_name]["title_description_topics"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}
					y_pred, y_pred_proba = combine_predictions(weights, probabilities, classes)
					results[topics_number][classifier_name]["all"] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}

				if "classes" not in results: results["classes"] = classes
				if "y_test" not in results: results["y_test"] = y_test.tolist()
			outfile.write(json.dumps(results, sort_keys = False).encode('utf-8'))
