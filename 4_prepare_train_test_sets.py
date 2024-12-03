import os
import pandas as pd
import gensim
import gensim.corpora as corpora
from numpy import savetxt
from properties import datafolder, num_topics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns

for f in os.listdir(datafolder):
	if f.startswith("3"):
		_, project_name, num_assignees, _ = f.split("_")
		# Create directory if it does not exist
		if not os.path.exists(os.path.join(datafolder, "4_" + project_name + "_" + str(num_assignees) + "_assignees")):
			os.makedirs(os.path.join(datafolder, "4_" + project_name + "_" + str(num_assignees) + "_assignees"))

		# Read data and split to training and test sets
		df = pd.read_csv(os.path.join(datafolder, "3_" + project_name + "_" + str(num_assignees) + "_assignees" + ".csv"), sep='\t', encoding='utf-8')
		y = df[["assignee_id"]]
		df.drop(['assignee_id'], axis=1)
		X = df
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

		# Extract topics from titles and descriptions
		corpus_tokens = [str(td).split() for td in (X_train['title'] + ' ' + X_train['description']).to_list()]
		# Filter out tokens that appear < 30 times, and tokens that appear in more than 20% of the documents
		dictionary = corpora.Dictionary(corpus_tokens)
		dictionary.filter_extremes(no_below=30, no_above=0.2)
		# Term Document Frequency
		corpus = [dictionary.doc2bow(text) for text in corpus_tokens]
		# Build LDA model for each topic configuration
		for topics_number in num_topics:
			lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topics_number,
														random_state=100, update_every=1, alpha='auto', per_word_topics=True)

			for xdatastr, xdata in [("X_train", X_train), ("X_test", X_test)]:
				# Compute data per issue
				topic_per_issue, topterms_per_issue, distribution_per_issue = [], [], []
				for row, issue in xdata.iterrows():
					issuetext = (str(issue[2]) + ' ' + str(issue[3])).split()
					issuebow = dictionary.doc2bow(issuetext)
					issuetopics = lda_model.get_document_topics(issuebow)
					issuetopics_full = []
					for itopic in range(topics_number):
						prob = [issuetopic[1] for issuetopic in issuetopics if issuetopic[0] == itopic]
						issuetopics_full.append((itopic, prob[0] if len(prob) > 0 else 0.0))
					distribution_per_issue.append([issuetopic[1] for issuetopic in issuetopics_full])
					topic_per_issue.append(sorted(issuetopics_full, key = lambda x: -x[1])[0][0])
					topterms_per_issue.append([j[0] for j in lda_model.show_topic(topic_per_issue[-1], topn=10)])
				xdata["topic_id"] = topic_per_issue
				xdata["top_terms"] = topterms_per_issue
				xdata_distribution = pd.DataFrame(distribution_per_issue, columns = ['topic_' + str(i) for i in range(topics_number)])
				# Save configuration to disk
				xdata.drop(xdata.columns[xdata.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
				xdata.to_csv(os.path.join(os.path.join(datafolder, "4_" + project_name + "_" + str(num_assignees) + "_assignees"), xdatastr + "_" + str(topics_number) + "_topics.csv"), sep='\t', encoding='utf-8')
				xdata_distribution.drop(xdata_distribution.columns[xdata_distribution.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
				xdata_distribution.to_csv(os.path.join(os.path.join(datafolder, "4_" + project_name + "_" + str(num_assignees) + "_assignees"), xdatastr + "_distribution_" + str(topics_number) + "_topics.csv"), sep='\t', encoding='utf-8')

		# Save also y to disk
		for ydatastr, ydata in [("y_train", y_train), ("y_test", y_test)]:
			ydata.drop(ydata.columns[ydata.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
			ydata.to_csv(os.path.join(os.path.join(datafolder, "4_" + project_name + "_" + str(num_assignees) + "_assignees"), ydatastr + ".csv"), sep='\t', encoding='utf-8')
