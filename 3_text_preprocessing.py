# Apply text preprocessing techniques to title and description text data

import os
import re
import spacy
import numpy
import pandas as pd
from spacy.tokens import Doc
from spacy.matcher import Matcher
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA
from properties import datafolder

try:
	nlp = spacy.load("en_core_web_md")
except:
	import spacy.cli
	spacy.cli.download("en_core_web_md")

# Patterns for re
TAG_RE = re.compile(r'<[^>]+>')		 # HTML or XML tags


# Create the list of words that spaCy will recognize as STOP WORDS. A new word can be added
# like this: nlp.Defaults.stop_words |= {"new_stopword1", "new_stopword2",}
# To see the current list of stop words, use print(STOP_WORDS)
# To remove a STOP WORD, use nlp.Defaults.stop_words.remove("new_stopword1")


def remove_span(doc, indeces):
	"""Remove the desired token from doc by providing the indeces"""
	np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA])
	np_array_2 = numpy.delete(np_array, (indeces), axis=0)
	doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(
		doc) if not (i in indeces)])
	doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA, LEMMA], np_array_2)
	return doc2


def remove_html_xml_tags(text):
	"""Removes HTML and XML tags"""
	return TAG_RE.sub('', text)


def remove_unwanted_characters(text):
	"""Removes unwanted characters that were not removed with other methods"""
	pattern = r'[#=<>|$!@#%^&*(){}\?+]'
	return re.sub(pattern, '', text)


def remove_words_with_less_than_n_characters(text):
	"""Removes words with less than n characters"""
	return ' '.join(word for word in text.split() if (len(word) > 3 and len(word) < 20))


def text2cleanDoc(text):
	"""
	Applies stop-word and punctuation removal 
	Input: string
	Output: spaCy Doc
	"""
	doc = nlp(text)
	doc_cleaned = [token for token in doc if not token.is_stop and not token.is_punct]
	return doc_cleaned


def text2cleanDoc_onlyNouns(text):
	"""
	Applies stop-word and punctuation removal, keeps only nouns 
	Input: string
	Output: spaCy Doc
	"""
	doc = nlp(text)
	doc_only_nouns = [token for token in doc if not token.is_stop and not token.is_punct and token.pos_ == "NOUN"]
	return doc_only_nouns


def text2cleanDoc_matchPatterns(text):
	"""
	Applies tokenization, stop-word removal and lemmatization
	Input: string
	Output: spaCy Doc
	"""
	# Apply basic techniques on the string, before creating the doc
	text = '"""' + text + '"""'
	text = remove_html_xml_tags(text)
	text = remove_words_with_less_than_n_characters(text)
	text = remove_unwanted_characters(text)
	doc = nlp(text)
	# Define the patterns for the Matcher to identify
	matcher = Matcher(nlp.vocab)
	patternURL = [{'LIKE_URL': True}]
	patternSpaces = [{"TEXT": {"REGEX": "\\s+"}}]
	patternCharacters = [{"TEXT": {"REGEX": "<"}}]
	patternLikeNumbers = [{'LIKE_NUM': True}]
	matcher.add('URL', [patternURL])
	matcher.add('Spaces', [patternSpaces])
	matcher.add('Numbers', [patternLikeNumbers])
	matcher.add('Characters', [patternCharacters])

	# Find the matches in doc and keep the indeces of the matched tokens
	matches = matcher(doc)
	indeces = []
	for match in matches:
		indeces.append(match[1])
	# Create a new doc without the removed tokens, keeping all the other information
	doc2 = remove_span(doc, indeces)
	# Remocve stop words and punctuation
	doc_cleaned = [token for token in doc2 if not token.is_stop and not token.is_punct]
	return doc_cleaned


def doc2lemmatizedStrings(doc):
	"""
	Returns a string containing the lemmatized tokens (lowercase strings) of the doc it takes as argument
	Input: spaCy Doc
	Output: list of tokens (strings)
	"""
	tokens = [token.lemma_.lower() for token in doc]
	return ' '.join(tokens)

for f in os.listdir(datafolder):
	if f.startswith("2"):
		_, project_name, num_assignees, _ = f.split("_")
		# Read df from step 2
		df = pd.read_csv(os.path.join(datafolder, "2_" + project_name + "_" + str(num_assignees) + "_assignees" + ".csv"), sep='\t', encoding='utf-8')

		# Delete all rows where description or title is empty
		df = df[(df.description.notna() & (df.title.notna()))]

		# Descriptions and titles convert to strings
		df['description'] = df['description'].astype(str)
		df['title'] = df['title'].astype(str)

		# Preprocess title and description: convert them to lists and pass them through functions
		df.description = [doc2lemmatizedStrings(text2cleanDoc_matchPatterns(description)) for description in df['description'].to_list()]
		df.title = [doc2lemmatizedStrings(text2cleanDoc_matchPatterns(title)) for title in df['title'].to_list()]

		# Drop unnamed column
		df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

		# Save df
		df.to_csv(os.path.join(datafolder, "3_" + project_name + "_" + str(num_assignees) + "_assignees" + ".csv"), sep='\t', encoding='utf-8')