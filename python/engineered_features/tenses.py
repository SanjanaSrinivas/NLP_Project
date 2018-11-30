import CMUTweetTagger as cmu
import numpy as np

def get_tense_vectors(sents):

	tagged_sents = cmu.runtagger_parse(sents)
	feat_vecs = []
	for tagged_sent in tagged_sents:
		feat_vec = []
		# past
		feat_vec.append(len([word for word in tagged_sent if word[1] in ['VBD', 'VBN']]))
		# present
		feat_vec.append(len([word for word in tagged_sent if (word[1] in ['VBP', 'VBG', 'VBZ'] or word[0] in ['now', 'Now', 'expected', 'Expected', 'about to', 'today', 'Today'])]))
		# future
		feat_vec.append(len([word for word in tagged_sent if word[1] == 'MD']))
		feat_vecs.append(feat_vec)
	return feat_vecs

if __name__ == "__main__":

	sents = ["My thoughts and prays go out to the city's in michigan that where hit by the floods",
			 "My whole room flooded man fuck",
			 "New @mousegrip @landslidesk8 #mousegrip #mousemovement in stock while sprays last @ Landslide Skate Park http://t.co/ivJ0teSpXs",
			 "No @katyperry concert for me! This flooding is ridiculous!",
			 "No band camp today due to flooding"]

	feat_vecs = get_tense_vectors(sents)

	print sents[1], feat_vecs[1]