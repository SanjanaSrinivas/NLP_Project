import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

def get_ner_vectors(sents):

    feat_vecs = []
    for sent in sents:
        feat_vec = []
        sent_doc = nlp(sent)
        entities = [X.label_ for X in sent_doc.ents]
        counts = Counter(entities)
        feat_vec.append(counts['PERSON'])
        feat_vec.append(counts['NORP'])
        feat_vec.append(counts['FAC'])
        feat_vec.append(counts['ORG'])
        feat_vec.append(counts['GPE'])
        feat_vec.append(counts['LOC'])
        feat_vec.append(counts['PRODUCT'])
        feat_vec.append(counts['EVENT'])
        feat_vec.append(counts['DATE'])
        feat_vec.append(counts['TIME'])
        feat_vec.append(counts['PERCENT'])
        feat_vec.append(counts['MONEY'])
        feat_vec.append(counts['QUANTITY'])
        feat_vec.append(counts['ORDINAL'])
        feat_vec.append(counts['CARDINAL'])
        feat_vecs.append(feat_vec)

    return feat_vecs


# PERSON	People, including fictional.
# NORP	Nationalities or religious or political groups.
# FAC	Buildings, airports, highways, bridges, etc.
# ORG	Companies, agencies, institutions, etc.
# GPE	Countries, cities, states.
# LOC	Non-GPE locations, mountain ranges, bodies of water.
# PRODUCT	Objects, vehicles, foods, etc. (Not services.)
# EVENT	Named hurricanes, battles, wars, sports events, etc.
# WORK_OF_ART	Titles of books, songs, etc.
# LAW	Named documents made into laws.
# LANGUAGE	Any named language.
# DATE	Absolute or relative dates or periods.
# TIME	Times smaller than a day.
# PERCENT	Percentage, including "%".
# MONEY	Monetary values, including unit.
# QUANTITY	Measurements, as of weight or distance.
# ORDINAL	"first", "second", etc.
# CARDINAL


if __name__ == "__main__":

    sents = [u"My thoughts and prays go out to the city's in michigan that where hit by the floods",
			 u"My whole room flooded man fuck",
			 u"New @mousegrip @landslidesk8 #mousegrip #mousemovement in stock while sprays last @ Landslide Skate Park http://t.co/ivJ0teSpXs",
			 u"No @katyperry concert for me! This flooding is ridiculous!",
			 u"No band camp today due to flooding"]

    feat_vecs = get_ner_vectors(sents)

    for sent, feat_vec in zip(sents, feat_vecs):
        print sent, feat_vec