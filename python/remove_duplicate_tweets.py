import difflib
import pandas

input_file_path = "data_bfr_modified.csv"

newTweets = []
with open(input_file_path) as f:
    tweets = [l.lower() for l in f.readlines()]
    print len(tweets)
    for i in range(0, len(tweets)):
        is_duplicate = False
        for j in range(0, i):
            if difflib.SequenceMatcher(None, tweets[i], tweets[j]).ratio() > 0.8:
                is_duplicate = True
                break
        if is_duplicate == False:
            newTweets.append(tweets[i].strip().replace('"', ''))
    print len(newTweets)

df = pandas.DataFrame(newTweets)
df.to_csv('data_modified.csv')