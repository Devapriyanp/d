import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


newdf = pd.read_csv(".csv")


newtext = newdf.iloc[:, 0].astype(str)   
newlabel = newdf.iloc[:, -1].astype(str) 


def clean_text(newtext):
    newtext = newtext.lower()
    newtext = re.sub(r'[^a-z\s]', '', newtext)
    newtext = re.sub(r'\s+', ' ', newtext)
    return newtext.strip()

newtext = newtext.apply(clean_text)
newtext = newtext.fillna("")


newlabel = pd.factorize(newlabel)[0]


vectorizer = TfidfVectorizer(stop_words='english')
newX = vectorizer.fit_transform(newtext)


newX_train, newX_test, newy_train, newy_test = train_test_split(
    newX, newlabel, test_size=0.3, stratify=newlabel, random_state=42
)


newmodel = MultinomialNB()
newmodel.fit(newX_train, newy_train)


newpred = newmodel.predict(newX_test)


print("Accuracy:", accuracy_score(newy_test, newpred))
print("Precision:", precision_score(newy_test, newpred, average='weighted', zero_division=0))
print("Recall:", recall_score(newy_test, newpred, average='weighted', zero_division=0))
print("F1-score:", f1_score(newy_test, newpred, average='weighted', zero_division=0))