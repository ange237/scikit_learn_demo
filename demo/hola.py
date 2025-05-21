import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('demo/amazon_cells_labelled.txt',
names=[ 'review', 'sentiment'], sep='\t')



reviews = df['review'].values
sentiments = df['sentiment'].values
reviews_train, reviews_test, sentiment_train, sentiment_test = train_test_split(reviews,
sentiments, test_size=0.2, random_state=500)


vectorizer = CountVectorizer()
vectorizer.fit(reviews)
X_train = vectorizer.transform(reviews_train)
X_test = vectorizer.transform(reviews_test)


classifier = LogisticRegression()
classifier.fit(X_train, sentiment_train)

accuracy = classifier.score(X_test, sentiment_test)
print("Accuracy:", accuracy)

"""Si inizia creando una lista di nuove recensioni, quindi si trasforma il nuovo testo in vettori
di caratteristiche numeriche. Infine, si prevede il sentiment della classe per i nuovi
campioni"""
new_reviews = ['Old version of python useless', 'Very good effort, but not five stars', 'Clear and concise']
X_new = vectorizer.transform(new_reviews)
print(classifier.predict(X_new))