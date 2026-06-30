from imports import *
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]

df.columns = ['label', 'message']

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=3000
)

X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)