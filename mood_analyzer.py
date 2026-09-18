from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Небольшой учебный датасет
texts = [
    "я очень счастлив",
    "сегодня прекрасный день",
    "мне нравится этот проект",
    "я доволен результатом",
    "все отлично",
    "это было здорово",
    "я рад",
    "мне очень понравилось",
    "я грущу",
    "сегодня плохой день",
    "мне не нравится",
    "я разочарован",
    "все ужасно",
    "я расстроен",
    "это было плохо",
    "мне грустно",
    "обычный день",
    "я иду в университет",
    "сейчас занимаюсь учебой",
    "сегодня обычная погода",
    "я читаю книгу",
    "у меня занятия",
]

labels = [
    "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative",
    "neutral", "neutral", "neutral", "neutral",
    "neutral", "neutral",
]

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)

print("🤖 AI Mood Analyzer")
print("Введите текст. Для выхода напишите: exit")

while True:
    user_text = input("\nВаш текст: ").strip()

    if user_text.lower() == "exit":
        print("До встречи!")
        break

    if not user_text:
        print("Введите текст.")
        continue

    prediction = model.predict([user_text])[0]
    probabilities = model.predict_proba([user_text])[0]
    confidence = max(probabilities) * 100

    print(f"Настроение: {prediction}")
    print(f"Уверенность модели: {confidence:.1f}%")
