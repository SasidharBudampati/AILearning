from transformers import pipeline

analysis = pipeline("sentiment-analysis")

text_to_review = [
    "The product is working excellent well",
    "Though I like the product features, it takes hell lot time to setup the product",
    "I may like to take coffee sometimes, but only specific coffee brands"
]

results = analysis(text_to_review)

for each_text, result in zip(text_to_review, results):
    print(f"Text for review : {each_text}")
    print(f"Review result : {result['label']} - \t Confidence : {result['score']:.2f}\n")