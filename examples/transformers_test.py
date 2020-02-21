from transformers import pipeline
nlp = pipeline('sentiment-analysis')
print(nlp('We are very happy to include pipeline into the transformers repository.'))