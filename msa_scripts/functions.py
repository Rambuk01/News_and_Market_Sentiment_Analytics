import torch
from transformers import pipeline
import requests
import re

# Build a function. Needs to return the sentiment of a list of sentenses.
def getSentiment(sentences: list):
    print(f"Initiating sentiment analysis.")
    classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
    result = classifier(sentences)    
    return result


def get_notes_from_the_underground():
    """
    This function retrieves *Notes from the Underground* from the Gutenberg
    website. It also does some text cleaning and separates each sentence into
    list elements.
    """
    url = 'https://www.gutenberg.org/cache/epub/600/pg600.txt'
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        text = response.text

        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|\r\n\r\n', text)
        # Clean text
        sentences = sentences[11:]
        sentences = sentences[:-117]
        return sentences
    else:
        raise Exception(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
    
    return text # Never happens?

def get_positive_score(x):
    """
    The pipeline returns 'POSITIVE' or 'NEGATIVE' and a probability, where the
    label is based on what is the most likely sentiment of the sentence. It
    turns out to be useful to have one continuous score from -1 to 1, which
    captures completely 'postive' if 1 and completely 'negative' if -1. This
    function handles that.
    """
    if x["label"] == "POSITIVE":
        res_x = x['score']
    elif x["label"] == "NEGATIVE":
        res_x = 1 - x['score']
    else:
        raise Exception(x["label"]+"This should not be possible")

    res_x = res_x*2-1 # Expand to -1 to 1 scale

    return res_x


from transformers import pipeline
classifier = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english", device=0)

result = classifier('This makes me feel like an apple. Consindering the work they put into it. A banana is like home.')
print(result)

#python -c "from transformers import pipeline; classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=0); print(classifier('I hate you'))"
