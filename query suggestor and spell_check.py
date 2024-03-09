"""Query Suggestor"""

import torch
from transformers import BertTokenizer, BertForMaskedLM
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Download NLTK resources
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

# Get English stopwords
english_stopwords = set(stopwords.words('english'))


# Define a function to predict next words
def predict_next_words(text, top_k=150):
    # Tokenize the input text
    tokenized_text = tokenizer.tokenize(text)
    
    # Add [CLS] and [SEP] tokens
    tokenized_text = ["[CLS]"] + tokenized_text + ["[SEP]"]
    
    # Convert tokens to IDs
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    # Mask the last token
    masked_index = len(indexed_tokens) - 1
    indexed_tokens[masked_index] = tokenizer.mask_token_id
    
    # Convert indexed tokens to tensor
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    # Generate predictions
    with torch.no_grad():
        outputs = model(tokens_tensor)

    # Get the predicted probabilities for the masked token
    predictions = outputs[0]
    predicted_probabilities = predictions[0, masked_index].cpu()

    # Get top-k predicted tokens
    top_k_probabilities, top_k_indices = predicted_probabilities.topk(top_k)

    # Convert token IDs back to words
    top_k_words = tokenizer.convert_ids_to_tokens(top_k_indices.tolist())

    # Filter out special tokens
    special_tokens = ['[CLS]', '[SEP]', '[MASK]', '[PAD]']
    top_k_words = [word for word in top_k_words if word not in special_tokens]

    # Filter out job field-related words
    filtered_words = [word for word in top_k_words if word.lower() not in english_stopwords ]

    # Filter out words based on heuristics to identify verbs, nouns, and adjectives
    final_filtered_words = []
    for word in filtered_words:
        if word.isalpha():  # Exclude words containing non-alphabetic characters
            if len(wordnet.synsets(word)) > 0:  # Check if the word exists in WordNet
                synset = wordnet.synsets(word)[0]
                pos_tag = synset.pos()
                if  pos_tag.startswith('n') :
                    final_filtered_words.append(word)
    

    return final_filtered_words


# Example usage
text = "machine learning and"
next_words = predict_next_words(text)
print("Next words:", next_words)

"""Spell Check"""

"""Use below code for our project

After careful consideration, i implemented logic to only English language and keep other languages the same without checking, as it is very complicated to include multiple languages for this task and only degenerates the task and system if i try including multiple languages.

I have used weighted systems which further enhance the task.
"""

from spellchecker import SpellChecker
from autocorrect import Speller
from langdetect import detect
from textblob import TextBlob

def weighted_spell_check_query(query):
    corrected_query = []
    # Split the query into individual terms
    terms = query.split()
    for term in terms:
        language = detect(term)
        if language == 'en':  # Check if the term is in English
            # Weighted spell check using multiple libraries
            corrected_term = weighted_spell_check_en(term)
        else:
            corrected_term = term  # Retain word if it's not in English
        corrected_query.append(corrected_term)
    return ' '.join(corrected_query)

def weighted_spell_check_en(term):
    # Weighted spell check using multiple libraries
    spellchecker = SpellChecker()
    autocorrect = Speller(lang='en')
    textblob = TextBlob(term)

    # Calculate weights for each library
    spellchecker_weight = 0.4
    autocorrect_weight = 0.3
    textblob_weight = 0.3

    # Spell check using each library
    spellchecker_correction = spellchecker.correction(term)
    autocorrect_correction = autocorrect(term)
    textblob_correction = str(textblob.correct())

    # Calculate weighted correction
    corrected_term = (
        spellchecker_weight * spellchecker_correction +
        autocorrect_weight * autocorrect_correction +
        textblob_weight * textblob_correction
    )

    return corrected_term

# Example usage
query = 'Mieuz vaut tard que jamais shhsh'
corrected_query = weighted_spell_check_query(query)
print(f"Suggested spelling for '{query}': {corrected_query}")

"""Speech Recognition"""

import speech_recognition as sr
recognizer = sr.Recognizer()

#Function to convert speech to text
def speechtotext():
    with sr.Microphone() as source:
        print("Listening...")
        #recognizer.adjustfor_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Listen to microphone input

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)  # Recognize speech using Google Speech Recognition
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error fetching results; {0}".format(e))
        
if __name__ == "__main__":
    while True:
        speech_text = speechtotext()
        if speech_text:
            print("You said:", speech_text)
            if speech_text.lower() == "stop":
                break
