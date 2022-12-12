#!/usr/bin/env python
# coding: utf-8

# # Building a rule-based chatbot

# In[1]:


import nltk
from nltk.chat.util import Chat, reflections


# In[2]:


set_pairs = [
    [
      r"my name is (.*)",
        ["Hello %1, How are you today?"]
    ],
    [
        r"Hi|Hey|Hello",
        ["Hello","Hey there"]
        
    ],
    [
        r"quit",
        ["Bye, Thanks for chatting :)", "It was nice talking with you"]
    ]
]


# In[3]:


def chatbot ():
    print("Hi, I am a rule-based chatbot!How may I help you")


# In[4]:


chat = Chat(set_pairs,reflections)
chat


# In[5]:


chat.converse()
if __name__ == "__main__":
    chatbot()


# # Retrieval-based chatbot

# In[6]:


import nltk
import numpy as np
import random
import string

import bs4 as bs
import requests
import re

import warnings
warnings.filterwarnings = False


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[8]:


nltk.download ("punkt")
nltk.download ("wordnet")


# ## Gathering data from Wikipedia

# In[9]:


r = requests.get ('https://en.wikipedia.org/wiki/Cuisine')
raw_html = r.text


# In[10]:


## cleaning up html data 
corpus_html = bs.BeautifulSoup(raw_html)

## Extracting paragraphs from cleaned html data
corpus_paragraphs = corpus_html.find_all ('p')
corpus_text = ''

## Concantenating all paragraphs
for paragraph in corpus_paragraphs:
    corpus_text += paragraph.text
    
## Normalize text in lower case
corpus_text = corpus_text.lower()


# In[11]:


corpus_text


# In[12]:


## Getting rid of empty spaces and special characters
corpus_text = re.sub(r'\ [[0-9]*\]',' ', corpus_text)
corpus_text = re.sub(r'\s+', ' ', corpus_text)


# In[13]:


## sentence_tokenize and word_tokenize corpus text
corpus_sentences = nltk.sent_tokenize (corpus_text)
corpus_words = nltk.word_tokenize (corpus_text)


# In[14]:


corpus_sentences
#corpus_words


# ## Generating greeting responses on predeefined et of inputs

# In[15]:


greeting_inputs = ['hey', 'hello', 'good morning', 'good afternoon', 'how far', 'whatsup', 'how you dey']
greeting_responses = ['hey', 'hello', 'good morning', 'good afternoon', 'how far', 'whatsup', 'how you dey']

def greet_response (greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)


# ## Preprocessing (punctuation removal and lemmatization)

# In[16]:


wn_lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_corpus(tokens):
    return [wn_lemmatizer.lemmatize(token) for token in tokens]
punct_removal_dict = dict((ord(punctuation), None) for punctuation in string.punctuation)

def get_processed_text (document):
    return lemmatize_corpus(nltk.word_tokenize(document.lower().translate(punct_removal_dict)))


# ## Language modelling with tf-idf

# In[17]:


def respond (user_input):
    bot_response = ''
    corpus_sentences.append(user_input)
    
    ## vectorizing the processed text
    word_vectorizer = TfidfVectorizer (tokenizer=get_processed_text, stop_words = 'english')
    corpus_word_vectors = word_vectorizer.fit_transform(corpus_sentences)
    
    cos_sim_vectors = cosine_similarity(corpus_word_vectors [-1], corpus_word_vectors)
    similar_response_idx = cos_sim_vectors.argsort()[0][-2]
    
    matched_vector = cos_sim_vectors.flatten()
    matched_vector.sort()
    vector_matched = matched_vector [-2]
    
    if vector_matched == 0:
        bot_response = bot_response + "I am sorry, what is it, again?"
        return bot_response
    else:
        bot_response = bot_response + corpus_sentences [similar_response_idx]
        return bot_response


# In[ ]:


chat = True
print ("Hello, what do you want to learn about cuisine today?")
while (chat == True) :
    user_query = input()
    user_query  = user_query.lower()
    if user_query != 'quit':
        if user_query == 'thanks' or user_query == 'thank you':
            chat =False
            print("CuisineBot: You are welcome!")
        else:
            if greet_response(user_query) != None:
                 print ("CuisineBo: " + greet_response(user_query))
            else:
                print("CuisineBot: ", end = " ")
                print(respond(user_query))
                corpus_sentences.remove(user_query)
    else:
        chat = False
        print("Cuisinebot: Goodbye")
                


# In[ ]:




