import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import emoji

class EnglishPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words('english')
        
    def preprocess_documents(self, documents_list: list, return_strings: bool = True):
        return [self.preprocess(doc, return_string=return_strings) for doc in documents_list]
    
    # Function to preprocess documents
    def preprocess(self, document, return_string: bool = True):
        # Tokenize
        document = self.remove_links_content(document)
        document = self.remove_emoji(document)
        document = self.remove_tags(document)
        document = self.remove_emails(document)
        document = self.remove_multiple_space(document)
        document = self.remove_hashtags(document)
        document = self.remove_punctuation(document)
        document = self.remove_multiple_space(document)
        
        words = word_tokenize(document.lower())
        # Remove stopwords and punctuations
        filtered_words = [word for word in words if word.isalnum() and not word in self.stop_words]
        # Lemmatize
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
        if return_string:
            return ' '.join(lemmatized_words)
        else:
            return lemmatized_words
    
    def remove_links_content(self, text):
        text = re.sub(r"http\S+", "", text)
        return text
    
    def remove_emails(self, text):
        return re.sub(r'\S+@\S*\s?', '', text)
    
    def remove_punctuation(self, text):
        """https://stackoverflow.com/a/37221663"""
        table = str.maketrans({key: None for key in string.punctuation})
        return text.translate(table)
    
    def remove_multiple_space(self, text):
        return re.sub(r"\s\s+", " ", text)

    def remove_hashtags(self, text):
        return re.sub(r'(?<=[\s\n])#\S+\s+', '', text)

    def remove_tags(self, text):
        return re.sub(r'[@&][\S]+', '', text)

    def remove_emoji(self, text):
        return emoji.replace_emoji(text,replace='')