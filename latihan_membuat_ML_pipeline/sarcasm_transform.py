
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_text as tf_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
#             "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
#             "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
#             "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
#             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", 
#             "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", 
#             "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
#             "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", 
#             "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", 
#             "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", 
#             "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
#             ]

LABEL_KEY = "is_sarcastic"
FEATURE_KEY = "headline"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

# def _la

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    
    outputs = {}
    
    # outputs[transformed_name(FEATURE_KEY)] = tft.tfidf(inputs[FEATURE_KEY], vocab_size=1000)
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    # tokenizer =  Tokenizer(num_words=1000, oov_token="<OOV>")
    # tokenizer.fit_on_texts(inputs[FEATURE_KEY])
    
    # sequences = tokenizer.texts_to_sequences(inputs[FEATURE_KEY])
    # outputs[transformed_name(FEATURE_KEY)] = pad_sequences(
    #     sequences,maxlen=120,
    #     padding="post",
    #     truncating="post"
    # )
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs

