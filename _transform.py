
import tensorflow as tf
import tensorflow_transform as tft


NUMERICAL_FEATURES = ["depth", "carat", "table", "x", "y", "z"]
CATEGORICAL_FEATURES = ["clarity", "color", "cut"]

LABEL_KEY = "price"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transfomed features.    
    """
    
    outputs = {}
    
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])
    
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(inputs[feature])
    
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    
    return outputs
