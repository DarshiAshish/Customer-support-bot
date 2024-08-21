import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from tensorflow.keras.models import load_model

def ember_v1_model():
    model = SentenceTransformer('llmrails/ember-v1')
    return model

import json
with open("responses.json","r") as f:
    response_data = json.load(f)

ann_model = load_model('customer_bot_model.h5')
print("Succesful loading of ann model")
model = ember_v1_model()
print("EMber v1 model")


def test_cases():
    test_cases = ["I want to cancel the order", "Let me know how to change my address", "List me all the payment methods that you will follow to process a payment", "working hours for customer support","delivery status", "how to get a refund", "how to write a review", "how to change to an other account"]
    test_final_array = []
    for each_one in test_cases:
        test_final_array.append(model.encode(each_one))

    test_final_array = np.array(test_final_array)
    pred_labels = ann_model.predict(test_final_array)
    predicted_indices = np.argmax(pred_labels, axis=1)
    print(predicted_indices)
    for each in predicted_indices:
        print(each)
        print(response_data[str(each)])
        print("-------------------------------------------------------------------")
    

# test_cases()

def custom_input_test():
    input_case = input("Enter the query")
    test_cases = [input_case]
    test_final_array = []
    for each_one in test_cases:
        test_final_array.append(model.encode(each_one))

    test_final_array = np.array(test_final_array)
    pred_labels = ann_model.predict(test_final_array)
    predicted_indices = np.argmax(pred_labels, axis=1)
    print(predicted_indices)
    print(response_data[str(predicted_indices[0])])
    
custom_input_test()