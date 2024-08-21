import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

def ember_v1_model():
    model = SentenceTransformer('llmrails/ember-v1')
    return model

def read_data():
    import json
    import pandas as pd

    train_data = pd.read_csv("Bitext_Sample_Customer_Service_Training_Dataset.csv")
    test_data = pd.read_csv("Bitext_Sample_Customer_Service_Testing_Dataset.csv")
    validate_data =  pd.read_csv("Bitext_Sample_Customer_Service_validation_Dataset.csv")

    train_data_labels = train_data[["intent"]]
    test_data_labels = test_data[["intent"]]
    validate_data_labels = validate_data[["intent"]]


    with open("mapping.json","r") as f:
        mapping_data = json.load(f)

    mapping_data
    train_data_labels['intent'] = train_data_labels['intent'].map(mapping_data).fillna(train_data_labels['intent'])
    validate_data_labels["intent"] = validate_data_labels["intent"].map(mapping_data).fillna(validate_data_labels["intent"])
    test_data_labels["intent"] = test_data_labels["intent"].map(mapping_data).fillna(test_data_labels["intent"])

    return np.array(train_data_labels), np.array(test_data_labels), np.array(validate_data_labels)

train_labels, test_labels, validate_labels = read_data()


def read_features():
    train_data_np =  np.load("train_features.npz")
    final_train = []
    for each in train_data_np:
        for each_one in train_data_np[each]:
            final_train.append(each_one)

    final_train = np.array(final_train)

    test_data_np=  np.load("test_features.npz")
    final_test = []
    for each in test_data_np:
        for each_one in test_data_np[each]:
            final_test.append(each_one)

    final_test = np.array(final_test)

    validate_data_np =  np.load("validate_features.npz")
    final_validate = []
    for each in validate_data_np:
        for each_one in validate_data_np[each]:
            final_validate.append(each_one)

    final_validate = np.array(final_validate)
    return final_train, final_test, final_validate

train_features, test_features, validate_features = read_features()

print("Done with the features part")

def train_neural_network():
    print("Neural network method")
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization

    # Define the number of classes
    num_classes = 27

    model = Sequential([
        Flatten(input_shape=(1024,)), 
        Dense(512, activation='relu'),  
        Dropout(0.2),
        BatchNormalization(),
        Dense(256, activation='relu'),  
        Dropout(0.1),
        Dense(128, activation='relu'),  
        Dense(64, activation='relu'),   
        Dense(32, activation='relu'),   
        Dense(num_classes, activation='softmax') 
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Print the model summary
    print(model.summary())

    # Train the model
    history = model.fit(train_features, train_labels,
                        epochs=100,  
                        batch_size=32,  
                        validation_data=(validate_features, validate_labels))
    print(history)

    
    test_loss, test_accuracy = model.evaluate(test_features, test_labels, verbose=2)
    model.save('customer_bot_model.h5')
    pred_labels = model.predict(test_features)
    predicted_indices = np.argmax(pred_labels, axis=1)

    from sklearn.metrics import precision_score, recall_score, classification_report
    print(classification_report(predicted_indices, test_labels))
    # print("Predicted Indices:", predicted_indices)
    print("Precision score: ",precision_score(predicted_indices, test_labels, average="weighted"))
    print("Recall score: ", recall_score(predicted_indices, test_labels, average="weighted"))
    print(f'Test accuracy: {test_accuracy*100:.2f}%')




train_neural_network()