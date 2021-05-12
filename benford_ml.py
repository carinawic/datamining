import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import json


labels = []
combined_data = []

with open("benford_input.txt", encoding="utf-8") as f:
    
    json_data = json.load(f)
    for i in json_data["users"]:
        labels.append(i["real"])
        combined_data.append([i["benford_degree"], i["degree"], i["clustering"]])


combined_training_data = combined_data[:12]
combined_training_labels = labels[:12]
combined_test_data = combined_data[12:]
combined_test_labels = labels[12:]


# keras.Sequential groups a linear stack of layers into a Model
model = Sequential()
# dense layer: we narrow all our nodes into 16 nodes
# ReLU (Rectified Linear Unit) = f(x) = 0 until a certain x-value thereafter linear 
model.add(Dense(16, activation='relu', input_dim=3))
# last layer is 1 neuron with a value from a sigmoid function 
model.add(Dense(1, activation='sigmoid'))

# creating the model
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# we try to fit our model to our training data
# epochs is the amount of training rounds
# verbose alters the terminal output type
model.fit(combined_training_data, combined_training_labels, epochs=500, verbose=2)

print("evaluating test data:")

# testing our network on the test data
test_results = model.evaluate(combined_test_data, combined_test_labels)
print(test_results)

'''
# printing a summary of the model
model.summary()

# making a prediction for a specific value:
predict = model.predict(test_data[0])
print("prediction:", predict[0])
print("actual:", test_labels[0])
'''