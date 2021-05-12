import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import json

benford_degrees = []
degrees = []
clusterings = []
labels = []

with open("benford_input.txt", encoding="utf-8") as f:
    
    
    json_data = json.load(f)
    for i in json_data["users"]:
        benford_degrees.append(i["benford_degree"])
        degrees.append(i["degree"])
        clusterings.append(i["clustering"])
        labels.append(i["real"])

training_data = benford_degrees[:12]
training_labels = labels[:12]
test_data = benford_degrees[12:]
test_labels = labels[12:]

print("training data:")
print(training_data)
print(training_labels)
print("test data:")
print(test_data)
print(test_labels)

model = Sequential()
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=500, verbose=2)

print("evaluating test data:")

test_results = model.evaluate(test_data, test_labels)
print(test_results)

# printing a summary of the model
#model.summary()

# making a prediction for a specific value:
'''
predict = model.predict(test_data[0])
print("prediction:", predict[0])
print("actual:", test_labels[0])
'''