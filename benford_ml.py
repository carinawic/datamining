import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import json


labels = []
combined_data = []

benfords_probs = {
    1 : 30.1,
    2 : 17.6,
    3 : 12.5,
    4 : 9.7,
    5 : 7.9,
    6 : 6.7,
    7 : 5.8,
    8 : 5.1,
    9 : 4.6
}

benford_probs_array = np.array(list(benfords_probs.values()))

def benford_score_pearson(first_digits_freq):
    freq_array = np.array(list(first_digits_freq.values()))
    my_rho = np.corrcoef(benford_probs_array, freq_array)
    return my_rho[0,1]


def benford_score_simple(first_digits_freq, total_num_values):
    total_percentage_diff = 0

    for key in list(first_digits_freq.keys()):
        print(f'{key} occured {first_digits_freq[key]} times out of {total_num_values} => {first_digits_freq[key] / total_num_values *100}% \
            which should be {benfords_probs[key]}% => we are off by {first_digits_freq[key] / total_num_values * 100 - benfords_probs[key]:.1f} %')
        
        total_percentage_diff += first_digits_freq[key] / total_num_values * 100 - benfords_probs[key]

    return 1 / (total_percentage_diff / total_num_values)


# output_file_short is a file containing items like
# {"user_id": 286543, "followers_count": 1185, "friends_count": 867}
def compute_benford_score(output_file_short, benford_score_type):
    
    d = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    total_num_values = 0

    with open(output_file_short) as master_file:
        for user in master_file:
            data = json.loads(user)
            first_digit = (int(str(data["friends_count"])[:1]))
            # Increment count of word by 1
            d[first_digit] = d[first_digit] + 1
            total_num_values += 1
        
        if benford_score_type == 'simple':
            return benford_score_simple(d, total_num_values)

        return benford_score_pearson(d)


def classifier():
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

compute_benford_score('hydrated_tweets_short.json', 'pearson')