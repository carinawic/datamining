import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt
import json
from scipy.stats import chisquare, chi2
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

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

benford_probs_array = list(benfords_probs.values())

# the two methods retaurn the same chi square value
def chisq_stat(o, e):
    return sum( [(o - e)**2/e for (o, e) in zip(o, e)] )


def benford_score_chisquare(fdf_array):
    # dof = no of categories - 1
    chisq, p = chisquare(f_obs=fdf_array, f_exp=benford_probs_array)
    # print(f'chisq = {chisq}')
    return chisq


def benford_score_pearson(fdf_array):
    my_rho = np.corrcoef(benford_probs_array, fdf_array)
    return my_rho[0,1]


def benford_score_simple(first_digits_freq, total_num_values):
    total_percentage_diff = 0

    if total_num_values!=0:
        for key in list(first_digits_freq.keys()):
        #print(f'{key} occured {first_digits_freq[key]} times out of {total_num_values} => {first_digits_freq[key] / total_num_values *100}% \
        #which should be {benfords_probs[key]}% => we are off by {first_digits_freq[key] / total_num_values * 100 - benfords_probs[key]:.1f} %')
        
        
            total_percentage_diff += abs((first_digits_freq[key] / total_num_values) * 100 - benfords_probs[key])

            #print("total_num_values", total_num_values)
            #print("total_percentage_diff", total_percentage_diff)
            return total_percentage_diff / total_num_values
        
    return -1    


# output_file_short is a file containing items like
# {"user_id": 286543, "followers_count": 1185, "friends_count": 867}
def compute_benford_score(input_dict, benford_score_method):
    
    dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    total_num_values = 0

    
    for friend in input_dict:
        friend_count = input_dict[friend]['friends_count']
        first_digit = (int(str(friend_count)[:1]))

        # Increment count of word by 1
        if(first_digit != 0):
            dict[first_digit] = dict[first_digit] + 1
            total_num_values += 1
    
    # convert the freq to precentages
    fdf_array = []
    for v in list(dict.values()):
        fdf_array.append(v / total_num_values * 100)
    # print(fdf_array)

    benford_score_func = {'simple': benford_score_simple, 
                          'pearson': benford_score_pearson, 
                          'chi-square': benford_score_chisquare}[benford_score_method]

    return benford_score_func(fdf_array)


def calculate_benford_for_each_user():
    # this method is just for plotting now
    
    # building a lookup table where we can input a user and see if real or fake
    with open("all_users.csv") as csvfile:
            
        real_or_fake_dict = {}
        for content in csvfile: 
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                #row[0] is id, row[1] is real or fake
                if(row[1].find("bot")>0):
                    real_or_fake_dict[row[0]] = "bot"
                else:
                    real_or_fake_dict[row[0]] = "real"
        
        # just checking how many bots vs real users we have
        botcounter = 0
        realcounter = 0

        for key in real_or_fake_dict:
            if real_or_fake_dict[key] == "bot":
                botcounter = botcounter + 1
            else:
                realcounter = realcounter + 1

        print("amount of real users: ", realcounter)
        print("amount of bots: ", botcounter)

    
    #separating the users to plot them in different colors later
    fakeuserbenford = []
    realuserbenford = []
    fakeusers = []
    realusers = []

    with open("final_data_v2.json") as json_file:
            
        for content in json_file: # there is only 1
                
            users = json.loads(content)
            n = 0    
            # each user who will have a unique benford's score
            for user in users:

                # ['friends'] returns the list of such elements:
                # {'12': {'followers_count': 5389306, 'friends_count': 4660}}
                friendproperties = (users[user]['friends'])
                # compute the benford score for accounts with more than 100 friends
                if len(friendproperties) < 100:
                    # print(f'friendproperties len {len(friendproperties)}')
                    continue

                benford_degree = compute_benford_score(friendproperties, 'chi-square')
                # print("user", user)
                # print("has benford degree", benford_degree)

                # filling our lists that we want to plot
                if user in real_or_fake_dict:
                    if real_or_fake_dict[user] == "bot":
                        fakeusers.append(user)
                        fakeuserbenford.append(benford_degree)
                    else:
                        realusers.append(user)
                        realuserbenford.append(benford_degree)               

                # n = n + 1
                # if n == 10:
                #     break

    # plotting users
    #print("amount of real users: ", len(realusers))
    #print("amount of bots: ", len(fakeusers))

    #plt.plot(fakeusers, fakeuserbenford, color='red', marker='o')
    #plt.plot(realusers, realuserbenford, color='green', marker='o')
    #plt.show()

def partition_kfold(to_be_partitioned):

    # from: https://scikit-learn.org/stable/modules/cross_validation.html

    # the size of the test set will be 1/K (i.e. 1/n_splits), 
    # so you can tune that parameter to control the test size 
    # (e.g. n_splits=3 will have test split of size 1/3 = 33% of your data
    kf = KFold(n_splits=3)

    '''
    the return value partition_kfold(X) is iterable such as:

    for train, test in partition_kfold(X):
       print(train)
       print(test)
    '''

    return kf.split(to_be_partitioned)

def partition_shufflesplit(to_be_partitioned):
    # from: https://scikit-learn.org/stable/modules/cross_validation.html

    # here we specify a test size instead, and the test data is picked randomly from all over the total data
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    return ss.split(to_be_partitioned)

def partition_with_regards_to_dataset(n):
    
    training_data = []
    training_labels = []
    test_data = []
    test_labels = []

    with open("benford_input.txt", encoding="utf-8") as f:
        
        json_data = json.load(f)
        dataset = ""
        counter = 0

        for i in json_data["users"]:

            if dataset != i["dataset"]: # new dataset
                counter = 0
                
            dataset = i["dataset"]
            
            # add the n first items of each dataset as test data
            if counter < n:
                test_labels.append(i["real"]) # i["real"] is 0 or 1
                test_data.append([i["benford_degree"], i["degree"], i["clustering"]])
            else:
                training_labels.append(i["real"]) # i["real"] is 0 or 1
                training_data.append([i["benford_degree"], i["degree"], i["clustering"]])

            counter = counter + 1

    return test_labels, test_data, training_labels, training_data


def classifier():
    

    # partition_with_regards_to_dataset(3)[0] 
    # will return the test_labels list, which contains the test_labels of the 3 first nodes in every dataset
    test_and_training_data = partition_with_regards_to_dataset(3) 

    print(test_and_training_data[0])
    
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
    model.fit(test_and_training_data[3], test_and_training_data[2], epochs=500, verbose=2)

    print("evaluating test data:")

    # testing our network on the test data
    test_results = model.evaluate(test_and_training_data[1], test_and_training_data[0])
    print(test_results)


    '''
    # printing a summary of the model
    model.summary()

    # making a prediction for a specific value:
    predict = model.predict(test_data[0])
    print("prediction:", predict[0])
    print("actual:", test_labels[0])
    '''

calculate_benford_for_each_user()
classifier()