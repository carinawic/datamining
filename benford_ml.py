from operator import index
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import matplotlib.pyplot as plt
import json
from scipy.stats import chisquare, chi2
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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


def partition_with_regards_to_dataset(profile_features_file, graph_features_file, n):
    
    x_train = [] # training data
    y_train = [] # training labels
    x_test = [] # test data
    y_test = [] # test labels

    dataset = ""
    counter = 0

    graph_features = False
    if graph_features_file:
        with open(graph_features_file, mode='r') as inp:
            reader = csv.reader(inp)
            dict_from_csv = {row[0]:(row[1], row[2], row[3]) for row in reader}
        graph_features = True

    # read the profile features of each user
    user_features = pd.read_csv(profile_features_file, header=0,  converters={'id': str})
    for _, row in user_features.iterrows(): 
        if dataset != row['dataset']: # new dataset
            counter = 0          
            
        dataset = row['dataset']
        user_type = row['real'] # row['real'][i] is 0 or 1
        sample = [row['ff_ratio'], row['no_tweets'], row['profile_has_name'], \
                row['no_friends'], row['no_followers'], row['following_rate'], \
                row['belongs_to_list'], row['location'], row['has_bio']]
        # add graph features if specified
        if graph_features:
            if row['id'] in dict_from_csv:
                elem = dict_from_csv[row['id']]
                sample += [elem[0], elem[1], elem[2]]
            else:
                sample += [0.0, 0.0, 0.0]

        # add the n first items of each dataset as test data
        if counter < n:
            x_test.append(sample)
            y_test.append(user_type)
        else:
            x_train.append(sample)
            y_train.append(user_type)

        counter = counter + 1

    return  x_train, y_train, x_test, y_test


def performance_metrics(classifier, y_test, y_pred):
    # evaluate the peformance of the model
    print('Confusion matrix ', confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    # see which features are the most important
    feature_imp = pd.Series(classifier.feature_importances_,index=feature_names).sort_values(ascending=False)
    print(feature_imp)


def random_forest(feature_names, x_train, y_train, x_test, y_test):
    # scale the features
    sc = StandardScaler()
    x_train = sc.fit_transform(np.array(x_train))
    x_test = sc.transform(np.array(x_test))
    # create a baseline model
    # instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
    # train the model on training data
    rf.fit(x_train, y_train)
    # use the forest's predict method on the test data
    y_pred = rf.predict(x_test)
    # evaluate model
    print('RandomForestClassifier performance before tuning')
    performance_metrics(rf, y_test, y_pred)

    # hyperparameter tunning
    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # method of selecting samples for training each tree
    bootstrap = [True, False]
    # create the random grid
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    print(param_grid)

    rf_tuned = RandomForestClassifier()
    # instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf_tuned, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    # fit the grid search to the data
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    # best_grid = grid_search.best_estimator_

    # train a model with the best parameters produced by GridSearch
    best_params_dict = grid_search.best_params_
    rf_best = RandomForestClassifier(n_estimators=best_params_dict['n_estimators'], 
        max_features=best_params_dict['max_features'],
        max_depth=best_params_dict['max_depth'],
        min_samples_split=best_params_dict['min_samples_split'], 
        min_samples_leaf=best_params_dict['min_samples_leaf'],
        bootstrap=best_params_dict['bootstrap'])

    y_pred = rf_best.predict(x_test)
    print('RandomForestClassifier performance after tuning')
    performance_metrics(rf_best, y_test, y_pred)


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

if __name__ == '__main__':
    # classify based on profile features
    feature_names = ['ff_ratio', 'no_tweets', 'profile_has_name', 'no_friends',
    'no_followers', 'following_rate', 'belongs_to_list', 'location', 'has_bio']
    x_train, y_train, x_test, y_test = partition_with_regards_to_dataset('profile_features.csv', '', 10)
    random_forest(feature_names, x_train, y_train, x_test, y_train)

    # classify based on profile features + graph features
    feature_names += ['beetweenness_centrality', 'local_clustering_coefficient', 'degree_centrality']
    x_train, y_train, x_test, y_test = partition_with_regards_to_dataset('profile_features.csv', 'graph_features.csv', 10)
    random_forest(feature_names, x_train, y_train, x_test, y_train)
