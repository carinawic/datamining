import sys
from pandas.core.algorithms import unique

import tweepy
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from math import sqrt

dataset_list = ["E13_real", "TFP_real", "FSF_bot", "INT_bot", "TWT_bot"]


def no_unique_friends():
    f = open('final_dataset.json', 'r')
    users = json.loads(f.read())

    friends_list = []
    for user in users:
        # ['friends'] returns the list of such elements:
        # {'12': {'followers_count': 5389306, 'friends_count': 4660}}
        friends = users[user]['friends']
        for friend in friends:
            friends_list.append(int(friend)) 

    unique_friends_df = pd.DataFrame(friends_list)
    print(unique_friends_df.info())
    unique_friends_df.drop_duplicates(keep='first', inplace=True)
    print(unique_friends_df.shape[0])


def subset_of_friends():
    f = open('final_dataset.json', 'r')
    users = json.loads(f.read())

    friends_list = []
    for user in users:
        # ['friends'] returns the list of such elements:
        # {'12': {'followers_count': 5389306, 'friends_count': 4660}}
        friends = users[user]['friends']
        for friend in friends:
            friends_list.append({'source_id': int(user), 'target_id': int(friend), 'friends_count': 
            friends[friend]['friends_count'], 'dataset': users[user]['user_dataset']})

    friends_df = pd.DataFrame(friends_list, columns = ['source_id', 'target_id', 'friends_count', 'dataset'])

    # write the friends to be crawled to 5 csv files divided by the dataset name
    for dataset in dataset_list:
        dataset_df = pd.DataFrame(friends_df[friends_df.dataset == dataset])
        min_val = min(dataset_df['friends_count'])
        max_val = max(dataset_df['friends_count'])
        print(dataset, dataset_df.shape[0], 'min =', min_val, 'max =', max_val)
        print(dataset, 'size = ', dataset_df.shape[0])
        dataset_df.drop_duplicates('target_id', keep='first', inplace=True)
        print(dataset, 'size = ', dataset_df.shape[0])
        dataset_df[['source_id', 'target_id', 'friends_count']].to_csv('crawl-friends/{}.csv'.format(dataset), index=False)
        # dataset_df[['source_id', 'target_id']].to_csv('edges/{}.csv'.format(dataset), index=False)

    # TODO: plot histogram of number of friends per user, not working yet
    # friend_counts = friends_df[friends_df.dataset == "TFP_real"]['friends_count'].astype(int).to_numpy()
    # print(len(friend_counts))
    # min_val = min(friend_counts)
    # max_val = max(friend_counts)
    # bin_width = max_val - min_val
    # print(friend_counts)
    # plt.hist(friend_counts, ec='black', bins='auto')
    # plt.show()


def get_subset_of_friends():
    for dataset in dataset_list:
        dataset_df = pd.read_csv("crawl-friends/{}.csv".format(dataset), header=0, usecols=[0, 1])    
        friends_df = dataset_df.groupby('source_id').head(50)
        friends_df.to_csv('crawl-friends/{}_friends.csv'.format(dataset), index=False)


"""
Script to get all the friends ID's 
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outputfile", help="Output file name with extension")
    parser.add_argument("-i", "--inputfile", help="Input file name with extension")
    parser.add_argument("-k", "--keyfile", help="Json api key file name")
    parser.add_argument("-c", "--idcolumn", help="user id column in the input file, string name")

    args = parser.parse_args()
    if args.inputfile is None or args.outputfile is None:
        parser.error("please add necessary arguments")

    if args.keyfile is None:
        parser.error("please add a keyfile argument")

    with open(args.keyfile) as f:
        keys = json.load(f)

    auth = tweepy.OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
    auth.set_access_token(keys['access_token'], keys['access_token_secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True, retry_delay=60 * 3, retry_count=5,
                     retry_errors={401, 404, 500, 503}, wait_on_rate_limit_notify=True)

    if not api.verify_credentials():
        print("Your twitter api credentials are invalid")
        sys.exit()
    else:
        print("Your twitter api credentials are valid.")

    output_file = args.outputfile

    inputfile_data = None
    if '.tsv' in args.inputfile:
        inputfile_data = pd.read_csv(args.inputfile, sep='\t')
        print('tab seperated file, using \\t delimiter')
    elif '.csv' in args.inputfile:
        inputfile_data = pd.read_csv(args.inputfile)
    elif '.txt' in args.inputfile:
        inputfile_data = pd.read_csv(args.inputfile, sep='\n', header=None, names=['user_id'])

    if not isinstance(args.idcolumn, type(None)):
        inputfile_data = inputfile_data.set_index(args.idcolumn)
    else:
        inputfile_data = inputfile_data.set_index('user_id')

    ids = list(inputfile_data.index)
    print('total ids: {}'.format(len(ids)))

    print('creating master csv file')
    try:
        with open(output_file, 'w') as outfile:
            outfile.write("source_id, target_id\n")
            for user in ids:
                sleep(6)  # needed to prevent hitting API rate limit
                back_off_counter = 1
                while True:
                    try:
                        print(user)
                        friends = api.friends_ids(user)
                        for friend in friends:
                            outfile.write("{}, {}".format(user, friend))
                            outfile.write('\n')
                        break
                    except tweepy.TweepError as ex:
                        print('Caught the TweepError exception:\n %s' % ex)
                        print(ex.response)
                        sleep(30 * back_off_counter)  # sleep to see if connection Error is resolved before retrying
                        back_off_counter += 1  # increase backoff
                        break
    except Exception as e:
        print('exception: ', e)


# main invoked here    
# main()
# subset_of_friends()
# get_subset_of_friends()
no_unique_friends()
