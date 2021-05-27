import sys

import tweepy
import json
import argparse
import pandas as pd
from time import sleep


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
main()
