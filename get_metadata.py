#!/usr/bin/python
# -*- coding: utf-8 -*-
#
#   /$$$$$$  /$$      /$$ /$$      /$$ /$$$$$$$$
#  /$$__  $$| $$$    /$$$| $$$    /$$$|__  $$__/
# | $$  \__/| $$$$  /$$$$| $$$$  /$$$$   | $$   
# |  $$$$$$ | $$ $$/$$ $$| $$ $$/$$ $$   | $$   
#  \____  $$| $$  $$$| $$| $$  $$$| $$   | $$   
#  /$$  \ $$| $$\  $ | $$| $$\  $ | $$   | $$   
# |  $$$$$$/| $$ \/  | $$| $$ \/  | $$   | $$   
#  \______/ |__/     |__/|__/     |__/   |__/  
#
#
# Developed during Biomedical Hackathon 6 - http://blah6.linkedannotation.org/
# Authors: Ramya Tekumalla, Javad Asl, Juan M. Banda
# Contributors: Kevin B. Cohen, Joanthan Lucero
import sys

import tweepy
import json
import math
import csv
import zipfile
import argparse
import os
import os.path as osp
import pandas as pd
from tweepy import TweepError
from time import sleep


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

    output_file_noformat = output_file.split(".", maxsplit=1)[0]
    print(output_file)
    output_file = '{}'.format(output_file)
    output_file_short = '{}_short.json'.format(output_file_noformat)
    compression = zipfile.ZIP_DEFLATED

    inputfile_data = None
    if '.tsv' in args.inputfile:
        inputfile_data = pd.read_csv(args.inputfile, sep='\t')
        print('tab seperated file, using \\t delimiter')
    elif '.csv' in args.inputfile:
        inputfile_data = pd.read_csv(args.inputfile)
    elif '.txt' in args.inputfile:
        inputfile_data = pd.read_csv(args.inputfile, sep='[ \n]', header=None, names= ['user_id', 'type'], engine="python")
    

    if not isinstance(args.idcolumn, type(None)):
        inputfile_data = inputfile_data.set_index(args.idcolumn)
    else:
        inputfile_data = inputfile_data.set_index('user_id')

    ids = list(inputfile_data.index)
    print('total ids: {}'.format(len(ids)))

    start = 0
    end = 100
    limit = len(ids)
    i = int(math.ceil(float(limit) / 100))

    if osp.isfile(args.outputfile) and osp.getsize(args.outputfile) > 0:
        with open(output_file, 'rb') as f:
            # may be a large file, seeking without iterating
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        last_user = json.loads(last_line)

        start = ids.index(last_user['id'])
        end = start + 100
        i = int(math.ceil(float(limit - start) / 100))

    print('metadata collection complete')
    print('creating master json file')
    try:
        with open(output_file, 'a') as outfile:
            for go in range(i):
                print('currently getting {} - {}'.format(start, end))
                sleep(6)  # needed to prevent hitting API rate limit
                id_batch = ids[start:end]
                start += 100
                end += 100
                back_off_counter = 1
                while True:
                    try:
                        users = api.lookup_users(id_batch)
                        break
                    except tweepy.TweepError as ex:
                        print('Caught the TweepError exception:\n %s' % ex)
                        sleep(30 * back_off_counter)  # sleep to see if connection Error is resolved before retrying
                        back_off_counter += 1  # increase backoff
                        break
                for user in users:
                    user._json['type'] = inputfile_data.at[user._json['id'], 'type']
                    json.dump(user._json, outfile)
                    outfile.write('\n')
    except:
        print('exception: continuing to zip the file')
    
    print('creating ziped master json file')
    zf = zipfile.ZipFile('{}.zip'.format(output_file_noformat), mode='w')
    zf.write(output_file, compress_type=compression)
    zf.close()

    # This is where we pick what we want from the user
    print('creating minimized json master file')
    with open(output_file_short, 'w') as outfile:
        with open(output_file) as json_data:
            for user in json_data:
                data = json.loads(user)
                t = {
                    "user_id": data["id"],
                    "followers_count": data["followers_count"],
                    "friends_count": data["friends_count"]
                }
                json.dump(t, outfile)
                outfile.write('\n')

    f = csv.writer(open('{}.csv'.format(output_file_noformat), 'w'))
    print('creating CSV version of minimized json master file')
    fields = ["user_id", "followers_count", "friends_count"]
    f.writerow(fields)
    with open(output_file_short) as master_file:
        for user in master_file:
            data = json.loads(user)
            f.writerow([data["user_id"], data["followers_count"], data["friends_count"]])


# main invoked here    
main()
