import pandas as pd
import json
import csv

dataset_list = ["E13_real", "TFP_real", "FSF_bot", "INT_bot", "TWT_bot"]


def remove_csv_duplicates(file_csv, drop_on):
    """
    Remove duplicates from a csv file
    :param file_csv: a csv file
    :param drop_on: list of attributes to drop duplicates on
    :return: a dictionary without duplicates
    """
    users = pd.read_csv(file_csv, header=0, dtype=int)
    print(users.info())
    users = users.drop_duplicates(subset=drop_on)
    print(users.info())
    return users.to_dict('index')


def get_users(columns=None):
    """
    Combine users from all datasets
    :param columns: what columns to use
    :return: dict of users
    """
    if columns is None:
        columns = [0, 4, 5]
    users = pd.read_csv("data/{}/users.csv".format(dataset_list[0]), header=0, usecols=columns)
    users["label"] = dataset_list[0]
    for i in range(1, len(dataset_list)):
        temp_users = pd.read_csv("data/{}/users.csv".format(dataset_list[i]), header=0, usecols=columns)
        temp_users["label"] = dataset_list[i]
        users = users.append(temp_users, ignore_index=True)
    users = users.set_index("id")
    return users.to_dict('index')


def create_dataset(edges, origin_users, extended_users, out_filename):
    """
    Function to create dataset with all users and their friends, and their friend and follower-counts
    :param edges: all edges between users
    :param origin_users: the original user dataset
    :param extended_users: the friends of our users dataset
    :param out_filename: filename of the outfile
    """
    user_info_dict = {}
    # For every unique key in original users, create an empty dict and add dataset type and friends dict
    for user_id in origin_users.keys():
        user_id = int(user_id)
        user_info_dict[user_id] = {}
        user_info_dict[user_id]["user_id"] = user_id
        user_info_dict[user_id]["user_dataset"] = origin_users[str(user_id)]["label"]
        user_info_dict[user_id]["friends"] = {}
        # Looks like this:
        # {3610511: {"user_id": 3610511, "user_dataset": "E13_real", "friends": {}}

    # list to see what users we have no information on
    no_match_users = []

    # For every row in the edges, get the index (starting at 0)
    for index in edges:
        # See if there is a match in the extended dataset
        try:
            user_info_dict[edges[index]["source_id"]]["friends"][edges[index]["target_id"]] = \
                extended_users[str(edges[index]["target_id"])]
        except Exception as e:
            print(edges[index]["target_id"], "not in external users")
            # If not, see if there is a match in the original users
            try:
                user_info_dict[edges[index]["source_id"]]["friends"][edges[index]["target_id"]] = \
                    extended_users[str(edges[index]["target_id"])]
            # If not, add to the no match list
            except Exception as e:
                print(edges[index]["target_id"], "also not in original users :/")
                no_match_users.append(str(edges[index]["target_id"]))

    # Make sure all users in the list are unique
    no_match_users = list(set(no_match_users))
    print(len(no_match_users))

    # Save these to a file
    with open("no_match.csv", "w") as nomatch:
        for user in no_match_users:
            nomatch.write(user)
            nomatch.write("\n")

    # Save data to file
    with open(out_filename, 'w') as json_file:
        json.dump(user_info_dict, json_file)


def combine_files(filenames, outfile):
    """
    Function to combine files. Appends row after row to a new file.
    :param filenames: a list of the names of the files to combine
    :param outfile: the name of the outfile
    """
    with open(outfile, 'w') as outfile:
        a = 0
        for fname in filenames:
            print(fname)
            print(a)
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                    a += 1


def minimize_file(filename, new_filename):
    """
    Saves an user file in a minimized format
    :param filename: the name of the input file
    :param new_filename: the name of the outfile
    """
    print('creating minimized json master file')
    with open("{}.json".format(new_filename), 'w') as outfile:
        json_data = open(filename, "r")
        a = 0
        for user in json_data.readlines():
            print(a)
            data = json.loads(user)
            t = {
                "user_id": data["id"],
                "followers_count": data["followers_count"],
                "friends_count": data["friends_count"]
            }
            json.dump(t, outfile)
            outfile.write('\n')
            a += 1
        json_data.close()

    f = csv.writer(open("{}.csv".format(new_filename), 'w'))
    print('creating CSV version of minimized json master file')
    fields = ["user_id", "followers_count", "friends_count"]
    f.writerow(fields)
    file = open("{}.json".format(new_filename), "r")
    for user in file.readlines():
        data = json.loads(user)
        f.writerow([data["user_id"], data["followers_count"], data["friends_count"]])
    file.close()


def reverse_csv(infile, outfile):
    with open(infile, "r") as file_in:
        with open(outfile, "w") as file_out:
            file_list = file_in.readlines()
            file_out.write(file_list[0])
            for row in reversed(list(file_list[1:])):
                file_out.write(row)
