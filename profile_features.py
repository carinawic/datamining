from datetime import date, datetime
import csv
import json
import pandas as pd

date_format = "%a %b %d %H:%M:%S %z %Y"
crawling_date = datetime.strptime("Sat May 29 15:59:00 +0000 2021", date_format)
months_threshold = 2 * 365 / 12

# followers_count / friends_count
def ff_ratio(user):
    # return user['followers_count'] / user['friends_count']
    if user['followers_count'] == 0:
        return 1
    return user['friends_count'] / (user['followers_count'] ** 2)


def no_tweets(user):
    return user['statuses_count']

    
def profile_has_name(user):
    if not user['name']:
        return 0
    return 1


def no_friends(user):
    return user['friends_count']


def no_followers(user):
    return user['followers_count']


# the ratio between the account’s following number to the age 
# of the account at the time the following number was recorded
def following_rate(user):
    # compute the age of the account in days
    account_date = datetime. strptime(user['created_at'], date_format)
    delta = crawling_date - account_date
    return user['followers_count'] / delta.days


# the account is more than 2 months old and still has a default profile picture
def default_profile_image(user):
    account_date = datetime. strptime(user['created_at'], date_format).replace(tzinfo=None)
    delta = datetime.now() - account_date
    if user['default_profile_image'] and delta.days > months_threshold:
        return 1
    return 0


def belongs_to_list(user):
    if user['listed_count'] > 0:
        return 1
    return 0


def location(user):
    if not user['location']:
        return 0
    return 1


def ff_ratio_50(user):
    if user['followers_count'] == 0 or \
        user['friends_count'] / user['followers_count'] >= 50:
        return 1
    return 0


def has_bio(user):
    if not user['description']:
        return 0
    return 1


# profile features dictionary
pf_dict = {
    'ff_ratio': ff_ratio,
    'no_tweets': no_tweets,
    'profile_has_name': profile_has_name,
    'no_friends': no_friends,
    'no_followers': no_followers,
    'following_rate': following_rate,
    'default_profile_image': default_profile_image,
    'belongs_to_list': belongs_to_list,
    'location': location,
    'ff_ratio_50': ff_ratio_50,
    'has_bio': has_bio
}


features = ['id', 'ff_ratio', 'no_tweets', 'profile_has_name', 'no_friends',
    'no_followers', 'following_rate', 'belongs_to_list', 'location', 'has_bio']


def compute_feature_vector(input_file, features, output_file):
    # save results to a CSV file to be used as features for our classifier
    f = csv.writer(open('{}.csv'.format(output_file), 'w'))
    fields = features + ['dataset', 'bot']
    f.writerow(fields)

    with open(input_file, 'r') as infile:
        for line in infile:
            user = json.loads(line)
            f_vec = [user['id_str']]
            for feature in features[1:]:
                f_vec.append(pf_dict[feature](user))
            f_vec.append(user['type'][1:-1])
            if "_real" in user['type']:
                f_vec.append(0)
            else:
                f_vec.append(1)
            f.writerow(f_vec)


def get_profile_features_subset():
    edges_files = ['crawl-friends/alex_2', 'crawl-friends/ella_2', 'crawl-friends/stefi_2']
    user_ids = []
    for file in edges_files:
        edges = pd.read_csv('{}.csv'.format(file), header=0, dtype=int)
        edges.drop_duplicates(subset='source_id', keep='first', inplace=True)
        user_ids.extend(edges['source_id'].to_list())

    print(len(user_ids))
    # select profile features
    profile_features_subset = pd.read_csv('profile_features.csv', header=0)
    print('profile_features_subset', profile_features_subset.shape[0])
    
    f = csv.writer(open('profile_features_subset.csv', 'w'))
    fields = features + ['dataset', 'bot']
    f.writerow(fields)
    for (_, feature) in profile_features_subset.iterrows():
        if feature['id'] in user_ids:
            f.writerow(feature)

# compute_feature_vector('hydrated_tweets.json', features, 'profile_features')
get_profile_features_subset()
