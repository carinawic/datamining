from datetime import date, datetime
import csv
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
    print(user['default_profile_image'])
    # TODO: not sure about this feature, the field should be True, but it's acutually nan
    # when I print it
    if user['default_profile_image'] and delta.days > months_threshold:
        return 1
    return 0


# belongs to a list
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
    'no_followers', 'following_rate', 'belongs_to_list',
    'location', 'has_bio']


def compute_feature_vector(dataset, features, output_file):
    # read only the columns needed for computing the features
    # columns: name, statuses_count, followers_count, friends_count, listed_count
    # crated_at, location, default_profile_image, description
    users_df = pd.read_csv("data/{}/users.csv".format(dataset), header=0, usecols=[0, 1, 3, 4, 5, 7, 8, 12, 14, 31])

     # save results to a CSV file to be used as features for our classifier
    f = csv.writer(open('{}.csv'.format(output_file), 'w'))
    f.writerow(features)

    for _, user in users_df.iterrows():
        f_vec = [user['id']]
        for feature in features[1:]:
            f_vec.append(pf_dict[feature](user))
        f.writerow(f_vec)

compute_feature_vector('TFP_real', features, 'profile_features')
