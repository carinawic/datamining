import csv
import pandas as pd
import networkx as nx
import networkit as nk
import pickle
import matplotlib.pyplot as plt
import argparse

dataset_list = ["E13_real", "TFP_real", "FSF_bot", "INT_bot", "TWT_bot"]

def remove_outsiders(user_df, edge_df):
    """
    Takes the user dataset and a dataset describing the edges, and removes the edges pointing to user who we have no
    data on.
    :param user_df: A pandas dataframe containing twitter user IDs
    :param edge_df: A pandas dataframe containing the source and target of the relations between twitter users
    :return: The new dataframe of edges
    """
    condition = edge_df.target_id.isin(user_df.id)
    edge_df.drop(edge_df[~condition].index, inplace=True)
    return edge_df


def generate_directed_network(edges_df):
    """
    Creates a directed nx graph from a pandas dataframe
    :param edges_df: A pandas dataframe containing the source and target of the relations between twitter users
    :return: a nx graph
    """
    G = nx.from_pandas_edgelist(edges_df, source="source_id", target="target_id", create_using=nx.DiGraph())
    return G


def draw_directed_graph(directed_graph):
    """
    Function to draw a directed graph using networkx. Not the greatest.
    :param directed_graph: the graph to be drawn
    """
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(directed_graph)
    nx.draw_networkx_nodes(directed_graph, pos)
    nx.draw_networkx_edges(directed_graph, pos, arrows=True)
    plt.show()


def save_pickle(to_save, filename):
    """
    Function to save an object as a pickle.
    :param to_save: The object to save
    :param filename: filename of the object
    """
    filehandler = open(filename, "w")
    pickle.dump(to_save, filehandler)


def read_pickle(filename):
    """
    Function to read a pickle from a file
    :param filename: the name of the pickle file
    :return: the pickeled object
    """
    opened_pickle = open(filename, "r")
    opened_object = pickle.loads(opened_pickle)
    return opened_object


def export_for_gephi(users, edges, type):
    edges.to_csv("{}_data.csv".format(type), index=False, quoting=csv.QUOTE_NONNUMERIC)
    users.to_csv("user_data.csv", sep=" ", index=False, quoting=csv.QUOTE_NONNUMERIC, columns=["id", "label"])


def export_for_getting_metatata(type):
    users_contacts = pd.read_csv("data/{}/{}.csv".format(dataset_list[0], type), header=0, dtype=int, usecols=[1])
    for dataset in dataset_list[1:]:
        temp_contacts = pd.read_csv("data/{}/{}.csv".format(dataset, type), header=0, dtype=int, usecols=[1])
        users_contacts = users_contacts.append(temp_contacts, ignore_index=True)
    print(users_contacts.info())
    users_contacts = users_contacts.drop_duplicates(subset=["target_id"], ignore_index=True)
    print(users_contacts.info())
    users_contacts.columns = ["user_id"]
    users_contacts.to_csv("user_{}_data.csv".format(type), sep=",", index=False)


def create_network(type, keep_outsiders, dataset_list):
    """
    Function to generate and save a networkx graph
    :param type: 'friends' or 'followers' or empty string to combine both friend and followers edges
    :param dataset_list: the datasets to be merged together
    :return: the networkx graph
    """
    users_exist = False
    edges_exist = False

    try:
        users = read_pickle("users_df.pickle")
        users_exist = True
    except Exception:
        print("No user pickle exists")

    try:
        edges = read_pickle("{}_edges_df.pickle".format(type))
        edges_exist = True
    except Exception:
        print("No {} edges pickle exists".format(type))

    if not users_exist:
        users = pd.read_csv("data/{}/users.csv".format(dataset_list[0]), header=0, usecols=[0, 2, 3, 4, 5, 6, 7, 33])
        users["label"] = dataset_list[0]
        for dataset in dataset_list[1:]:
            temp_users = pd.read_csv("data/{}/users.csv".format(dataset), header=0, usecols=[0, 2, 3, 4, 5, 6, 7, 33])
            temp_users["label"] = dataset
            users = users.append(temp_users, ignore_index=True)
        # save_pickle(users, "users_df.pickle")

    if not edges_exist:
        if type:
            edges = pd.read_csv("data/{}/{}.csv".format(dataset_list[0], type), header=0, dtype=int)
            for dataset in dataset_list[1:]:
                temp_edges = pd.read_csv("data/{}/{}.csv".format(dataset, type), header=0, dtype=int)
                edges = edges.append(temp_edges, ignore_index=True)
                # print(f'edges size {edges.shape[0]}')
                # save_pickle(edges, "{}_edges_df.pickle".format(type))
        else:
            edges = pd.read_csv("data/{}/{}.csv".format(dataset_list[0], "friends"), header=0, dtype=int)
            edges_followers = pd.read_csv("data/{}/{}.csv".format(dataset_list[0], "followers"), header=0, dtype=int)
            edges = edges.append(edges_followers, ignore_index=True)
            for dataset in dataset_list[1:]:
                temp_edges_friends = pd.read_csv("data/{}/{}.csv".format(dataset, "friends"), header=0, dtype=int)
                temp_edges_followers = pd.read_csv("data/{}/{}.csv".format(dataset, "followers"), header=0, dtype=int)
                edges = edges.append(temp_edges_followers, ignore_index=True)
                edges = edges.append(temp_edges_friends, ignore_index=True)

    # edges.drop_duplicates().reset_index(drop=True)
    if not keep_outsiders:
        egdes = remove_outsiders(users, edges)

    # export data in csv format
    # users[["id", "dataset"]].to_csv("all_users.csv", index=False, header=True)
    # new_edges.to_csv("edges.csv", index=False, header=True)

    G = generate_directed_network(edges)
    print(f'Number of nodes {G.number_of_nodes()}\nNumber of edges {G.number_of_edges()}')
    # save_pickle(G, "graph.pickle")

    return G


# functions used when analysing each dataset individually
def save_edges_csv(dataset, edges_df, type):
    """
    Function to save a pandas dataframe of edges to a csv file
    :param dataset: the name of the dataset
    :edges_df: the pandas dataframe of edges
    :type: the type of the edges which can be 'friends', 'followers' or both described as 'all'
    """
    edges_df.to_csv ("data/{}/{}_{}_new_edges.csv".format(dataset, dataset, type), index=False, header=True)


def generate_network(dataset, **kwargs):
    """
    Function to generate a graph from a dataset
    :dataset: the dataset for generating the graph
    :**kwargs: if specified, type describes which edges to consider ('friends' or 'followers'), otherwise
    all edges are considered when constructiong the graph
    :return: the networkx graph
    """
    # create a network with the friends or the followers of users in a dataset
    if 'type' in kwargs:
        users = pd.read_csv("data/{}/users.csv".format(dataset), header=0, usecols=[0, 2, 3, 4, 5, 6, 7])
        edges = pd.read_csv("data/{}/{}.csv".format(dataset, kwargs['type']), header=0, dtype=int)    
        edges = remove_outsiders(users, edges)
        # save_edges_csv(dataset, edges, kwargs['type'])
    else:
        # create a network with both friends and followers
        friends = pd.read_csv("data/{}/{}.csv".format(dataset, 'friends'), header=0, dtype=int)
        followers = pd.read_csv("data/{}/{}.csv".format(dataset, 'followers'), header=0, dtype=int)
        # drop duplicates might be redundant    
        edges = pd.concat([friends, followers]).drop_duplicates().reset_index(drop=True)
        # save_edges_csv(dataset, edges, 'all')
        sources = edges['source_id']
        targets = edges['target_id']
        users = pd.concat([sources, targets]).drop_duplicates().reset_index(drop=True)

    print(f'No of nodes {users.shape[0]}')
    print(f'No of edges {edges.shape[0]}')
    G = generate_directed_network(edges)
    
    return G


def generate_network_from_edges(inputfile):
    edges = pd.read_csv(inputfile, header=0)
    print(f'No of edges {edges.shape[0]}')
    G = generate_directed_network(edges)
    return G


def generate_network_lcc(edges_files):
    all_edges = pd.DataFrame(columns=['source_id', 'target_id'])
    for file in edges_files:
        edges = pd.read_csv('{}.csv'.format(file), header=0, dtype=int)
        all_edges = all_edges.append(edges, ignore_index=True)

    print('No of edges', all_edges.shape[0])
    print(all_edges.head(10))
    G = generate_directed_network(all_edges)
    return G


def convert_graph_to_gml(G, filepath):
    nx.write_gml(G, filepath)
    # might need to write ids to a file in network.py to save RAM


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", help="Specify the type of the users in the network")
    args = parser.parse_args()

    if args.type == "fake":
        print('fake')
        edges_files = ['crawl-friends/stefi_2', 'crawled-friends/stefi_2']
    elif args.type == "real":
        print('real')
        # generate a small network to analyse the lcc with the users for 
        # which we crawled the friends of 50 of their friends
        edges_files = ['crawl-friends/alex_2',  'crawled-friends/alex_2', 'crawl-friends/ella_2',
        'crawled-friends/ella_2']
    else:
        edges_files = ['crawled-friends/alex_2', 'crawled-friends/stefi_2', 'crawled-friends/ella_2', 
        'crawl-friends/E13_real_friends', 'crawl-friends/FSF_bot_friends', 'crawl-friends/INT_bot_friends', 
        'crawl-friends/TFP_real_friends', 'crawl-friends/TWT_bot_friends']
        
    G = generate_network_lcc(edges_files)
    convert_graph_to_gml(G, "network.gml")
