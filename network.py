import pandas as pd
import networkx as nx
import pickle
import matplotlib.pyplot as plt

dataset_list = ["E13", "FSF", "INT", "TFP", "TWT"]


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


def create_network(type):
    """
    Function to generate and save a networkx graph
    :param type: 'friends' or 'followers'
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
        users = pd.read_csv("data/{}/users.csv".format(dataset_list[0]), header=0, usecols=[0, 2, 3, 4, 5, 6, 7])
        for dataset in dataset_list[1:]:
            temp_users = pd.read_csv("data/{}/users.csv".format(dataset), header=0, usecols=[0, 2, 3, 4, 5, 6, 7])
            temp_users["dataset_name"] = dataset
            users = users.append(temp_users, ignore_index=True)
            users.append(temp_users, ignore_index=True)
        #save_pickle(users, "users_df.pickle")

    if not edges_exist:
        edges = pd.read_csv("data/{}/{}.csv".format(dataset_list[0], type), header=0, dtype=int)
        for dataset in dataset_list[1:]:
            temp_edges = pd.read_csv("data/{}/{}.csv".format(dataset, type), header=0, dtype=int)
            edges.append(temp_edges, ignore_index=True)
        #save_pickle(edges, "{}_edges_df.pickle".format(type))

    new_edges = remove_outsiders(users, edges)

    G = generate_directed_network(new_edges)
    #save_pickle(G, "graph.pickle")

    return G


if __name__ == '__main__':
    G = create_network("friends")
    draw_directed_graph(G)