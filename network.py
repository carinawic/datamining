import pandas as pd
import networkx as nx
import networkit as nk
import pickle
import matplotlib.pyplot as plt

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


def create_network(type, dataset_list):
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
        for dataset in dataset_list[1:]:
            temp_users = pd.read_csv("data/{}/users.csv".format(dataset), header=0, usecols=[0, 2, 3, 4, 5, 6, 7, 33])
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
                temp_edges_friends = pd.read_csv("data/{}/{}.csv".format(dataset_list[0], "friends"), header=0, dtype=int)
                temp_edges_followers = pd.read_csv("data/{}/{}.csv".format(dataset_list[0], "followers"), header=0, dtype=int)
                edges = edges.append(temp_edges_followers, ignore_index=True)
                edges = edges.append(temp_edges_friends, ignore_index=True)

    edges.drop_duplicates().reset_index(drop=True)
    # egdes = remove_outsiders(users, edges)

    # export data in csv format
    # users[["id", "dataset"]].to_csv("all_users.csv", index=False, header=True)
    # new_edges.to_csv("edges.csv", index=False, header=True)

    G = generate_directed_network(edges)
    print(f'Number of nodes {G.number_of_nodes()}\nNumber of edges {G.number_of_edges()}')
    # save_pickle(G, "graph.pickle")

    return G


def explore_connectivity(directed_graph):
    """
    Function to study graph connectivity
    :param directed_graph: the graph to be analysed
    """
    # strongly connected - contains a directed path from u to v AND a directed
    # path from v to u for every pair of vertices u, v
    print(f'Strong connectivity {nx.is_strongly_connected(directed_graph)}')
    # print(f'No of SCCs {nx.number_strongly_connected_components(directed_graph)}')

    # connected - contains a directed path from u to v OR a directed path from
    # v to u for every pair of vertices u, v

    # weakly connected - replacing all of G's directed edges with undirected 
    # edges produces a connected (undirected) graph.
    print(f'Weak connectivity {nx.is_weakly_connected(directed_graph)}')
    # print(f'No of WCCs {nx.number_weakly_connected_components(directed_graph)}')

    # number of giant components (how many nodes it contains)
    # number of disconnected components


def compute_graph_stats(directed_graph):
    """
    Function to compute graph properties
    :param directed_graph: the graph to be analysed
    """
    explore_connectivity(directed_graph)
    # node degree power law distribution

    # average in-degree and out-degree


    # diameter
    # output: Found infinite path length because the graph is not connected
    # print(f'Graph diameter {nx.diameter(directed_graph.to_undirected())}')

    # clustering coefficent
    print(f'Clustering coefficent {nx.average_clustering(directed_graph)}')
    
    # eigenvector centrality


def clustering_alg(directed_graph):
    # run a clustering algorithm
    return


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
        save_edges_csv(dataset, edges, kwargs['type'])
    else:
        # create a network with both friends and followers
        friends = pd.read_csv("data/{}/{}.csv".format(dataset, 'friends'), header=0, dtype=int)
        followers = pd.read_csv("data/{}/{}.csv".format(dataset, 'followers'), header=0, dtype=int)
        # drop duplicates might be redundant    
        edges = pd.concat([friends, followers]).drop_duplicates().reset_index(drop=True)
        save_edges_csv(dataset, edges, 'all')
        sources = edges['source_id']
        targets = edges['target_id']
        users = pd.concat([sources, targets]).drop_duplicates().reset_index(drop=True)

    print(f'No of nodes {users.shape[0]}')
    print(f'No of edges {edges.shape[0]}')
    G = generate_directed_network(edges)
    
    return G


def analyse_each_dataset():
    """
    Function to generate and analyse the graph resulting from each of the five datasets
    """
    for dataset in dataset_list:
        print(f'>>>>>>>>>>>> {dataset} <<<<<<<<<<<<')
        G = generate_network(dataset, type='followers')
        compute_graph_stats(G)


def convert_graph_to_gml(G, filepath):
    nx.write_gml(G, filepath)

if __name__ == '__main__':
    # G = create_network("friends", dataset_list)

    humans_graph = create_network("", dataset_list[0:2])
    convert_graph_to_gml(humans_graph, "humans_graph.gml")
    # compute_graph_stats(humans_graph)

    # bots_graph = create_network("", dataset_list[2:4])
    # convert_graph_to_gml(bots_graph, "bots_graph.gml")
    # compute_graph_stats(bots_graph)
