import pandas as pd
import networkx as nx
import networkit as nk
import pickle
import matplotlib.pyplot as plt
import argparse
import powerlaw
import csv


def compute_diameter(G):
    # Extract the largest connected component, otherwise the resulting diameter will be infinite
    G = nk.graphtools.toUndirected(G)
    newGraph = nk.components.ConnectedComponents.extractLargestConnectedComponent(G, True)
    no_nodes_lcc = newGraph.numberOfNodes()
    print(f"No of nodes LCC {no_nodes_lcc}")
    # Initialize algorithm to compute the exact diameter of the input graph
    diam = nk.distance.Diameter(newGraph,algo=1)
    diam.run()
    # The return value of getDiameter is a pair of integers, the lower and upper bound of the
    # diameter. In this case, the diameter is the first value of the pair
    print(f"Graph diameter {diam.getDiameter()}")


def clustering(G):
    # The clustering coefficient is computed as the average of the local clustering
    # coefficient over all nodes. The local clustering coefficient focuses on a single
    # node and counts how many possible edges between neighbors of the node exist
    cc = nk.globals.clustering(G)
    print(f"Clustering coefficent {cc}")


# the networkit implementation is based on parallel power iteration
def eigenvector_centrality(G):
    ec = nk.centrality.EigenvectorCentrality(G)
    ec.run()
    top_10 = ec.ranking()[:10] # the 10 most central nodes
    print(top_10)


def degree_distribution(G):
    dd = sorted(nk.centrality.DegreeCentrality(G).run().scores(), reverse=True)
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("number of nodes")
    plt.plot(dd)
    plt.show()

    fit = powerlaw.Fit(dd)
    # Typically the exponent falls in the range 2 < alpha < 3
    print(f"Powerlaw degree distribution {fit.alpha}")


def analyse_connectivity(G):
    scc = nk.components.StronglyConnectedComponents(G)
    scc.run()
    no_scc = scc.numberOfComponents()
    wcc = nk.components.WeaklyConnectedComponents(G)
    wcc.run()
    no_wcc = wcc.numberOfComponents()
    print("No of SCCs %d\nNo of WCCs %d"%(no_scc, no_wcc))


def compute_network_stats(G):
    # analyse_connectivity(G)
    # degree_distribution(G)
    # eigenvector_centrality(G)
    # clustering(G)
    compute_diameter(G)


def compute_graph_features(G, input_file, output_file):
    # compute betweenness centrality
    btwn = nk.centrality.Betweenness(G, normalized=True).run()
    print(btwn.score(84))

    # compute local clustering coefficient
    # if turbo is set to true, the running time is reduced significantly, but it requires
    # O(m) additional memory - in practice, it should be a bit less than half of the memory
    # that is needed for the graph itself
    G_undirected = nk.graphtools.toUndirected(G)
    G_undirected.removeSelfLoops()
    lcc = nk.centrality.LocalClusteringCoefficient(G_undirected, turbo=True).run()
    print(lcc.score(84))
   
    # compute node degree
    d = nk.centrality.DegreeCentrality(G, normalized=True).run()
    print(d.score(84))

    # extract node lables since networkit doen't store additional info about nodes or edges
    # temporary solution, might need to write ids to a file in network.py to save RAM
    G_nx = nx.read_gml(input_file)
    labels_list = list(G_nx.nodes)

    # save results to a CSV file to be used as features for our classifier
    f = csv.writer(open('{}.csv'.format(output_file), 'w'))
    fields = ["used_id", "beetweenness_centrality", "local_clustering_coefficient", "node_degree"]
    f.writerow(fields)
    for u in G.iterNodes():
        f.writerow([labels_list[u], btwn.score(u), lcc.score(u), d.score(u)])


def load_data_gml(filepath):
    G = nk.readGraph(filepath, nk.Format.GML)
    nodes = G.numberOfNodes()
    edges = G.numberOfEdges()
    
    print("No of nodes = %d\nNo of edges = %d"%(nodes, edges))
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="Input file name in gml format")
    parser.add_argument("-o", "--outputfile", help="Output file for graph features")

    args = parser.parse_args()

    if args.inputfile is None:
        parser.error("Please specify input file")
        exit()

    if args.outputfile is None:
        parser.error("Please specify output file")
        exit()

    G = load_data_gml(args.inputfile)
    nk.overview(G)
    # compute_network_stats(G)
    compute_graph_features(G, args.inputfile, args.outputfile)
