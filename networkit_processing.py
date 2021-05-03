import pandas as pd
import networkx as nx
import networkit as nk
import pickle
import matplotlib.pyplot as plt
import argparse
import powerlaw


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


def load_data_gml(filepath):
    G = nk.readGraph(filepath, nk.Format.GML)
    nodes = G.numberOfNodes()
    edges = G.numberOfEdges()
    
    print("No of nodes = %d\nNo of edges = %d"%(nodes, edges))
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", help="Input file name")

    args = parser.parse_args()

    if args.inputfile is None:
        parser.error("please specify input file")
        exit()

    G = load_data_gml(args.inputfile)
    compute_network_stats(G)
