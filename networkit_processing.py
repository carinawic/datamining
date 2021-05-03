import pandas as pd
import networkx as nx
import networkit as nk
import pickle
import matplotlib.pyplot as plt
import argparse
import powerlaw


def degree_distribution(G):
    dd = sorted(nk.centrality.DegreeCentrality(G).run().scores(), reverse=True)
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("number of nodes")
    plt.plot(dd)
    plt.show()

    fit = powerlaw.Fit(dd)
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
    degree_distribution(G)



def load_data_gml(filepath):
    G = nk.readGraph(filepath, nk.Format.GML)
    nodes = G.numberOfNodes()
    edges = G.numberOfEdges()
    
    print("#nodes = %d, edges = %d"%(nodes, edges))
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
    
