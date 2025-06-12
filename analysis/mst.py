#!/usr/bin/env python
import multiprocessing as mp
from pathlib import Path
from random import sample

import networkx as nx
import osmnx as ox
import pandas as pd

ox.settings.log_console = False

graphml_path = "/data/snm/models/graphml"
save_path = "./mst-results_no_treewidth.csv"
cpus = 18  # mp.cpu_count()


def calculate_stats(G):
    # generate length-weighted MST of G
    T = nx.minimum_spanning_tree(G, weight="length")

    # ratio of edge count in T to edge count in G
    edge_count_orig = len(G.edges)
    edge_count_mst = len(T.edges)
    edge_count_ratio = edge_count_mst / edge_count_orig

    # ratio of edge length in T to edge length in G
    length_orig = sum(nx.get_edge_attributes(G, "length").values())
    length_mst = sum(nx.get_edge_attributes(T, "length").values())
    length_ratio = length_mst / length_orig

    # calculate G's circuit rank and its ratio to edge count in G
    circuit_rank = len(G.edges) - len(G.nodes) + len(list(nx.connected_components(G)))
    circuit_rank_ratio = circuit_rank / len(G.edges)

    # calculate treewidth of G (slow and high-memory consumption)
    treewidth = None
    # treewidth, _ = nx.algorithms.approximation.treewidth_min_degree(nx.Graph(G))

    return (
        edge_count_orig,
        edge_count_mst,
        edge_count_ratio,
        length_orig,
        length_mst,
        length_ratio,
        circuit_rank,
        circuit_rank_ratio,
        treewidth,
    )


def process_graph(fp):
    print(ox.ts(), f"Processing {fp.stem}")

    # load (undirected) graph from file and process even if not fully connected
    G = ox.convert.to_undirected(ox.io.load_graphml(fp))
    stats = calculate_stats(G)

    # same thing again, but only for largest connected component of G
    C = nx.MultiGraph(G.subgraph(max(nx.connected_components(G), key=len)))
    stats_c = calculate_stats(C)

    # what proportion of G is in the C?
    cc_edge_ratio = len(C.edges) / len(G.edges)
    cc_length_ratio = sum(nx.get_edge_attributes(C, "length").values()) / sum(
        nx.get_edge_attributes(G, "length").values()
    )

    results = [fp.stem, *stats, *stats_c, cc_edge_ratio, cc_length_ratio]
    return results


# shuffle the filepaths so 1 cpu doesn't get stuck with the big files
filepaths = list(Path(graphml_path).glob("*/*.graphml"))
filepaths = sample(filepaths, len(filepaths))  # [0:200]
args = ((fp,) for fp in filepaths)
print(ox.ts(), f"There are {len(filepaths):,} total GraphML files")

# multiprocess the queue
with mp.get_context().Pool(cpus) as pool:
    results = pool.starmap_async(process_graph, args).get()

cols = [
    "id",
    "edge_count_orig",
    "edge_count_mst",
    "edge_count_ratio",
    "length_orig",
    "length_mst",
    "length_ratio",
    "circuit_rank",
    "circuit_rank_ratio",
    "treewidth",
    "edge_count_orig_cc",
    "edge_count_mst_cc",
    "edge_count_ratio_cc",
    "length_orig_cc",
    "length_mst_cc",
    "length_ratio_cc",
    "circuit_rank_cc",
    "circuit_rank_ratio_cc",
    "treewidth_cc",
    "cc_edge_ratio",
    "cc_length_ratio",
]
df_results = pd.DataFrame(results, columns=cols)
df_results.to_csv(save_path, index=False)

print(ox.ts(), f"Saved {len(df_results):,} results to {save_path!r}")
