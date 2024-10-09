#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

# LIBRARY IMPORTS
import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
    draw_networkx_nodes,
    draw_networkx_edges,

)
import matplotlib
from operator import itemgetter
import random
random.seed(9001) # For reproducibility
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")


# METADATAS
__author__ = "Essmay Touami"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Essmay Touami"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Essmay Touami"
__email__ = "essmay.touami@etu.u-paris.fr"
__status__ = "Developpement"


# FUNCTIONS
def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'r') as f:
        while True:
            f.readline() # ignore the first line starting with '@'
            sequence = f.readline().strip()  
            f.readline()  # ignore the line starting with '+'
            f.readline()  # ignore the quality line
            if not sequence:
                break
            yield sequence
    

def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i : i + kmer_size] # return a kmer of size kmer_size


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    #  read the fastq file and cut the reads into kmers
    for read in read_fastq(fastq_file):
        # count the number of occurrences of each kmer
        for kmer in cut_kmer(read, kmer_size):
            # kmer already seen at least once
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    # create a directed graph
    graph = DiGraph()

    # add the edges between the kmers
    for kmer, weight in kmer_dict.items():
        # add the first k-1 prefix of the kmer
        prefix = kmer[:-1]
        # add the last k-1 suffix of the kmer
        suffix = kmer[1:]
        # add the edge between the prefix and the suffix
        # with the weight that is the number of occurrences of the kmer
        graph.add_edge(prefix, suffix, weight=weight)
    
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if delete_entry_node and delete_sink_node:
            # Remove all nodes in the path
            graph.remove_nodes_from(path)
        elif delete_entry_node:
            # Remove all nodes except the last one
            graph.remove_nodes_from(path[:-1])
        elif delete_sink_node:
            # Remove all nodes except the first one
            graph.remove_nodes_from(path[1:])
        else:
            # Remove all nodes except the first and last one
            graph.remove_nodes_from(path[1:-1])


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    best_path = None
    best_score = -1 # Lower bound
    for i, path in enumerate(path_list):
        # Score based on frequency and length
        score = weight_avg_list[i] + path_length[i]
        # Add randomness
        score += random.random()
        # Update the best path
        if score > best_score:
            best_score = score
            best_path = path

    if best_path:
        # Remove all paths except the best one
        remove_paths(graph, path_list, delete_entry_node, delete_sink_node)
    
    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    # Get all simple paths between the ancestor and descendant
    paths = list(all_simple_paths(graph, source=ancestor_node, target=descendant_node))
    
    # Calculate lengths and weights for each path
    lengths = []
    weights = []
    for path in paths:
        weight = sum(graph.subgraph(path).edges(data=True)[0][2]['weight'] for edge in graph.subgraph(path).edges(data=True))
        lengths.append(len(path))
        weights.append(weight)
    
    # Select the best path
    return select_best_path(graph, paths, lengths, weights, delete_entry_node=True, delete_sink_node=True)


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble_detected = True
    
    while bubble_detected:
        bubble_detected = False
        
        # Iterate over each node in the graph
        for node in graph.nodes:
            predecessors = list(graph.predecessors(node))
            
            # If the node has more than one predecessor, we might have a bubble
            if len(predecessors) > 1:
                # Check each pair of predecessors for a common ancestor
                for i in range(len(predecessors)):
                    for j in range(i + 1, len(predecessors)):
                        ancestor = nx.lowest_common_ancestor(graph, predecessors[i], predecessors[j])
                        
                        if ancestor is not None:
                            bubble_detected = True
                            # Resolve the bubble between the ancestor and the current node
                            graph = solve_bubble(graph, ancestor, node)
                            break
                    if bubble_detected:
                        break
            if bubble_detected:
                break  # Break outer loop if a bubble is detected
    
    return graph


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    pass


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    pass


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    starting_nodes = [node for node in graph.nodes() if len(list(graph.predecessors(node))) == 0]
    return starting_nodes


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    sink_nodes = [node for node in graph.nodes() if len(list(graph.successors(node))) == 0]
    return sink_nodes


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    # Iterate over all possible pairs of starting and ending nodes
    for start_node in starting_nodes:
        for end_node in ending_nodes:
            # Check if path exist between the starting and ending nodes
            if has_path(graph, start_node, end_node):
                # Get all simple path
                paths = all_simple_paths(graph, start_node, end_node)
                # Iterate over all simple paths
                for path in paths:
                    contig = path[0] # Initialize the contig with the first node
                    for node in path[1:]:
                        contig += node[-1] # Add the last character of the node to the contig
                    contigs.append((contig, len(contig)))
    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w") as f:
        for i, (contig, length) in enumerate(contigs_list):
            f.write(f">contig_{i} len={length}\n")
            f.write(f"{contig}\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = random_layout(graph)
    draw_networkx_nodes(graph, pos, node_size=6)
    draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Plot the graph
    if args.graphimg_file:
        draw_graph(args.graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
