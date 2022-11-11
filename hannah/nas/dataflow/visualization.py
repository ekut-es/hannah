import networkx as nx
import pyvis as pv


def opgraph_to_nx(graph):
    pos = {}
    x = 0
    y = 0
    queue = [graph]
    visited = []
    nx_graph = nx.DiGraph()
    while queue:
        node = queue.pop(-1)

        visited.append(node)
        nx_graph.add_node(node.id)

        for operand in node.operands:
            nx_graph.add_edge(operand.id, node.id)
            if operand not in visited:
                queue = [operand] + queue

    return nx_graph


def draw(graph):
    G = opgraph_to_nx(graph)

    nt = pv.network.Network('1000px', '1000px', layout=True)
    nt.from_nx(G)
    nt.toggle_physics(False)
    nt.show('nx.html')


