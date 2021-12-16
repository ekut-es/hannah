import networkx as nx
import numpy as np
import itertools


class ConnectivityGenerator:
    def get_random_dag(self):
        raise NotImplementedError

    def get_dag(self, cfg):
        raise NotImplementedError

    def get_default_graph(self):
        raise NotImplementedError

    def get_knobs(self):
        raise NotImplementedError


class PathConstrainer(ConnectivityGenerator):
    def __init__(self, max_parallel_paths, max_nodes, stack=[1], share_dag=False) -> None:
        self.max_parallel_paths = max_parallel_paths
        self.max_nodes = max_nodes
        self.complete_dag = []
        self.paths = []
        self.dag = None
        self.share_dag = share_dag
        self.stack = stack

        complete_dag = nx.DiGraph()
        complete_dag.add_nodes_from([i for i in range(self.max_nodes)])
        add_edges_densly(complete_dag)

        self.paths = list(nx.all_simple_edge_paths(complete_dag, 0, max(complete_dag.nodes)))
        self.complete_dag = complete_dag

    def get_random_dag(self):
        if len(self.complete_dag) == 1:
            return self.complete_dag
        idx = np.random.choice(range(len(self.paths)), size=self.max_parallel_paths)
        subgraph = self.get_dag(idx)
        return subgraph

    def get_dag(self, cfg):
        path_indices = [v for k, v in cfg['paths'].items()]
        assert len(path_indices) == self.max_parallel_paths
        chosen_paths = []
        for i in path_indices:
            chosen_paths.extend(self.paths[i])

        chosen_paths = [tuple(e) for e in chosen_paths]
        subgraph = nx.edge_subgraph(self.complete_dag, chosen_paths).copy()
        return subgraph

        # dag = nx.DiGraph()
        # idx = 0
        # new_nodes = None
        # for s in range(self.stack[cfg['stack']]):
        #     if new_nodes:
        #         idx = new_nodes[-1]
        #     new_nodes = [n + idx + 1 for n in subgraph.nodes]
        #     new_edges = [(idx + u, idx + v) for u, v in subgraph.edges]

        #     dag.add_nodes_from(new_nodes)
        #     dag.add_edges_from(new_edges)
        #     if new_nodes[0] > 0:
        #         dag.add_edge(new_nodes[0] - 1, new_nodes[0])
        # return dag

    def get_default_graph(self):
        return self.get_dag({'paths': {i: 0 for i in range(self.max_parallel_paths)}})

    def get_knobs(self):
        knobs = {'paths': {}}
        for i in range(self.max_parallel_paths):
            knobs['paths'][i] = list(range(len(self.paths)))

        return knobs

    def reset_dag(self):
        self.dag = None

    def get_paths(self):
        return self.paths

    def get_path(self, i):
        return self.paths[i]

    def enumerate_path_combinations(self):
        return itertools.combinations_with_replacement(range(len(self.paths)), self.max_parallel_paths)


class DARTSCell(nx.DiGraph):
    def __init__(self, num_inputs=2, num_nodes=4, num_outputs=1) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.num_outputs = num_outputs

        input_nodes = list(range(num_inputs))
        intermediate_nodes = list(range(num_inputs, num_inputs + num_nodes))
        output_nodes = list(range(num_inputs + num_nodes, num_inputs + num_nodes + num_outputs))

        self.add_nodes_from([(n, {'type': 'intermediate'}) for n in intermediate_nodes])
        add_edges_densly(self)

        self.add_nodes_from([(n, {'type': 'input'}) for n in input_nodes])
        self.add_edges_from([(u, v) for u in input_nodes for v in intermediate_nodes])

        self.add_nodes_from([(n, {'type': 'output'}) for n in output_nodes])
        self.add_edges_from([(u, v) for u in intermediate_nodes for v in output_nodes])


class DARTSMakroarchitecture(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.input_nodes = []
        self.output_nodes = []
        self.intermediate_nodes = []

    def add_cells(self, cells):
        new_nodes = [(0, {'type': 'input'})]
        cell_nodes = []
        cell_nodes.append(new_nodes)
        self.add_nodes_from(new_nodes)
        for i, cell in enumerate(cells):
            idx = new_nodes[-1][0]
            types = nx.get_node_attributes(cell, 'type')
            new_nodes = [(n + idx + 1, {'type': types[n], 'cell': i}) for n in sorted(cell.nodes)]
            new_edges = [(idx + 1 + u, idx + 1 + v) for u, v in sorted(cell.edges)]

            self.add_nodes_from(new_nodes)
            self.add_edges_from(new_edges)

            if i == 0:
                self.add_edges_from([(0, new_nodes[n][0]) for n in range(cell.num_inputs)])
            else:
                cell_edges = [(cell_nodes[-cell.num_inputs+n][-1][0], new_nodes[n][0]) for n in range(cell.num_inputs)]
                self.add_edges_from(cell_edges)
            cell_nodes.append(new_nodes)

    def to_line_graph(self, redux_cell_indices=[]):
        g = nx.line_graph(self)
        nodes = list(g.nodes)
        node_types = nx.get_node_attributes(self, 'type')
        cell_map = nx.get_node_attributes(self, 'cell')
        cell_map.update({0: -1})
        redux_cells = np.zeros(max(cell_map.values()) + 1)
        for index in redux_cell_indices:
            redux_cells[index] = 1

        g.add_nodes_from(['in', 'out'])
        g.add_edges_from([('in', (u, v)) for u, v in nodes if u == 0])
        g.add_edges_from([((u, v), 'out') for u, v in nodes if v == len(self.nodes)-1])
        node_labels = {i: old for i, old in enumerate(nx.topological_sort(g))}

        g = nx.relabel_nodes(g, {old: i for i, old in enumerate(nx.topological_sort(g))})
        for node in g.nodes:
            n = node_labels[node]
            if n == 'in':
                g.nodes[node]['type'] = 'identity'
            elif n == 'out':
                g.nodes[node]['type'] = 'concat'
            else:
                u, v = n
                if node_types[v] == 'input':
                    if v > u+2 and (u, u+2) in node_labels.values() and redux_cells[cell_map[u+2]]:
                        g.nodes[node]['type'] = 'factorize_reduce'

                    else:  # elif node_types[u] == 'output':
                        g.nodes[node]['type'] = 'concat'

                elif node_types[v] == 'output':
                    g.nodes[node]['type'] = 'add'
                elif node_types[u] == 'input' and redux_cells[cell_map[u]]:
                    g.nodes[node]['cell'] = cell_map[v]
                    g.nodes[node]['type'] = 'redux'
                else:
                    g.nodes[node]['cell'] = cell_map[v]
                    g.nodes[node]['type'] = 'op'

        g.node_labels = node_labels
        return g


class DARTSGraph(ConnectivityGenerator):
    def __init__(self, dart_cells, redux_cell_indices=[]) -> None:
        super().__init__()
        self.ooe = DARTSMakroarchitecture()
        self.ooe.add_cells(dart_cells)
        self.m_arch = self.ooe.to_line_graph(redux_cell_indices)

    def get_knobs(self):
        return {}

    def get_default_graph(self):
        return self.m_arch

    def get_dag(self, cfg):
        return self.m_arch

    def get_random_dag(self):
        return self.m_arch


# borrowed from NASLib
def get_dense_edges(g):
    """
    Returns the edge indices (i, j) that would make a fully connected
    DAG without circles such that i < j and i != j. Assumes nodes are
    already created.
    Returns:
        list: list of edge indices.
    """
    edges = []
    nodes = sorted(list(g.nodes()))
    for i in nodes:
        for j in nodes:
            if i != j and j > i:
                edges.append((i, j))
    return edges


def add_edges_densly(g):
    """
    Adds edges to get a fully connected DAG without cycles
    """
    g.add_edges_from(get_dense_edges(g))


def get_node_coord(g, first_node=0):
    positions = {first_node: [0, 0]}
    prev = 0
    d = 0
    for e in nx.edge_dfs(g):
        x = e[1] if isinstance(e[1], int) else int(e[1].split('_')[0])
        if x < prev:
            d += 1
        # else:
        #     d = min(0, d-1)
        prev = x
        positions[e[1]] = [x, d]

    cur_x = 0
    ct = 0.2
    step = 0.2
    for key, pos in positions.items():
        if pos[0] == cur_x:
            pos[0] = cur_x + ct
            ct += step
        else:
            cur_x = pos[0]
            ct = step

    return positions
