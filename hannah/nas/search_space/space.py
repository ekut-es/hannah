import networkx as nx
import numpy as np
import hannah.nas.search_space.utils as utils

from copy import deepcopy
from hannah.nas.search_space.connectivity_constrainer import ConnectivityGenerator


class Cell(nx.DiGraph):
    def __init__(self, ops=None) -> None:
        super().__init__()
        if ops:
            self.add_sequential_operators(ops)

    def add_sequential_operators(self, operators):
        operators = list(operators)
        last_node = None if not self.nodes else list(self.nodes)[-1]
        for op in operators:
            # node, node_data = op.to_node()
            node = op.new()  # deepcopy(op)
            self.add_node(node)
            if last_node:
                self.add_edge(last_node, node)
            last_node = node

    def get_attribute(self, key):
        attr_dict = {}
        for n in self.nodes:
            attr_dict[n] = n.attrs[key]
        return attr_dict

    def new(self):
        return deepcopy(self)


class Subgraph(nx.DiGraph):
    def __init__(self, cells, connectivity_graph) -> None:
        super().__init__()
        self.cells = cells
        self.connectivity_graph = connectivity_graph
        self.instance = False

        # assuming cells is either a dict or list with cells[i] = cell for
        # corresponding node i in connectivity graph
        out_dict = {}
        in_dict = {}
        for c_node in self.connectivity_graph.nodes:
            nodes = [(c, {
                     'connectivity': c_node,
                     'id': '{}_{}_{}'.format(c.attrs['op'], c_node, i).replace("['", '').replace("']", '')
                     })
                     for i, c in enumerate(cells[c_node])]

            self.add_nodes_from(nodes)
            self.add_edges_from(cells[c_node].edges())
            cell_nodes = list(cells[c_node])
            out_dict[c_node] = cell_nodes[-1]
            if len(cell_nodes) > 1:
                in_dict[c_node] = {'regular': cell_nodes[0]}
                in_dict[c_node].update({'add': node for node in cell_nodes if 'add' in node.attrs['op']})
            else:
                if 'add' in cell_nodes[0].attrs['op']:
                    in_dict[c_node] = {'regular': cell_nodes[0], 'add': cell_nodes[0]}
                else:
                    in_dict[c_node] = {'regular': cell_nodes[0]}

        for c_node in self.connectivity_graph.nodes:
            in_edges = self.connectivity_graph.in_edges(c_node)
            for i, e in enumerate(in_edges):
                if i == 0:
                    self.add_edge(
                        out_dict[e[0]],
                        in_dict[e[1]]['regular']
                    )
                else:
                    self.add_edge(
                        out_dict[e[0]],
                        in_dict[e[1]]['add']
                    )

    def get_attribute(self, key):
        attr_dict = {}
        for n in self.nodes:
            if key in n.attrs:
                attr_dict[n] = n.attrs[key]
            else:
                attr_dict[n] = None
        return attr_dict

    def get_global_attr_options(self):
        global_attrs = {}
        for node in self.nodes:
            for key, values in node.attrs.items():
                if key in global_attrs:
                    global_attrs[key].extend(values)
                else:
                    global_attrs[key] = list(values)

        for key, values in global_attrs.items():
            global_attrs[key] = list(set(values))
        return global_attrs

    def instantiate(self, cfg):
        new_graph = self.new()
        if isinstance(cfg, dict):
            for node in new_graph.nodes:
                node_id = new_graph.nodes[node]['id']
                node.instantiate(cfg[node_id])
        else:
            index_start = 0
            for node in new_graph.nodes:
                index_end = index_start + len(node.attrs)
                node.instantiate(cfg[index_start:index_end])
                index_start = index_end

        new_graph.instance = True
        return new_graph

    def infer_shapes(self, input):
        # for node in self.nodes:
        for node in nx.topological_sort(self):
            args = []
            in_edges = self.in_edges(node)
            if len(in_edges) == 0:
                args = np.array([input])
                self.nodes[node]['output_shape'] = node.infer_shape(args)
            else:
                for u, v in self.in_edges(node):
                    args.append(self.nodes[u]['output_shape'])

            args = np.array(args)
            self.nodes[node]['output_shape'] = node.infer_shape(args)
            # print(self.nodes[node]['id'], self.nodes[node]['output_shape'])

    def get_draw_coord(self, vertical=False, scale=1):
        con_pos = utils.get_node_coord(self.connectivity_graph)
        pos = {}
        for i, node in enumerate(self.nodes):
            con = self.nodes[node]['connectivity']
            if vertical:
                pos[node] = (con_pos[con][1] * scale, i)
            else:
                pos[node] = (i, con_pos[con][1] * scale)

        return pos

    def new(self):
        return deepcopy(self)


class Space:
    def __init__(self,
                 cell_dict: dict,
                 connectivity_gen: ConnectivityGenerator,
                 max_stack=[1]
                 ) -> None:
        self.cell_dict = cell_dict
        self.connectivity_gen = connectivity_gen
        self.default_graph = self.get_default_graph()
        self.knobs = self.get_knobs()

    def get(self, cfg):
        connect_graph = self.connectivity_gen.get_dag(cfg)
        subgraph = Subgraph(cells=self.cell_dict, connectivity_graph=connect_graph)
        instance = subgraph.instantiate(cfg)
        return instance

    def get_random(self, random_state=None):
        random_cfg_vec = utils.get_random_cfg_vec(self.knobs, random_state=random_state)
        random_cfg = utils.vec_to_knob(random_cfg_vec, self.knobs)
        return self.get(random_cfg)

    def get_knobs(self):
        knobs = {}
        knobs.update(self.connectivity_gen.get_knobs())

        for node in self.default_graph:
            node_id = self.default_graph.nodes[node]['id']
            knobs[node_id] = node.get_knobs()
        return knobs

    def get_default_graph(self):
        c_graph = self.connectivity_gen.get_default_graph()
        return Subgraph(cells=self.cell_dict,
                        connectivity_graph=c_graph)
