# import networkx as nx
# import matplotlib.pyplot as plt
# from networkx.drawing.nx_pydot import graphviz_layout


# class Visualizer:
#     def __init__(self, graph) -> None:
#         self.graph = graph
#         self.nx_graph = nx.DiGraph()

#         queue = [self.graph]
#         visited = []
#         while queue:
#             n = queue.pop()
#             visited.append(n)
#             self.nx_graph.add_node(n.id, type=str(type(n)).split('.')[-1].split('\'')[0])

#             for operand in n.operands:
#                 self.nx_graph.add_edge(operand.id, n.id)
#                 if operand not in visited:
#                     queue.append(operand)

#     def draw(self):
#         pos = graphviz_layout(self.nx_graph, prog="dot", root='input')
#         # pos['input'] = (0, 1200)
#         self.nx_graph.graph["graph"] = dict(rankdir="LR")
#         labels = {}
#         for n in self.nx_graph.nodes:
#             labels[n] = self.nx_graph.nodes[n]['type']
#         nx.draw(self.nx_graph, pos, node_color='y')
#         nx.draw_networkx_labels(self.nx_graph, pos, labels=labels, font_size=8)
#         plt.show()
