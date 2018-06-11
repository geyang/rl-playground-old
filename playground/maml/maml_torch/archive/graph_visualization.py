from graphviz import Digraph
import torch
from torch.autograd import Variable

import torch_helpers as h


def get_str_id(o):
    return str(id(o))


def make_dot(*roots, name=None, params=None, max_depth=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        :param *nodes torch Variables
        :param name Name of graph, string
        :param params dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
        :param max_depth int
    """
    # assert isinstance(params.values()[0], Variable)
    params_map = {} if params is None else {get_str_id(v): k for k, v in params.items()}

    def add_param(parent, node):
        p = get_str_id(parent)
        n = get_str_id(node)
        dot.edge(p, n)
        dot.node(n, size_to_str(node.size()), fillcolor='orange')

    def add_module(parent, module):
        p = get_str_id(parent)
        m = get_str_id(module)
        dot.edge(p, m)
        dot.node(m, size_to_str(module.size()), fillcolor='orange')

    def remove_backward(name):
        if not name or len(name) < 8:
            return name
        elif name[-16:] == 'BackwardBackward':
            return name[:-16] + ' ⁻¹'
        elif name[-8:] == 'Backward':
            return name[:-8]
        return name

    node_attr = dict(style='filled',
                     fontname='Monospace',
                     fontsize='12',
                     color='white',
                     fillcolor='transparent',
                     shape='rect',
                     align='left',
                     labeldistance="5",
                     ranksep='0.1',
                     height='0.3')
    edge_attr = dict(fontname='Monospace',
                     fontsize='12')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"), edge_attr=edge_attr)

    class PlaceholderNode():
        def __init__(self, text):
            self.text = text

    BRACKETS = ['｛', '｝', '⦃', '⦄']

    def size_to_str(size):
        return '⦃' + (', ').join(['%d' % v for v in size]) + '⦄'

    NAME_TEMPLATE = "{prefix}-{dtype}{size}"

    seen = set()

    def add_nodes(node, parent_id=None, name='', edgelabel='', depth=0):
        depth += 1
        if max_depth is not None and depth > max_depth:
            node = PlaceholderNode('...')
        node_id = get_str_id(node)
        if parent_id is not None:
            dot.edge(node_id, parent_id, label=" {} ".format(edgelabel) if edgelabel else '')
        if node in seen:
            return
        seen.add(node)

        if isinstance(node, PlaceholderNode):
            dot.node(get_str_id(node), node.text, fontcolor='white', fillcolor='grey', shape='Mrecord')
        elif isinstance(node, Variable):
            if node.volatile:
                prefix = 'volatile'
                fillcolor = 'red'
            elif not node.requires_grad:
                prefix = 'const'
                fillcolor = 'pink'
            else:
                prefix = 'var'
                fillcolor = '#23aaff'

            if parent_id is None:  # is root node
                color = 'black'  # todo: might want to do more than this
            else:
                color = None

            dtype = h.TYPES[node.data.__class__].title()

            name = name or NAME_TEMPLATE
            if node_id in params_map:
                name = params_map[node_id]
            node_name = name.format(prefix=prefix, dtype=dtype, size=size_to_str(node.size()))
            dot.node(get_str_id(node), node_name, fontcolor='white', color=color, fillcolor=fillcolor, shape='Mrecord')
        elif torch.is_tensor(node):
            prefix = h.TYPES[node.__class__].title() + 'Tensor'
            node_name = prefix + size_to_str(node.size())
            dot.node(get_str_id(node), node_name, style="rounded, filled", fillcolor='orange', fontcolor='white')
        else:
            node_name = remove_backward(str(type(node).__name__))
            if hasattr(node, 'constant') and node.constant is not None:
                node_name += '⟨{}⟩'.format(node.constant)
                attr = dict(
                    fontcolor='white',
                    color='transparent',
                    fillcolor='orange',
                    shape='Mrecord'
                )
            else:
                attr = dict(
                    color='black',
                    fillcolor='transparent',
                )
            dot.node(get_str_id(node), node_name, **attr)

        if hasattr(node, 'variable') and node.variable is not None:
            add_nodes(node.variable, node_id, edgelabel='variable', depth=depth)
        if hasattr(node, 'grad_fn') and node.grad_fn is not None:
            add_nodes(node.grad_fn, node_id, depth=depth)
        if hasattr(node, 'grad') and node.grad is not None:
            add_nodes(node.grad, node_id, edgelabel='grad', depth=depth)
        if hasattr(node, 'next_functions'):
            for u in node.next_functions:
                if u[0] is None:
                    pass  # todo: add string 'None'
                else:
                    add_nodes(u[0], node_id, depth=depth)

        try:
            if hasattr(node, 'saved_tensors'):
                for t in node.saved_tensors:
                    add_nodes(t, node_id, depth=depth)
        except RuntimeError:
            pass

    for root in roots:
        add_nodes(root, name=name)
    return dot


if __name__ == "__main__":
    import numpy as np
    import torch_helpers as h
    import torch.nn

    # x = h.varify(np.ones(10)) ** h.const(np.random.randn(10)) + 10
    x = h.const(torch.randn(1))
    y = h.varify(np.ones(1))
    fc = torch.nn.Linear(1, 40)
    o = fc(x) + y
    g = make_dot(o, max_depth=3)
    g.render('graphviz_test/example')
