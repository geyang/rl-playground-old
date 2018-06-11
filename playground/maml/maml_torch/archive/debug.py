from graph_visualization import make_dot


def graph(*var, params=None, path='figures/debug_graph', name="", block=True, max_depth=None):
    dot = make_dot(*var, name=name, max_depth=max_depth, params=params)
    dot.render(path.format(name=name))
    if block:
        raise Exception('Stop Here')
