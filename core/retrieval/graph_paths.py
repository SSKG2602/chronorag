class GraphNotConfigured(RuntimeError):
    pass


def get_graph_paths(*_args, **_kwargs):
    raise GraphNotConfigured("Graph retrieval disabled in research scaffold")
