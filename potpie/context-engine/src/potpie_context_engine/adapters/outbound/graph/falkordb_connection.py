"""Select the transaction client for both FalkorDB driver shapes."""


def transaction_client(graph):
    client = graph.client
    # Server Graph.client is FalkorDB(connection=Redis); Lite Graph.client
    # may already be Redis, whose optional connection attribute is None.
    connection = getattr(client, "connection", None)
    return client if connection is None else connection
