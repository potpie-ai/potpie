import json
import logging
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


class Neo4jDriverSingleton:
    _instance = None
    load_dotenv()

    neo4j_config = {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"),
        "password": os.getenv("NEO4J_PASSWORD"),
        "max_connection_lifetime": int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", 2)),
        "max_connection_pool_size": int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", 2)),
    }

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = GraphDatabase.driver(
                cls.neo4j_config["uri"],
                auth=(cls.neo4j_config["username"], cls.neo4j_config["password"]),
                max_connection_lifetime=cls.neo4j_config["max_connection_lifetime"],
                max_connection_pool_size=cls.neo4j_config["max_connection_pool_size"],
                connection_timeout=30,  # Sets a 30-second timeout for establishing a new connection
                connection_acquisition_timeout=60,  # Sets a 60-second timeout for acquiring a connection from the pool
            )
            # Verify the connection
            try:
                cls._instance.verify_connectivity()
                logging.info("Successfully connected to Neo4j database")
            except Neo4jError as e:
                logging.error(f"Failed to connect to Neo4j: {str(e)}")
                raise
        return cls._instance

    @classmethod
    def close_instance(cls):
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


class Neo4jGraph:
    def __init__(self):
        self.driver = Neo4jDriverSingleton.get_instance()

    def close(self):
        Neo4jDriverSingleton.close_instance()

    def upsert_node(self, function_identifier, properties, project_id):
        properties["project_id"] = project_id
        with self.driver.session() as session:
            session.write_transaction(
                self._upsert_node, function_identifier, project_id, properties
            )

    def add_edge(self, node1_id, node2_id, relationship_type):
        with self.driver.session() as session:
            session.write_transaction(
                self._add_edge, node1_id, node2_id, relationship_type
            )

    def connect_nodes(
        self,
        parent_function,
        called_function_identifier,
        project_id,
        relationship_properties,
    ):
        with self.driver.session() as session:
            session.write_transaction(
                self._connect_nodes,
                parent_function,
                called_function_identifier,
                project_id,
                relationship_properties,
            )

    def find_outbound_neighbors(
        self, endpoint_id, project_id, with_bodies=False, outbound=True, inbound=False
    ):
        with self.driver.session() as session:
            return session.read_transaction(
                self._find_outbound_neighbors,
                endpoint_id,
                project_id,
                with_bodies,
                outbound,
                inbound,
            )

    def get_node_by_id(self, node_id, project_id):
        with self.driver.session() as session:
            return session.read_transaction(self._get_node_by_id, node_id, project_id)

    def find_inbound_neighbors(self, with_bodies=False):
        if with_bodies:
            return """
            MATCH (start:Function {id: $identifier, project_id: $project_id})
            OPTIONAL MATCH (neighbor:Function {project_id: $project_id})-[:CALLS]->(start)
            RETURN start, collect({neighbor: neighbor, body: neighbor.body}) AS neighbors
            """
        else:
            return """
            MATCH (start:Function {id: $identifier, project_id: $project_id})
            OPTIONAL MATCH (neighbor:Function {project_id: $project_id})-[:CALLS]->(start)
            RETURN start, collect(neighbor) AS neighbors
            """

    def fetch_first_order_neighbors(self, node_id, project_id):
        with self.driver.session() as session:
            return session.read_transaction(
                self._fetch_first_order_neighbors, node_id, project_id
            )

    def traverse(self, identifier, project_id, neighbors_fn):
        neighbors_query = neighbors_fn(with_bodies=False)
        with self.driver.session() as session:
            return session.read_transaction(
                self._traverse, identifier, project_id, neighbors_query
            )

    def get_node_file_property(self, identifier, project_id):
        with self.driver.session() as session:
            return session.read_transaction(
                self._get_node_file_property, identifier, project_id
            )

    def add_extends_relationship(self, base_class_id, derived_class_id, project_id):
        with self.driver.session() as session:
            session.write_transaction(
                self._add_extends_relationship,
                base_class_id,
                derived_class_id,
                project_id,
            )

    def get_class_hierarchy(self, class_name, project_id):
        with self.driver.session() as session:
            return session.read_transaction(
                self._get_class_hierarchy, class_name, project_id
            )

    def get_multiple_class_hierarchies(self, classnames, project_id):
        with self.driver.session() as session:
            return session.read_transaction(
                self._get_multiple_class_hierarchies, classnames, project_id
            )

    @staticmethod
    def _get_class_hierarchy(tx, class_name, project_id):
        try:
            query = """
                MATCH (derived:Function {class_name: $class_name, project_id: $project_id})
                CALL {
                    WITH derived
                    MATCH path = (derived)-[:EXTENDS*0..]->(base)
                    RETURN base AS node, length(path) AS depth
                    ORDER BY depth
                }
                RETURN DISTINCT node
                ORDER BY depth
                """
            result = tx.run(query, class_name=class_name, project_id=project_id)

            class_hierarchy = []
            for record in result:
                node = record["node"]
                class_info = {
                    "filepath": node["file"],
                    "class_name": node["class_name"],
                    "start": node["start"],
                    "end": node["end"],
                    "id": node["id"],
                    "project_id": node["project_id"],
                }
                class_hierarchy.append(class_info)

            return class_hierarchy

        except Neo4jError as neo4j_error:
            logging.error(
                f"Neo4j error in fetching class hierarchy for extends relation: {neo4j_error}"
            )
            return []
        except Exception as e:
            logging.error(f"Unexpected error in _get_class_hierarchy: {e}")
            return []

    @staticmethod
    def _get_multiple_class_hierarchies(tx, classnames, project_id):
        try:
            query = """
                MATCH (derived:Function)
                WHERE derived.class_name IN $classnames AND derived.project_id = $project_id
                MATCH path = (derived)-[:EXTENDS*0..]->(base)
                WITH derived.class_name AS root_class_name, base AS node, length(path) AS depth
                ORDER BY root_class_name, depth
                RETURN DISTINCT root_class_name, node, depth
                """
            result = tx.run(query, classnames=classnames, project_id=project_id)

            class_hierarchies = []
            seen_nodes = set()
            for record in result:
                node = record["node"]
                root_classname = record["root_class_name"]

                # Check if we've already added this node
                if node.id in seen_nodes:
                    continue

                seen_nodes.add(node.id)

                class_info = {
                    "filepath": node["file"],
                    "classname": node["class_name"],
                    "start": node["start"],
                    "end": node["end"],
                    "id": node["id"],
                    "root_classname": root_classname,
                    "project_id": node["project_id"],
                }
                class_hierarchies.append(class_info)

            return class_hierarchies

        except Neo4jError as neo4j_error:
            logging.error(
                f"Neo4j error in fetching class hierarchy for extends relation: {neo4j_error}"
            )
            return []
        except Exception as e:
            logging.error(f"Unexpected error in _get_multiple_class_hierarchies: {e}")
            return []

    @staticmethod
    def _delete_nodes_by_project_id(tx, project_id):
        try:
            query = """
            MATCH (n {project_id: $project_id})
            DETACH DELETE n
            """
            tx.run(query, project_id=project_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete nodes with project_id {project_id}: {str(e)}"
            )

    def delete_nodes_by_project_id(self, project_id):
        with self.driver.session() as session:
            session.write_transaction(self._delete_nodes_by_project_id, project_id)

    @staticmethod
    def _upsert_node(tx, function_identifier, project_id, properties):
        # Serialize complex properties to strings if needed
        serialized_properties = {
            key: (json.dumps(value) if isinstance(value, (dict, list)) else value)
            for key, value in properties.items()
        }

        query = (
            "MERGE (n:Function {id: $function_identifier, project_id: $project_id}) "
            "SET n += $properties "
            "RETURN n"
        )
        tx.run(
            query,
            function_identifier=function_identifier,
            properties=serialized_properties,
            project_id=project_id,
        )

    @staticmethod
    def _connect_nodes(
        tx,
        parent_function,
        called_function_identifier,
        project_id,
        relationship_properties,
    ):
        query = (
            "MATCH (a:Function {id: $parent_function, project_id: $project_id}), (b:Function {id: $called_function_identifier, project_id: $project_id}) "
            "MERGE (a)-[r:CALLS]->(b) "
            "SET r += $relationship_properties "
            "RETURN r"
        )
        tx.run(
            query,
            parent_function=parent_function,
            called_function_identifier=called_function_identifier,
            relationship_properties=relationship_properties,
            project_id=project_id,
        )

    @staticmethod
    def _add_edge(tx, node1_id, node2_id, relationship_type):
        query = (
            "MATCH (a:Function {id: $node1_id, project_id: $project_id}), (b:Function {id: $node2_id, project_id: $project_id}) "
            "MERGE (a)-[r:%s]->(b) "
            "RETURN r" % relationship_type
        )
        tx.run(query, node1_id=node1_id, node2_id=node2_id)

    @staticmethod
    def _find_outbound_neighbors(
        tx, endpoint_id, project_id, with_bodies, outbound, inbound
    ):
        match_clause = (
            "MATCH (start:Function {id: $endpoint_id, project_id: $project_id})"
        )
        recursive_match = ""

        if outbound and inbound:
            recursive_match = (
                "MATCH (start)-[:CALLS*]-(neighbor:Function {project_id: $project_id})"
            )
        elif outbound:
            recursive_match = (
                "MATCH (start)-[:CALLS*]->(neighbor:Function {project_id: $project_id})"
            )
        elif inbound:
            recursive_match = (
                "MATCH (start)<-[:CALLS*]-(neighbor:Function {project_id: $project_id})"
            )

        if with_bodies:
            return_clause = "RETURN start, collect({neighbor: neighbor, body: neighbor.body}) AS neighbors"
            query = f"""
                {match_clause}
                CALL {{
                    WITH start
                    {recursive_match}
                    RETURN neighbor, neighbor.body AS body
                }}
                {return_clause}
                """
        else:
            return_clause = "RETURN start, collect(neighbor) AS neighbors"
            query = f"""
                {match_clause}
                CALL {{
                    WITH start
                    {recursive_match}
                    RETURN neighbor
                }}
                {return_clause}
                """

        result = tx.run(query, endpoint_id=endpoint_id, project_id=project_id)
        record = result.single()
        if not record:
            return []

        start_node = dict(record["start"])
        neighbors = record["neighbors"]
        combined = [start_node] + neighbors if neighbors else [start_node]
        return combined

    @staticmethod
    def _get_node_by_id(tx, node_id, project_id):
        query = "MATCH (n:Function {id: $node_id, project_id: $project_id}) RETURN n"
        result = tx.run(query, node_id=node_id, project_id=project_id)
        record = result.single()
        if record:
            return dict(record["n"])
        return None

    @staticmethod
    def _find_inbound_neighbors(tx, endpoint_id, project_id, with_bodies):
        query = f"""
        MATCH (start:Function {{id: $endpoint_id, project_id: $project_id}})
        CALL {{
            WITH start
            MATCH (neighbor:Function {{project_id: $project_id}})-[:CALLS*]->(start)
            RETURN neighbor{', neighbor.body AS body' if with_bodies else ''}
        }}
        RETURN start, collect({{neighbor: neighbor{', body: neighbor.body' if with_bodies else ''}}}) AS neighbors
        """
        result = tx.run(query, endpoint_id=endpoint_id, project_id=project_id)
        record = result.single()
        if not record:
            return []

        start_node = dict(record["start"])
        neighbors = record["neighbors"]
        combined = [start_node] + neighbors if neighbors else [start_node]
        return combined

    @staticmethod
    def _fetch_first_order_neighbors(tx, node_id, project_id):
        query = """
            MATCH (n:Function {id: $node_id, project_id: $project_id})-[:CALLS]->(neighbor:Function)
            RETURN neighbor
            """
        logging.info(
            f"project_id : {project_id}, _fetch_first_order_neighbors: query: {query}"
        )
        result = tx.run(query, node_id=node_id, project_id=project_id)
        return [dict(record["neighbor"]) for record in result]

    @staticmethod
    def _get_node_file_property(tx, identifier, project_id):
        query = """
        MATCH (n:Function {id: $identifier, project_id: $project_id})
        RETURN n.file AS file
        """
        result = tx.run(query, identifier=identifier, project_id=project_id)
        record = result.single()
        if record:
            return record["file"]
        return None

    @staticmethod
    def _traverse(tx, identifier, project_id, neighbors_query):
        query = f"""
        {neighbors_query}
        """
        result = tx.run(query, identifier=identifier, project_id=project_id)
        record = result.single()
        if not record:
            return []

        start_node = dict(record["start"])
        neighbors = [dict(neighbor) for neighbor in record["neighbors"]]
        combined = [start_node] + neighbors if neighbors else [start_node]
        return combined

    @staticmethod
    def _find_neighbors(tx, identifier, project_id, with_bodies, outbound, inbound):
        match_clause = (
            "MATCH (start:Function {id: $identifier, project_id: $project_id})"
        )
        recursive_match = ""

        if outbound and inbound:
            recursive_match = (
                "MATCH (start)-[:CALLS*]-(neighbor:Function {project_id: $project_id})"
            )
        elif outbound:
            recursive_match = (
                "MATCH (start)-[:CALLS*]->(neighbor:Function {project_id: $project_id})"
            )
        elif inbound:
            recursive_match = (
                "MATCH (start)<-[:CALLS*]-(neighbor:Function {project_id: $project_id})"
            )

        if with_bodies:
            return_clause = "RETURN start, collect({neighbor: neighbor, body: neighbor.body}) AS neighbors"
        else:
            return_clause = "RETURN start, collect(neighbor) AS neighbors"

        query = f"""
            {match_clause}
            {recursive_match}
            {return_clause}
        """
        result = tx.run(query, identifier=identifier, project_id=project_id)
        record = result.single()
        if not record:
            return []

        start_node = dict(record["start"])
        neighbors = [dict(neighbor) for neighbor in record["neighbors"]]
        combined = [start_node] + neighbors if neighbors else [start_node]
        return combined

    @staticmethod
    def _add_extends_relationship(tx, base_class_id, derived_class_id, project_id):
        try:
            query = """
                MATCH (base:Function {id: $base_class_id, project_id: $project_id}),
                      (derived:Function {id: $derived_class_id, project_id: $project_id})
                MERGE (derived)-[r:EXTENDS]->(base)
                RETURN r
                """
            tx.run(
                query,
                base_class_id=base_class_id,
                derived_class_id=derived_class_id,
                project_id=project_id,
            )
        except Neo4jError as neo4j_error:
            logging.error(
                f"Neo4j error in creating class hierarchy for extends relation: {neo4j_error}"
            )
        except Exception as e:
            logging.error(f"Error in adding extends relationship in database: {e}")

    def atomic_transaction(self, operations):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                try:
                    for operation in operations:
                        operation(tx)
                    tx.commit()
                except Exception as e:
                    tx.rollback()
                    raise e
