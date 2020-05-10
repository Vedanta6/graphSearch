from collections import deque


class Node:

    def __init__(self, node_id, children_ids=None):
        self._id = node_id
        self._children = list() if children_ids is None else list(children_ids)
        self._visited = {1: False}

    @property
    def children(self):
        return self._children

    @property
    def id(self):
        return self._id

    def add_children(self, children):
        self._children.extend(children)

    def visit(self, print_message=True, search_id=1):
        if print_message:
            print(f'Search #{search_id}: visiting Node {self.id}')
        self._visited.update({search_id: True})

    def was_visited(self, search_id=1):
        return self._visited[search_id] if search_id in self._visited else False

    def reset_visit(self):
        self._visited = {1: False}


class Graph:

    def __init__(self, graph_dict: dict):
        self.nodes = [Node(node_id=n_id) for n_id in graph_dict]
        for node_i, children_ids in enumerate(graph_dict.values()):
            self.nodes[node_i].add_children([n for n in self.nodes if n.id in children_ids])

    def get_nodes_ids(self):
        return [n.id for n in self.nodes]

    def clear_nodes_status(self):
        [n.reset_visit() for n in self.nodes]

    def get_node_by_id(self, node_id):
        found_nodes = [n for n in self.nodes if n.id == node_id]
        return found_nodes[0] if len(found_nodes) > 0 else None

    def show_edges(self):
        graph_level = ((((n.id, ch.id) for ch in n.children) for n in self.nodes))
        for node_level in graph_level:
            for children_level in node_level:
                print(children_level)

    def run_dfs_search_recursion(self, root_node: Node, target_node: Node, search_id=1, path_tracing=None):
        root_node.visit(search_id=search_id)
        path_tracing = {root_node.id: None} if path_tracing is None else path_tracing
        if id(root_node) != id(target_node):
            for child_node in root_node.children:
                if not child_node.was_visited(search_id=search_id):
                    path_tracing.update({child_node.id: root_node.id})
                    if self.run_dfs_search_recursion(
                        child_node, target_node, search_id=search_id, path_tracing=path_tracing
                    ):
                        return path_tracing
        else:
            return True

    @staticmethod
    def restore_path(node_id, parents_dict):
        t_node = node_id
        restored_path = [t_node]
        while t_node:
            t_node = parents_dict[t_node]
            if t_node is not None:
                restored_path.append(t_node)

        return restored_path[::-1]

    def run_dfs_search(self, root_node: Node, target_node: Node, search_id=1):
        path_tracing = self.run_dfs_search_recursion(root_node, target_node, search_id=search_id)
        return self.restore_path(target_node.id, path_tracing)

    @staticmethod
    def run_bfs_search(root_node: Node, target_node: Node, print_message=True, search_id=1):
        queue = deque()
        root_node.visit(print_message=print_message, search_id=search_id)
        queue.append(root_node)
        path_tracing = {root_node.id: None}
        while queue:
            node = queue.popleft()

            for child in node.children:
                if not child.was_visited(search_id=search_id):
                    child.visit(print_message=print_message, search_id=search_id)
                    path_tracing[child.id] = node.id

                    yield child, path_tracing

                    if id(child) == id(target_node):
                        queue.clear()
                        break
                    else:
                        queue.append(child)

    def run_full_bfs_search(self, root_node: Node, target_node: Node, print_nodes=True):
        graph_gen = self.run_bfs_search(root_node, target_node)
        while True:
            try:
                node, parents_dict = next(graph_gen)
                graph_path = self.restore_path(node.id, parents_dict)
                if print_nodes:
                    print(graph_path)
                if id(node) == id(target_node):
                    return graph_path
            except StopIteration:
                break

    @staticmethod
    def intersection_exists(gen_paths, gen_another_paths):
        intersections = set(gen_paths).intersection(gen_another_paths)
        if intersections:
            return intersections
        else:
            return None

    def run_bidirectional_search(self, one_node, another_node):
        graph_gen = self.run_bfs_search(one_node, another_node, print_message=True, search_id=1)
        graph_another_gen = self.run_bfs_search(another_node, one_node, print_message=True, search_id=2)
        gen_paths = dict()
        gen_another_paths = dict()
        for _ in range(len(self.nodes)):
            exceptions_counter = 0
            # run 1 step of BFS for one_node
            try:
                one_node_, parents_dict = next(graph_gen)
                gen_paths.update(parents_dict)
            except StopIteration:
                exceptions_counter += 1
            # run 1 step of BFS for another_node
            try:
                another_node_, another_parents_dict = next(graph_another_gen)
                gen_another_paths.update(another_parents_dict)
            except StopIteration:
                exceptions_counter += 1
            intersections = self.intersection_exists(gen_paths, gen_another_paths)
            if intersections:
                intersections = list(intersections)
                one_intersection_id = intersections[0]
                another_interaction_id = intersections[-1]
                one_path = self.restore_path(one_intersection_id, gen_paths)
                another_path = self.restore_path(another_interaction_id, gen_another_paths)
                merged_path = one_path[:-1 if len(intersections) == 1 else len(one_path)] + another_path[::-1]
                print(merged_path)
                return merged_path
            elif exceptions_counter > 1:
                break


    def search(self, root_node_id=None, target_node_id=None, algorithm='DFS'):
        if not self.nodes:
            return 'Graph has not been configured'
        else:
            nodes_ids = self.get_nodes_ids()
            if root_node_id is None:
                root_node_id = min(nodes_ids)
            if target_node_id is None:
                target_node_id = max(nodes_ids)
            root_node = self.get_node_by_id(root_node_id)
            print(f'Root node is set to {root_node.id}')
            target_node = self.get_node_by_id(target_node_id)
            print(f'Target node is set to {target_node.id}')
            graph_path = None
            if algorithm == 'DFS':
                graph_path = self.run_dfs_search(root_node, target_node)
            elif algorithm == 'BFS':
                graph_path = self.run_full_bfs_search(root_node, target_node, print_nodes=False)
            elif algorithm == 'Bidirectional':
                graph_path = self.run_bidirectional_search(root_node, target_node)
            else:
                print('Algorithm has not been implemented')
            if graph_path is not None:
                print('Resulting path: ', graph_path)



graph_itself = {
    0: {1, 5},
    1: {3, 4},
    2: {1},
    3: {2, 4},
    4: {},
    5: {}

}

graph = Graph(graph_itself)
graph.search(target_node_id=3)
