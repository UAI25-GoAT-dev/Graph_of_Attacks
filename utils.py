import random, copy


def get_shortest_path(leaf_id, graph:dict, depth:int):
    assert leaf_id in graph.keys(), "Provide a valid leaf_id"
    node_id = leaf_id
    shortest_path = [node_id]
    while (node_id!=0) and (len(shortest_path)<depth):
        node_id = graph[node_id]["parent_id"]
        shortest_path.append(node_id)
    shortest_path.reverse()
    return shortest_path


def extract_sub_graph(shortest_path:list, graph:dict):
    assert len(shortest_path) > 0, "Provide a non-empty shortest_path"
    sub_graph = {node_id:[shortest_path[idx+1]] if idx!=len(shortest_path)-1 else [] for idx,node_id in enumerate(shortest_path)}
    q = [shortest_path[0]]
    while q:
        p_id = q.pop(0)
        for node_id,node_content in graph.items():
            if node_content["parent_id"]==p_id:
                q.append(node_id)
                if node_id not in shortest_path:
                    sub_graph[p_id].append(node_id)
                    sub_graph[node_id] = []
    return sub_graph


def get_dfs_path(root_id, sub_graph:dict):
    assert root_id in sub_graph.keys(), "Provide a valid root_id"
    pre_traversal, stack = [], [root_id]
    while stack:
        node_id = stack.pop(-1)
        stack.extend(sub_graph[node_id][::-1])
        pre_traversal.append(node_id)
    return pre_traversal


def get_prompt(conv, graph:dict, depth:int, n_sample:int, mode:str="Seq"):
    leaf_id, Paths, messages = conv.parent_id, [], []
    if graph:
        main_shortest_path = get_shortest_path(leaf_id=leaf_id, graph=graph, depth=depth)
        sub_graph = extract_sub_graph(shortest_path=main_shortest_path, graph=graph)
        leaves_ids = [node_id for node_id in sub_graph.keys() if ((not sub_graph[node_id]) and (node_id!=leaf_id))]
        for l_id in sorted(leaves_ids):
            idx = -1
            shortest_path = get_shortest_path(leaf_id=l_id, graph=graph, depth=depth)
            for i,node_id in enumerate(shortest_path):
                if node_id not in sub_graph.keys():
                    idx = i
            Paths.append(shortest_path[idx+1:])
        random.shuffle(Paths)
        Idx = random.sample(range(len(Paths)), k=len(Paths))
        Paths = list(map(Paths.__getitem__,Idx))[:n_sample]
        Paths.append(main_shortest_path)
        for path in Paths:
            for node_id in path:
                if node_id!=0:
                    messages.extend(graph[node_id]["messages"])


    new_conv = copy.deepcopy(conv)
    new_conv.messages = messages
    usr_in = conv.messages[0]
    usr_in[1] = f"\n__ID__: {conv.self_id}\n__PARENT ID__: {conv.parent_id}\n"+usr_in[1]
    new_conv.messages.append(usr_in)
    new_conv.messages.append(conv.messages[1])
    prompt = new_conv.get_prompt()[:-len(new_conv.sep2)]
    return prompt
