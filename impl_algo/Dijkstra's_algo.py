# Dijkstra's algorithm
# implement Dijkstra


import math
route_list = [[0, 50, 80, 0, 0], [0, 0, 20, 15, 0 ], [0, 0, 0, 10, 15], [0, 0, 0, 0, 30], [0, 0, 0, 0, 0]] # 初期のノード間の距離のリスト

node_num = len(route_list) #ノードの数

unsearched_nodes = list(range(node_num)) # 未探索ノード
distance = [math.inf] * node_num # ノードごとの距離のリスト
previous_nodes = [-1] * node_num # 最短経路でそのノードのひとつ前に到達するノードのリスト
distance[0] = 0 # 初期のノードの距離は0とする

def get_target_min_index(min_index, distance, unsearched_nodes):
    start = 0
    while True:
        index = distance.index(min_index, start)
        found = index in unsearched_nodes
        if found:
            return index
        else:
            start = index + 1

while(len(unsearched_nodes) != 0): #未探索ノードがなくなるまで繰り返す
    # まず未探索ノードのうちdistanceが最小のものを選択する
    posible_min_distance = math.inf # 最小のdistanceを見つけるための一時的なdistance。初期値は inf に設定。
    for node_index in unsearched_nodes: # 未探索のノードのループ
        if posible_min_distance > distance[node_index]: 
            posible_min_distance = distance[node_index] # より小さい値が見つかれば更新
    target_min_index = get_target_min_index(posible_min_distance, distance, unsearched_nodes) # 未探索ノードのうちで最小のindex番号を取得
    unsearched_nodes.remove(target_min_index) # ここで探索するので未探索リストから除去

    target_edge = route_list[target_min_index] # ターゲットになるノードからのびるエッジのリスト
    for index, route_dis in enumerate(target_edge):
        if route_dis != 0:
            if distance[index] > (distance[ target_min_index] + route_dis):
                distance[index] = distance[ target_min_index] + route_dis # 過去に設定されたdistanceよりも小さい場合はdistanceを更新
                previous_nodes[index] =  target_min_index #　ひとつ前に到達するノードのリストも更新

# show results


print("経路")
previous_node = node_num - 1
while previous_node != -1:
    if previous_node !=0:
        print(str(previous_node + 1) + " <- ", end='')
    else:
        print(str(previous_node + 1))
    previous_node = previous_nodes[previous_node]

print("距離")
print(distance[node_num - 1])
