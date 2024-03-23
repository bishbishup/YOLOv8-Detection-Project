# 假设 track_ids 是你提供的列表
track_ids = [[1.0, '杨敏'], [2.0, '杜佳'], [3.0, '沈宇'], [4.0, '徐秀云'], [5.0, '刘静'], [6.0, '胡辉']]
# 假设 results_boxes_data 是另一个列表，这里用一个简单的示例代替
results_boxes_data = [10, 20, 30, 40, 50, 60]

# 使用 zip() 函数同时遍历两个列表
for track_id, person in zip(track_ids, results_boxes_data):
    track_id_number, person_name = track_id
    print("轨迹ID:", track_id_number, "姓名:", person_name, "数据:", person)