# my_dict = {}  # 创建一个空字典
#
# my_dict[1.0] = 'value'  # 给字典添加键值对
#
# print(my_dict)
# print(my_dict[1.0])

my_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

# 判断字典中是否存在指定的键
if 'key5' in my_dict.keys():
    print('字典中存在键 "key2"')
else:
    print('字典中不存在键 "key2"')