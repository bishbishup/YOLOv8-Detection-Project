import face_recognition

# 加载目标人脸的图像并提取特征
target_image = face_recognition.load_image_file("face/1-张建华.png")
target_encoding = face_recognition.face_encodings(target_image)[0]

# 加载数据库中的人脸图像并提取特征
database_image1 = face_recognition.load_image_file("database_face1.jpg")
database_encoding1 = face_recognition.face_encodings(database_image1)[0]

database_image2 = face_recognition.load_image_file("database_face2.jpg")
database_encoding2 = face_recognition.face_encodings(database_image2)[0]

# 计算目标人脸与数据库中其他人脸的相似度
face_distances = [
    face_recognition.face_distance([target_encoding], database_encoding1)[0],
    face_recognition.face_distance([target_encoding], database_encoding2)[0]
]

# 找出相似度最高的人脸
best_match_index = face_distances.index(min(face_distances))
if best_match_index == 0:
    best_match_image = "database_face1.jpg"
    best_match_encoding = database_encoding1
else:
    best_match_image = "database_face2.jpg"
    best_match_encoding = database_encoding2

# 输出匹配到的人脸和相似度
print("找到与目标人脸最相似的人脸：", best_match_image)
print("相似度：", 1 - min(face_distances))