# 定义一个函数来读取文件并返回行的列表
def read_file_lines(filename):
    with open(filename, 'r') as file:
        return file.readlines()

# 读取两个文件的内容
file1_lines = read_file_lines('from_all_video_scores.txt')
file2_lines = read_file_lines('from_phase&videos_score.txt')

# 将文件内容从列表中的字符串转换为集合，去除空白字符，并转换为小写（如果需要）
file1_set = set([line.strip().lower() for line in file1_lines])
file2_set = set([line.strip().lower() for line in file2_lines])

# 找出在第一个文件中但不在第二个文件中的行
unique_to_file1 = file1_set - file2_set

# 找出在第二个文件中但不在第一个文件中的行
unique_to_file2 = file2_set - file1_set

# 打印结果
print("Lines unique to from_all_video_scores.txt:")
for line in unique_to_file1:
    print(line)

print("\nLines unique to from_phase&videos_score.txt:")
for line in unique_to_file2:
    print(line)