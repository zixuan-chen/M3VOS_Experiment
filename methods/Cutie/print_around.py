
def print_around(file_path, line_number, column_number, context_size=10):
    # 打开文件
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 确保行号在文件范围内
    if line_number > len(lines):
        print("指定的行号超出了文件的范围。")
        return

    # 获取指定行的内容
    line_content = lines[line_number - 1]  # 减1是因为行号通常是从1开始的

    # 确保列号在行内容范围内
    if column_number > len(line_content):
        print("指定的列号超出了行内容的范围。")
        return

    # 计算上下文的起始和结束位置
    start = max(0, column_number-context_size)
    end = min(len(line_content), column_number + context_size)
    print("length of string is", len(line_content))
    print("start: %d end: %d" % (start, end))
    # 打印出指定行的内容，高亮显示指定的列号位置
    # print(f"Line {line_number}: {line_content}")
    print("Context around column {}:".format(column_number))
    print(line_content[start:end])


def append_to_line(file_path, line_number, append_text):
    try:
        # 读取文件内容到列表中
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 检查指定行号是否在文件范围内
        if line_number > len(lines) or line_number < 1:
            print("行号超出文件范围。")
            return

        # 修改指定行的内容
        lines[line_number - 1] = lines[line_number - 1].rstrip('\n') + append_text + '\n'

        # 将修改后的内容写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

        print(f"已在第 {line_number} 行末尾添加 '{append_text}'。")

    except FileNotFoundError:
        print("文件未找到。")
    except IOError as e:
        print(f"文件操作出错：{e}")
if __name__ == "__main__":
    # file_path = "/home/bingxing2/home/scx8ah2/dataset/BURST/train/train.json"  # 待检查的文件路径
    # line_number = 1  # 指定的行号
    # append_text = '"}}}'  # 要添加的字符串
    # append_to_line(file_path, line_number, append_text)
    file_path = "/home/bingxing2/home/scx8ah2/dataset/BURST/train/train.json"  # 待检查的文件路径
    line_number = 1  # 待检查的行号
    column_number = 25165607  # 待检查的列号
    print_around(file_path, line_number, column_number, context_size=400)