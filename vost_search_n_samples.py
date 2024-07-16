

def vost_select_samples(input_file, output_file, max_select=5):

    # 假设我们选择每个动词至多max_select个条目

    # 读取原始文件并按动词分组
    data = []
    with open(input_file, 'r') as file:
        for line in file:
            data.append(line.strip())

    # 按动词对数据进行分组
    verb_to_items = {}
    for item in data:
        parts = item.split('_')
        verb = parts[1]
        if verb not in verb_to_items:
            verb_to_items[verb] = []
        verb_to_items[verb].append(item)

    # 对每个动词选择至多n个条目
    selected_items = {}
    for verb, items in verb_to_items.items():
        selected_items[verb] = items[:max_select]

    # 将结果写入到另一个文件
    with open(output_file, 'w') as file:
        for verb in selected_items:
            file.write('\n'.join(selected_items[verb]))
            file.write('\n')

    print(f"Selected items for each verb have been written to {output_file}")

if __name__ == "__main__":
    vost_select_samples("methods/VOST/ImageSets/train.txt", "methods/VOST/ImageSets/subtrain.txt", 10)