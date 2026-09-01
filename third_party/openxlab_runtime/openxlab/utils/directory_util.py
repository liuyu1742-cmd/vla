import os


def print_directory(path='/', hide_file=False, print_depth=10):
    if print_depth < 1:
        return
    print(f"🌳 目录树: {path}")

    # 用于缩进目录层级的计数器
    indent = 0

    # 递归打印目录树的辅助函数
    def print_tree(current_path, indent):
        if indent >= print_depth:
            return

        # 获取当前路径下的所有文件和文件夹
        for item in os.listdir(current_path):
            # 排除隐藏文件和文件夹（如果hide_file为False）
            if not hide_file and item.startswith('.'):
                continue

            # 构建文件或文件夹的完整路径
            item_path = os.path.join(current_path, item)

            # 打印文件或文件夹名称，使用缩进表示层级
            print('  ' * indent + '|-- ' + item)

            # 递归打印子目录的目录树
            if os.path.isdir(item_path):
                print_tree(item_path, indent + 1)

    print_tree(path, indent)
