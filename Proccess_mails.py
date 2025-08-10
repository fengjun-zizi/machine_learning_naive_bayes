import os, re

def load_tokenized_texts(root_dir):
    """
    返回：docs = [[w1, w2, ...], [w1, w2, ...], ...]
    按文件名排序；递归读所有子文件夹里的 .txt。
    """
    docs = []
    file_paths = []
    labels = []

    # 递归收集所有 .txt 文件
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                file_paths.append(os.path.join(dirpath, fn))

    # 为了结果稳定，按路径排序
    file_paths.sort()


    for path in file_paths:
        filname = os.path.basename(path).lower()
        if "spmsgc" in filname:
            labels.append("spam")
        elif "msg" in filname:
            labels.append("ham")
        else:
            labels.append("unknown")

    #print(labels)


    for p in file_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()
            # 只保留字母/数字/撇号，作为单词
            tokens = re.findall(r"[a-z0-9']+", text)
            docs.append(tokens)

    return docs, labels



# ====== 使用示例 ======
root = "/Volumes/Samsung T7/Visual Studio Code/python_language/machine_learning/sms+spam+collection/test-mails"
docs, labels = load_tokenized_texts(root)
#print(f"共读取 {len(docs)} 个文件")
#print(f"训练数据", docs)
# print("第1个文件路径：", paths[0])
# print("第二个个文件路径：", paths[147])
#print("测试数据：", labels)