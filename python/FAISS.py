import faiss                   # 导入 faiss 库
import numpy as np             # 用于数值计算

# 创建或加载 FAISS 索引
def create_faiss_index(dimension):
    # 使用 L2 距离度量的 IndexFlatL2
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离的索引
    return index

# 示例：加载数据并添加到 FAISS 索引
def add_to_faiss(index, data):
    # 将数据添加到索引中
    data = np.array(data).astype(np.float32)  # 确保数据是float32类型
    index.add(data)  # 添加数据
    return index

# 获取相关文档的 FAISS 检索
def faiss_search(index, query_vector, k=5):
    # 从FAISS索引中查找最相似的k个文档
    distances, indices = index.search(np.array([query_vector]).astype(np.float32), k)
    return indices, distances

# FAISS 索引的创建示例
if __name__ == "__main__":
    # 假设文档向量的维度是128
    index = create_faiss_index(128)

    # 示例文档向量
    docs = np.random.random((10, 128))  # 10个文档，每个128维
    index = add_to_faiss(index, docs)

    # 查询向量
    query = np.random.random(128)
    indices, distances = faiss_search(index, query)
    print(f"Top 5 closest document indices: {indices}")
    print(f"Distances: {distances}")
