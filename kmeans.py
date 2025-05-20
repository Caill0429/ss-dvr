import torch


class MiniBatchKMeansCUDA:
    def __init__(self, n_clusters=10, batch_size=1000, max_iter=100, tol=1e-4, device='cuda'):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

    def fit(self, X):

        # 随机初始化质心
        centroids = X[torch.randint(0, X.size(0), (self.n_clusters,))]  # 随机选取 n_clusters 个数据点作为初始质心
        centroids = centroids.to(self.device)

        prev_centroids = torch.zeros_like(centroids).to(self.device)

        for i in range(self.max_iter):
            # 每次从数据中随机选择一个小批量
            indices = torch.randint(0, X.size(0), (self.batch_size,))
            batch = X[indices]

            # 计算小批量样本到所有质心的距离
            distances = torch.cdist(batch, centroids)  # [batch_size, n_clusters]

            # 为每个样本分配最近的质心
            labels = torch.argmin(distances, dim=1)

            # 计算每个质心的新位置
            new_centroids = torch.zeros_like(centroids).to(self.device)
            for j in range(self.n_clusters):
                # 取出属于当前质心的所有点
                cluster_points = batch[labels == j]
                if cluster_points.size(0) > 0:
                    new_centroids[j] = cluster_points.mean(dim=0)

            # 计算质心的变化量
            centroid_shift = torch.norm(new_centroids - centroids)

            # 更新质心
            centroids = new_centroids

            # 如果质心的变化小于给定的容忍度，提前停止
            if centroid_shift < self.tol:
                print(f"Converged at iteration {i + 1}")
                break

        self.centroids = centroids

        batch_count = 100
        batch_number = int(8000000 / batch_count)

        labels = []

        for i in range(batch_count):
            labels.append(torch.argmin(torch.cdist(X[i*batch_number:i*batch_number+batch_number], centroids), dim=1))
        # self.labels_ = torch.argmin(torch.cdist(X, centroids), dim=1)  # 计算每个数据点的最终标签

        labels = torch.concatenate(labels)
        self.labels_ = labels

        return self

    def predict(self, X):
        X = X.to(self.device)
        return torch.argmin(torch.cdist(X, self.centroids), dim=1)


# 示例使用：
# if __name__ == "__main__":
#     with torch.no_grad():
#         # 生成一个随机数据集 (8000000, 512)
#         n_samples = 8000000
#         n_features = 512
#         X = torch.randn(n_samples, n_features, device='cuda')  # 使用 torch.tensor 生成数据
#
#         # 初始化和训练 Mini-batch K-means
#         model = MiniBatchKMeansCUDA(n_clusters=4, batch_size=1000, max_iter=100, device='cuda')
#
#         start_time = time.time()
#         model.fit(X)
#         end_time = time.time()
#
#         print(f"Training time: {end_time - start_time} seconds")
#
#         # 预测新的数据点
#         new_data = torch.randn(10, n_features).cuda()  # 假设我们有 10 个新的数据点
#         labels = model.predict(new_data)
#         print(f"Predicted labels: {labels.cpu().numpy()}")
