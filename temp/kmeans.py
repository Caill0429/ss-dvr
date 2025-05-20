import torch


class MiniBatchKMeansCUDA:
    def __init__(self, n_clusters=10, batch_size=1000, max_iter=100, tol=1e-4, device='cuda', seed=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.seed = seed

    def fit(self, X):

        centroids = X[torch.randint(0, X.size(0), (self.n_clusters,), generator=self.seed)]
        centroids = centroids.to(self.device)

        prev_centroids = torch.zeros_like(centroids).to(self.device)

        for i in range(self.max_iter):
            indices = torch.randint(0, X.size(0), (self.batch_size,), generator=self.seed)
            batch = X[indices]

            distances = torch.cdist(batch, centroids)

            labels = torch.argmin(distances, dim=1)

            new_centroids = torch.zeros_like(centroids).to(self.device)
            for j in range(self.n_clusters):
                cluster_points = batch[labels == j]
                if cluster_points.size(0) > 0:
                    new_centroids[j] = cluster_points.mean(dim=0)

            centroid_shift = torch.norm(new_centroids - centroids)

            centroids = new_centroids

            if centroid_shift < self.tol:
                print(f"Converged at iteration {i + 1}")
                break

        self.centroids = centroids

        batch_count = 100
        batch_number = int(8000000 / batch_count)

        labels = []

        for i in range(batch_count):
            labels.append(torch.argmin(torch.cdist(X[i*batch_number:i*batch_number+batch_number], centroids), dim=1))

        labels = torch.concatenate(labels)
        self.labels_ = labels

        return self

    def predict(self, X):
        X = X.to(self.device)
        return torch.argmin(torch.cdist(X, self.centroids), dim=1)

