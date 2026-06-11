import torch
import torch_scatter
from sklearn.neighbors import NearestNeighbors
from fast_pytorch_kmeans import KMeans

class GaussianConv(torch.nn.Module):
    def __init__(self, xyz, input_channel=256, layers_channel=[256, 128, 64, 32, 3], downsample_layer=[], upsample_layer=[], K=8, out_channel=None):
        super(GaussianConv, self, ).__init__()
        assert len(downsample_layer) == len(upsample_layer) == 0 or \
            (len(downsample_layer) == len(upsample_layer) and max(downsample_layer) < min(upsample_layer)) ,\
            'downsample_layer and upsample_layer must be the same length and satisfy max(downsample_layer) < min(upsample_layer) or both are empty lists'

        # out_channel (Evaluation 1, Variant B): override the final head width so the
        # decoder can emit per-Gaussian SH coefficients (3*(deg+1)^2) instead of RGB.
        # None -> keep the legacy RGB head (last layer == 3); fully backward compatible.
        # The last layer is intentionally left WITHOUT a sigmoid (see forward), which
        # is correct for unbounded SH coefficients as well as the legacy RGB output.
        if out_channel is not None:
            layers_channel = list(layers_channel[:-1]) + [int(out_channel)]

        self.K = K
        self.input_channel = input_channel
        self.N = xyz.shape[0]
        self.downsample_layer = downsample_layer
        self.upsample_layer = upsample_layer

        # 推論加速旋鈕（不影響訓練/載入）：
        #   use_gpu_knn  -> knn_indices 常駐 GPU LongTensor，消除每次 forward 的
        #                   host->device 拷貝（舊版是 numpy，PyTorch 每次 forward 都會
        #                   把它搬上裝置）。預設開。設 False 走 legacy numpy 路徑，
        #                   方便用 timing_inference.py 量 GPU-idx vs numpy 的差。
        #   infer_dtype  -> 設成 torch.float16 時，decoder forward 走 fp16（kernels/bias
        #                   預先轉半精度），給推論加速量測用。None=fp32（預設）。
        self.use_gpu_knn = True
        self.infer_dtype = None
        self._kernels_cast = None
        self._bias_cast = None

        self.init_kmeans_knn(xyz, len(downsample_layer))
        self.init_conv_params(input_channel, layers_channel)

    @torch.no_grad()
    def init_kmeans_knn(self, xyz, len_sample_layer):
        # GPU 常駐 LongTensor（推論預設）與 numpy（legacy A/B）兩份都留。
        self.knn_indices = []        # list[LongTensor]（在 xyz.device 上）
        self._knn_indices_np = []    # list[np.ndarray]（legacy 路徑用）
        self.kmeans_labels = []
        device = xyz.device

        # get original knn_indices
        xyz_numpy = xyz.cpu().numpy()
        nn = NearestNeighbors(n_neighbors=self.K, algorithm='auto')
        nn.fit(xyz_numpy)
        _, knn_indices = nn.kneighbors(xyz_numpy) # [N, K]
        self._knn_indices_np.append(knn_indices)
        self.knn_indices.append(torch.as_tensor(knn_indices, dtype=torch.long, device=device))

        last_N = self.N
        last_xyz = xyz

        for i in range(len_sample_layer):
            print('Using KMeans to cluster point clouds in level', i)
            kmeans = KMeans(n_clusters=last_N//self.K, mode='euclidean', verbose=1)
            self.kmeans_labels.append(kmeans.fit_predict(last_xyz)) # [N]
            down_centroids = torch_scatter.scatter(last_xyz, self.kmeans_labels[-1], dim=0, reduce='mean') # [cluster_num=N//5, D]

            # get knn_indices for downsampled point clouds
            nn = NearestNeighbors(n_neighbors=self.K, algorithm='auto')
            nn.fit(down_centroids.cpu().numpy())
            _, knn_indices = nn.kneighbors(down_centroids.cpu().numpy())
            self._knn_indices_np.append(knn_indices)
            self.knn_indices.append(torch.as_tensor(knn_indices, dtype=torch.long, device=device))

            last_N = down_centroids.shape[0]
            last_xyz = down_centroids

    def init_conv_params(self, input_channel, layers_channel):
        self.kernels = []
        self.bias = []
        for out_channel in layers_channel:
            self.kernels.append(torch.randn(out_channel, self.K*input_channel)*0.1)  # [out_channel, K*input_channel]
            self.bias.append(torch.zeros(1, out_channel))  # [1, out_channel]
            input_channel = out_channel

        self.kernels = torch.nn.ParameterList(self.kernels)
        self.bias = torch.nn.ParameterList(self.bias)

    @torch.no_grad()
    def set_infer_dtype(self, dtype):
        '''切換 decoder 推論精度（如 torch.float16）。dtype=None 還原 fp32。

        會預先把 kernels/bias 轉好一份指定精度的 detached copy，避免在 forward
        內每次重轉，量測 fp16 加速時才公平。不更動原本的 fp32 Parameter。
        '''
        self.infer_dtype = dtype
        if dtype is None:
            self._kernels_cast = None
            self._bias_cast = None
        else:
            self._kernels_cast = [k.detach().to(dtype) for k in self.kernels]
            self._bias_cast = [b.detach().to(dtype) for b in self.bias]

    def _knn_index(self, sample_level):
        '''取第 sample_level 層的 KNN 索引；GPU LongTensor 或 legacy numpy。'''
        if self.use_gpu_knn:
            return self.knn_indices[sample_level]
        return self._knn_indices_np[sample_level]

    def forward(self, features):
        '''
        Args:
            features: [N, D]
            D: input_channel
            S: output_channel
        '''
        dt = self.infer_dtype
        if dt is not None:
            features = features.to(dt)

        sample_level = 0
        for i in range(len(self.kernels)):
            if i in self.downsample_layer:
                sample_level += 1
                features = torch_scatter.scatter(features, self.kmeans_labels[sample_level-1], dim=0, reduce='mean')
            elif i in self.upsample_layer:
                sample_level -= 1
                features = features[self.kmeans_labels[sample_level]]

            knn_indices = self._knn_index(sample_level)

            knn_features = features[knn_indices] # [N, K, D]
            knn_features = knn_features.reshape(knn_features.size(0), -1) # [N, K*D]
            if dt is not None:
                kernel_i = self._kernels_cast[i]
                bias_i = self._bias_cast[i]
            else:
                kernel_i = self.kernels[i]
                bias_i = self.bias[i]
            features = knn_features @ kernel_i.T + bias_i # [N, S]
            features = torch.sigmoid(features) if i != len(self.kernels)-1 else features

        return features # [N, S]
