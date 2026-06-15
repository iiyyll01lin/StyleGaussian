import os

import torch
import torch_scatter
from sklearn.neighbors import NearestNeighbors
from fast_pytorch_kmeans import KMeans

# 視為「關閉」的環境變數字面值（大小寫不敏感）。
_SG_DECODER_COMPILE_FALSEY = {"", "0", "false", "no", "off"}

# 啟動 parity 自檢容忍度（compiled vs eager 在 identical input 上的 max|Δ|）。
# fp32 在 experiments/decoder_compile_ab.py 實測 ~4.6e-5；給一個遠高於該值、但仍
# 足以擋住「compile 真的改了數值」的門檻。fp16 fused 路徑融合/重排後 rounding
# 漂移較大（但仍受控），故用較寬的門檻。default-on 是 *guarded*：只有實測
# max|Δ| < 對應門檻才真的放行 compile，否則永久 fallback 回 eager。
_SG_PARITY_TOL_FP32 = 1e-3
_SG_PARITY_TOL_FP16 = 5e-2


def _resolve_decoder_compile(flag):
    '''決定 GaussianConv 是否啟用 torch.compile（DEDICATED gate，不與 reimpl 的
    REIMPL_COMPILE 混淆）。

    建構參數 ``compile`` 優先：``True``/``False`` 直接生效；``None`` 表示沿用
    環境變數 ``SG_DECODER_COMPILE``（**預設 = 開**，guarded default-on）。

    注意：default-on 不等於「無條件編譯」——實際是否走 compiled 路徑由 forward 第
    一次呼叫時的「啟動 parity 自檢」決定（見 ``_run_startup_parity_check``）：只有
    eager-vs-compiled 的 max|Δ| < 容忍度才放行；否則永久 fallback 回 eager，數值與
    未改前一致。要硬性關閉請設 ``SG_DECODER_COMPILE=0`` 或建構傳 ``compile=False``。
    '''
    if flag is not None:
        return bool(flag)
    return os.environ.get("SG_DECODER_COMPILE", "1").strip().lower() \
        not in _SG_DECODER_COMPILE_FALSEY


class GaussianConv(torch.nn.Module):
    def __init__(self, xyz, input_channel=256, layers_channel=[256, 128, 64, 32, 3], downsample_layer=[], upsample_layer=[], K=8, out_channel=None, compile=None):
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

        # torch.compile opt-in（DEDICATED gate，預設關閉、eager fallback）。
        #   env SG_DECODER_COMPILE=1 或建構參數 compile=True（kwarg 優先）啟用。
        # 關鍵：只編譯「內層 compute」(_forward_impl)，不包整個 nn.Module，
        # 因此 gaussian_model.py 的 capture()/restore()/optimizer（state_dict /
        # load_state_dict / parameters）完全不受影響。N 與每層寬度固定 →
        # 不會反覆 recompile。
        self._compile_enabled = _resolve_decoder_compile(compile)
        self._compiled_fn = None

        # 啟動 parity 自檢狀態（guarded default-on）：第一次 forward 真的要走 compiled
        # 前，先在當下的 input 上比較 eager vs compiled 的 max|Δ|，只有 < 容忍度才
        # 放行。三態：_parity_checked 是否已驗過；_parity_ok 是否通過；_parity_max_abs
        # 量到的 max|Δ|（None=尚未檢查）。
        self._parity_checked = False
        self._parity_ok = False
        self._parity_max_abs = None

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

    def _safe_to_compile(self):
        '''是否走「安全可編譯」路徑。

        條件：gpu-knn（避開 numpy 索引 graph-break）+ 無 down/up-sample（避開
        torch_scatter.scatter 的 graph-break）+ dtype 為 fp32（infer_dtype is None）
        **或** fp16（infer_dtype == torch.float16）。

        fp16 曾被排除（只准 fp32）；現放寬以原型「fp16 + compiled fused decoder」，
        看 compile（~1.47x）與 fp16（~3.7x）兩個加速能否疊加。bf16 不在白名單
        （gfx1151 已知 bug，且 set_infer_dtype 也不該被設成 bf16）。實際放不放行仍由
        啟動 parity 自檢把關（_run_startup_parity_check）。
        '''
        return (
            self._compile_enabled
            and self.use_gpu_knn
            and self.infer_dtype in (None, torch.float16)
            and not self.downsample_layer
            and not self.upsample_layer
        )

    def _parity_tol(self):
        '''啟動 parity 自檢容忍度，依 infer_dtype 取 fp32 / fp16 門檻。'''
        return _SG_PARITY_TOL_FP16 if self.infer_dtype == torch.float16 else _SG_PARITY_TOL_FP32

    @torch.no_grad()
    def _run_startup_parity_check(self, features, compiled_fn):
        '''啟動時 parity 自檢（guarded default-on 的關鍵）。

        在當下這批 ``features`` 上比較 eager ``_forward_impl`` 與 ``compiled_fn`` 的
        輸出 max|Δ|，只有 < ``_parity_tol()`` 才把 ``_parity_ok`` 設 True 放行 compile；
        否則停用 compile、永久 fallback 回 eager。比對邏輯對齊
        ``experiments/decoder_compile_ab.py``（max|Δ| on identical input）。只跑一次
        （結果快取在 ``_parity_checked`` / ``_parity_ok``）。
        '''
        self._parity_checked = True
        try:
            eager_out = self._forward_impl(features).float()
            compiled_out = compiled_fn(features).float()
            max_abs = float((compiled_out - eager_out).abs().max().item())
            self._parity_max_abs = max_abs
            tol = self._parity_tol()
            self._parity_ok = max_abs < tol
            if not self._parity_ok:
                import warnings

                warnings.warn(
                    f"GaussianConv 啟動 parity 自檢未過：max|Δ|={max_abs:.3e} >= "
                    f"tol={tol:.1e}（infer_dtype={self.infer_dtype}）→ 停用 compile、"
                    f"永久 fallback 回 eager。",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._compile_enabled = False
                self._compiled_fn = None
        except Exception as exc:  # pragma: no cover - 視 runtime 後端而定
            import warnings

            warnings.warn(
                f"GaussianConv 啟動 parity 自檢執行失敗（{exc}）→ 停用 compile、"
                f"fallback 回 eager。",
                RuntimeWarning,
                stacklevel=2,
            )
            self._parity_ok = False
            self._compile_enabled = False
            self._compiled_fn = None

    def _get_compiled_fn(self):
        '''lazy 建立 torch.compile 後的 _forward_impl；失敗印 warning 後 fallback
        回 eager（回傳 None 讓 forward 走原路徑）。'''
        if self._compiled_fn is not None:
            return self._compiled_fn
        try:
            self._compiled_fn = torch.compile(self._forward_impl, mode="max-autotune")
        except Exception as exc:  # pragma: no cover - 視 runtime 後端而定
            import warnings

            warnings.warn(
                f"SG_DECODER_COMPILE 已開啟，但 torch.compile 失敗，fallback 回 eager：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._compile_enabled = False
            self._compiled_fn = None
        return self._compiled_fn

    def forward(self, features):
        '''
        Args:
            features: [N, D]
            D: input_channel
            S: output_channel
        '''
        if self._safe_to_compile():
            fn = self._get_compiled_fn()
            if fn is not None:
                # guarded default-on：第一次走 compiled 前先做啟動 parity 自檢，
                # 不通過就永久 fallback 回 eager。
                if not self._parity_checked:
                    self._run_startup_parity_check(features, fn)
                if self._parity_ok:
                    try:
                        return fn(features)
                    except Exception as exc:  # pragma: no cover - 視 runtime 後端而定
                        import warnings

                        warnings.warn(
                            f"GaussianConv compiled forward 失敗，fallback 回 eager：{exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._compile_enabled = False
                        self._compiled_fn = None
        return self._forward_impl(features)

    def _forward_impl(self, features):
        '''decoder 的純計算迴圈（torch.compile 的目標）；內容與原 forward 一致，
        eager 路徑數值 byte-identical。'''
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
