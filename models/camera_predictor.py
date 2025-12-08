import torch
import torch.nn as nn

from utils.general_utils import rot6d_to_matrix

class CameraParamPredictor(nn.Module):
    """
    Predict camera parameters from the *view-dependent* embeddings z_dep.

    Inputs:
        z_dep_i: (B, T, D) dependent embedding for image_i_t
        z_dep_j: (B, T, D) dependent embedding for image_j_t

    Outputs:
        T_ij: (B, 4, 4) transform from cam_i -> cam_j (world == cam_i)
        T_ji: (B, 4, 4) transform from cam_j -> cam_i (world == cam_j)
        K_i:  (B, 3, 3) intrinsics for camera i (image_i_t)
        K_j:  (B, 3, 3) intrinsics for camera j (image_j_t)

    Design:
        - We global-average-pool each z_dep over tokens -> one vector per image.
        - A pairwise MLP over [feat_i, feat_j] predicts:
            * 6D rotation (two columns of R)
            * 3D translation t
          which we convert to a 4x4 SE(3) matrix.
        - A per-view MLP over feat_i (or feat_j) predicts 4 intrinsics scalars
          (fx, fy, cx, cy), then we map them to a well-behaved 3x3 K.
    """

    def __init__(
        self,
        dep_latent_dim: int,
        img_height: int,
        img_width: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.dep_latent_dim = dep_latent_dim
        self.img_height = img_height
        self.img_width = img_width

        pair_dim = dep_latent_dim * 2  # concat [feat_i, feat_j]

        # Pose head: predicts 6D rotation + 3D translation
        #   output size = 9: [r1x, r1y, r1z, r2x, r2y, r2z, tx, ty, tz]
        self.pose_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 9),
        )

        # Intrinsics head: per-view intrinsics from z_dep_i or z_dep_j.
        #   output size = 1: [delta_f]
        self.intrinsics_mlp = nn.Sequential(
            nn.Linear(dep_latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
    
        # Initialize pose_mlp to predict identity transform at start
        self._init_pose_to_identity()
        self._init_intrinsics_to_nominal()

    def _init_intrinsics_to_nominal(self):
        """
        Initialize intrinsics_mlp so that its final outputs are all zeros
        at the beginning of training, for ANY input feature.

        This guarantees that the predicted intrinsics K are at their
        nominal values (fx = fy = 0.5*(H+W).
        """
        last_layer: nn.Linear = self.intrinsics_mlp[-1]

        # Zero weights: outputs do not depend on input initially
        nn.init.zeros_(last_layer.weight)
        # Zero bias: raw outputs are exactly zero => all deltas are zero
        nn.init.zeros_(last_layer.bias)

    def _init_pose_to_identity(self):
        """
        Initialize pose_mlp so that, at the beginning of training,
        it predicts:

            rot_6d = [1, 0, 0, 0, 1, 0]  -> R = I
            t      = [0, 0, 0]          -> translation = 0

        for ANY input. This guarantees T_ij and T_ji ~ Identity at step 0.
        """
        last_layer: nn.Linear = self.pose_mlp[-1]

        # Zero weights so output does not depend on input initially
        nn.init.zeros_(last_layer.weight)

        # Bias encodes [rot_6d, t]
        with torch.no_grad():
            bias = torch.zeros(9)
            # 6D rotation that maps to identity when passed through rot6d_to_matrix
            bias[0] = 1.0  # first column: (1,0,0)
            bias[4] = 1.0  # second column: (0,1,0)
            # bias[2], bias[3], bias[5] remain 0, giving rot6d = [1,0,0,0,1,0]
            # translation part (last 3) stays zero
            last_layer.bias.copy_(bias)

    # --------------------------- helper: intrinsics ---------------------------

    def _predict_intrinsics(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, D) pooled dependent embedding for one view.

        Returns:
            K: (B, 3, 3) pinhole intrinsics matrix with:
                 fx, fy > 0, cx, cy near image center.
        """
        B = feat.shape[0]
        delta_f = self.intrinsics_mlp(feat).squeeze(-1)  # (B,)

        # Nominal values around which we predict *deltas* (in pixels)
        base_f  = 0.5 * (self.img_height + self.img_width)  # ~focal length in px
        base_cx = self.img_width * 0.5
        base_cy = self.img_height * 0.5

        # ------------------------------------------------------------------
        # Symmetric focal offsets around base_f (in pixel domain)
        # ------------------------------------------------------------------
        delta_f = 0.5 * base_f * torch.tanh(delta_f)
        f = (base_f + delta_f).clamp(min=1.0)

        # ------------------------------------------------------------------
        # Principal point near the image center (same idea as before)
        # ------------------------------------------------------------------
        K = torch.zeros(B, 3, 3, device=feat.device, dtype=feat.dtype)
        K[:, 0, 0] = f
        K[:, 1, 1] = f
        K[:, 0, 2] = base_cx
        K[:, 1, 2] = base_cy
        K[:, 2, 2] = 1.0
        return K

    # --------------------------- helper: relative pose ------------------------

    def _predict_relative_T(
        self,
        feat_src: torch.Tensor,
        feat_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict transform from src camera coords to tgt camera coords:

            X_tgt ~ T_src_to_tgt * X_src

        feat_src, feat_tgt: (B, D)
        Returns:
            T: (B, 4, 4)
        """
        B = feat_src.shape[0]
        pair = torch.cat([feat_src, feat_tgt], dim=-1)  # (B, 2D)
        out = self.pose_mlp(pair)                      # (B, 9)

        rot_6d = out[:, :6]                            # (B, 6)
        t = out[:, 6:9]                                # (B, 3)

        R = rot6d_to_matrix(rot_6d)                    # (B, 3, 3)

        # Build 4x4 homogeneous matrix
        T = torch.eye(4, device=feat_src.device, dtype=feat_src.dtype).unsqueeze(0).repeat(B, 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        return T

    # --------------------------- main forward ---------------------------------

    def forward(
        self,
        z_dep_i: torch.Tensor,
        z_dep_j: torch.Tensor,
    ):
        """
        z_dep_i, z_dep_j: (B, T, D_dep)

        Returns:
            T_ij: (B, 4, 4)  cam_i -> cam_j  (world == cam_i)
            T_ji: (B, 4, 4)  cam_j -> cam_i  (world == cam_j)
            K_i:  (B, 3, 3) intrinsics for view i
            K_j:  (B, 3, 3) intrinsics for view j
        """
        # Relative poses (use SAME network with swapped inputs for ji)
        T_ij = self._predict_relative_T(z_dep_i, z_dep_j)  # cam_i -> cam_j
        T_ji = self._predict_relative_T(z_dep_j, z_dep_i)  # cam_j -> cam_i

        # Intrinsics per view
        K_i = self._predict_intrinsics(z_dep_i)
        K_j = self._predict_intrinsics(z_dep_j)

        return T_ij, T_ji, K_i, K_j