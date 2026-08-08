# WAFT checkpoints

The optical-flow preprocessor uses the official WAFT `waftv2` implementation
under `dataset/third_party/WAFT` with the authors' recommended a1 downstream
checkpoint.

Expected local files (ignored by Git):

```text
dataset/assets/waft/waft-a1-downstream.pth
dataset/assets/waft/depth-anything-ckpts/depth_anything_v2_vits.pth
```

Sources:

- WAFT a1 downstream checkpoint: <https://drive.google.com/file/d/1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC/view>
- Depth Anything V2 Small: <https://huggingface.co/depth-anything/Depth-Anything-V2-Small>

WAFT source is pinned as a Git submodule from
<https://github.com/princeton-vl/WAFT>.
