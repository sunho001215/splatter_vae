# Source this file before launching anything that may touch a GPU.
#   source scripts/gpu_env.sh
#   CUDA_VISIBLE_DEVICES=$MIG4_0 python scripts/train.py ...
# Local devices are MIG 1g.24gb instances (one per process): MIG<gpu>_<instance>.
export MIG4_0=MIG-8da55219-bd6b-5260-8be1-3b2a76ac733c MIG4_1=MIG-c4ec2a95-30cc-5398-82bf-d9b11787fc70
export MIG4_2=MIG-423203d9-5801-5892-a489-30f16a5a2d4a MIG4_3=MIG-02b55a66-ed41-53c2-9004-c3f5619d6369
export MIG5_0=MIG-05a151e7-c6bd-5344-91cb-8bfb3a342631 MIG5_1=MIG-b50e85c3-8498-5ef0-906e-5eda9db34b90
export MIG5_2=MIG-2ebd9e11-f2e0-5e22-870a-7779d44ec10a MIG5_3=MIG-c3511b41-5f78-52f1-9b93-9c6794cdbbdd
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
