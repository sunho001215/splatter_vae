"""SAM 3 worker (optional; separate ``sam3`` env: Python >= 3.12, PyTorch >= 2.7, gated HF checkpoint).

UNVERIFIED: written against facebookresearch/sam3 README APIs at the pinned commit; never executed here.

task ``segment_images``: input ``images`` (N,H,W,3) uint8, params ``prompts`` (list of text prompts),
``score_threshold``. Image API: build_sam3_image_model -> Sam3Processor.set_image -> set_text_prompt.
task ``segment_video``: input ``images`` (T,H,W,3) uint8 of one camera, params ``prompts``, video predictor
start_session / add_prompt(text) / propagate_in_video. Output per prompt and frame: instance masks, scores,
boxes, ids; an explicit ``count`` of 0 marks an absent detection (never an empty-mask placeholder).
outputs: ``masks`` (K,H,W) bool, ``scores`` (K,), ``boxes_xyxy`` (K,4) original pixels (integer-centre
convention as upstream), ``frame`` (K,), ``prompt_index`` (K,), ``instance_id`` (K,), ``count`` (N_frames, P).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _protocol import git_commit, main, sha256  # noqa: E402


def handle(request, inputs):
    import numpy as np

    repo = Path(request["repo_dir"]).resolve()
    sys.path.insert(0, str(repo))
    params = request["params"]
    prompts = list(params["prompts"])
    thr = float(params.get("score_threshold", 0.5))
    images = inputs["images"]
    masks, scores, boxes, frames, pidx, iids = [], [], [], [], [], []
    count = np.zeros((len(images), len(prompts)), np.int32)
    if request["task"] == "segment_images":
        from PIL import Image
        from sam3.model.sam3_image_processor import Sam3Processor  # upstream
        from sam3.model_builder import build_sam3_image_model  # upstream

        processor = Sam3Processor(build_sam3_image_model(**params.get("build_kwargs", {})))
        for f, image in enumerate(images):
            state = processor.set_image(Image.fromarray(image))
            for p, prompt in enumerate(prompts):
                out = processor.set_text_prompt(state=state, prompt=prompt)
                s = out["scores"].float().cpu().numpy()
                keep = s >= thr
                m = out["masks"].cpu().numpy().reshape(len(s), *image.shape[:2])[keep]
                b = out["boxes"].float().cpu().numpy()[keep]
                count[f, p] = int(keep.sum())
                for k in range(int(keep.sum())):
                    masks.append(m[k]); scores.append(s[keep][k]); boxes.append(b[k])
                    frames.append(f); pidx.append(p); iids.append(k)
    elif request["task"] == "segment_video":
        from sam3.model_builder import build_sam3_video_predictor  # upstream

        predictor = build_sam3_video_predictor(**params.get("build_kwargs", {}))
        frame_dir = Path(request.get("frame_dir") or (Path.cwd() / "frames"))
        frame_dir.mkdir(exist_ok=True)
        from PIL import Image

        for f, image in enumerate(images):
            Image.fromarray(image).save(frame_dir / f"{f:05d}.jpg", quality=95)
        H, W = images.shape[1:3]
        for p, prompt in enumerate(prompts):
            sid = predictor.handle_request(dict(type="start_session", resource_path=str(frame_dir)))["session_id"]
            predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt))
            for item in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                                                            propagation_direction="forward")):
                f, out = item["frame_index"], item["outputs"]
                probs = np.asarray(out["out_probs"], float)
                keep = probs >= thr
                count[f, p] = int(keep.sum())
                for k in np.flatnonzero(keep):
                    x, y, w, h = np.asarray(out["out_boxes_xywh"][k], float) * [W, H, W, H]
                    masks.append(np.asarray(out["out_binary_masks"][k], bool)); scores.append(probs[k])
                    boxes.append([x, y, x + w, y + h]); frames.append(f); pidx.append(p)
                    iids.append(int(out["out_obj_ids"][k]))  # SAM 3 object id: identity within this video only
            predictor.handle_request(dict(type="close_session", session_id=sid))
    else:
        raise ValueError(request["task"])
    H, W = images.shape[1:3]
    outputs = {"masks": np.array(masks, bool).reshape(-1, H, W), "scores": np.array(scores, np.float32),
               "boxes_xyxy": np.array(boxes, np.float32).reshape(-1, 4), "frame": np.array(frames, np.int32),
               "prompt_index": np.array(pidx, np.int32), "instance_id": np.array(iids, np.int32), "count": count}
    identity = {"name": "SAM3", "repo": "facebookresearch/sam3", "commit": git_commit(str(repo)),
                "checkpoint": request.get("checkpoint"), "checkpoint_sha256": sha256(request.get("checkpoint")),
                "is_mock": False}
    return identity, outputs, {"prompts": prompts, "score_threshold": thr, "task": request["task"]}


if __name__ == "__main__":
    main(handle)
