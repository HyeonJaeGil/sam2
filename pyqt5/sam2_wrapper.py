# sam2_wrapper.py

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor


class SAM2Interface:
    """
    Wrapper around SAM2VideoPredictor to manage raw frames, user prompts,
    and progressive propagation.
    """
    def __init__(self):
        self.sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(
            self.model_cfg, self.sam2_checkpoint, "cuda"
        )
        self.raw_frames = []  # List of original RGB frames
        self.frame_cache = {}  # user-prompt overlays per frame
        self.propagated_masks = {}  # propagated overlays per frame
        self.obj_masks_per_frame = {}  # raw binary masks per frame for lookup
        self.frames_with_prompt = set()  # frames where user added prompts
        self.colormap = self._create_colormap(num_colors=20)
        self.last_propagated_frame = -1  # index of last propagated frame

    def init(self, video_path):
        """
        Load raw frames and initialize inference state.
        """
        self.state = self.predictor.init_state(video_path, offload_video_to_cpu=True, offload_state_to_cpu=True)
        if os.path.isdir(video_path):
            self.raw_frames = self._load_image_frames(video_path)
        elif os.path.isfile(video_path):
            self.raw_frames = self._load_video_frames(video_path)
        else:
            raise ValueError(f"Invalid video path: {video_path}")
        self.reset_state()
        self.h, self.w = self.raw_frames[0].shape[:2]

    def _create_colormap(self, num_colors=20):
        cmap = plt.get_cmap("tab20")
        colors = (np.array([cmap(np.arange(num_colors))[:, :3]]).squeeze() * 255).astype(np.uint8)
        return colors

    def _load_video_frames(self, path):
        frames = []
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _load_image_frames(self, path):
        frames = []
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = Image.open(os.path.join(path, fname)).convert('RGB')
                frames.append(np.array(img))
        return frames

    def get_raw_frame(self, idx):
        if not (0 <= idx < len(self.raw_frames)):
            raise IndexError("Frame index out of range.")
        return self.raw_frames[idx]

    def get_cached_frame(self, idx):
        if not (0 <= idx < len(self.raw_frames)):
            raise IndexError("Frame index out of range.")
        return self.frame_cache.get(idx)

    def get_masked_frame(self, idx):
        if not (0 <= idx < len(self.raw_frames)):
            raise IndexError("Frame index out of range.")
        return self.propagated_masks.get(idx)

    def get_merged_mask(self, obj_ids, mask_logits, frame_idx):
        """
        Merge per-object mask logits into a single RGB overlay, and
        store raw binary masks for point lookup.
        """
        merged = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        masks = []
        for oid, logit in zip(obj_ids, mask_logits):
            mask_np = logit.cpu().numpy().reshape(self.h, self.w)
            color = self.colormap[oid % len(self.colormap)]
            merged[mask_np > 0] = color
            masks.append((oid, mask_np))
        self.obj_masks_per_frame.setdefault(frame_idx, []).extend(masks)
        return merged

    def propagate_until(self, target_frame_idx):
        """
        Propagate tracking for all frames up to and including target_frame_idx.
        """
        if not self.frames_with_prompt:
            return
        if target_frame_idx <= self.last_propagated_frame:
            return
        start_idx = self.last_propagated_frame + 1
        num_frames = target_frame_idx - self.last_propagated_frame - 1
        if num_frames < 0: # no frames to propagate
            return

        for out_idx, obj_ids, logits in self.predictor.propagate_in_video(
            self.state,
            start_frame_idx=start_idx,
            max_frame_num_to_track=num_frames
        ):
            merged = self.get_merged_mask(obj_ids, logits, out_idx)
            overlay = cv2.addWeighted(self.raw_frames[out_idx], 1, merged, 0.6, 0)
            self.frame_cache[out_idx] = overlay
            self.propagated_masks[out_idx] = overlay

        self.last_propagated_frame = target_frame_idx

    def propagate_next(self, current_frame_idx):
        """
        Propagate and navigate to the next frame after current_frame_idx.
        Returns the next frame index if successful, else None.
        """
        # ensure prior frames are tracked
        self.propagate_until(current_frame_idx)
        next_idx = current_frame_idx + 1
        if next_idx < len(self.raw_frames) and next_idx > self.last_propagated_frame:
            self.propagate_until(next_idx)
            return next_idx
        return None

    def add_new_points_or_box(self, obj_id, frame_idx, points, labels):
        """
        Add a prompt and generate an updated mask overlay for frame_idx.
        """
        self.propagate_until(frame_idx - 1)
        _, obj_ids, logits = self.predictor.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        self.frames_with_prompt.add(frame_idx)
        self.last_propagated_frame = frame_idx - 1 # wrong code
        merged = self.get_merged_mask(obj_ids, logits, frame_idx)
        overlay = cv2.addWeighted(self.raw_frames[frame_idx], 1, merged, 0.6, 0)
        self.frame_cache[frame_idx] = overlay
        return merged

    def get_object_id_at_point(self, point, frame_idx):
        x, y = point
        for oid, mask in self.obj_masks_per_frame.get(frame_idx, []):
            if mask[y, x] > 0:
                return oid
        return None

    def reset_state(self):
        """
        Clear predictor state and all cached masks.
        """
        self.predictor.reset_state(self.state)
        self.frame_cache.clear()
        self.propagated_masks.clear()
        self.obj_masks_per_frame.clear()
        self.frames_with_prompt.clear()
        self.last_propagated_frame = -1
        for idx, raw in enumerate(self.raw_frames):
            self.propagated_masks[idx] = raw

    def save_cached_frames(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for idx, frame in self.frame_cache.items():
            Image.fromarray(frame).save(os.path.join(save_dir, f"frame_{idx:04d}.jpg"))

    def __len__(self):
        return len(self.raw_frames)
