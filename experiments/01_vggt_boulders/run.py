"""Feed-forward VGGT reconstruction of an ETH3D scene.

Saves predicted cameras/depth, a confidence-filtered colored point cloud, and (if the
scene ships COLMAP calibration) camera-pose errors against it.

    .venv/bin/python experiments/01_vggt_boulders/run.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

ROOT = Path(__file__).resolve().parents[2]


def load_colmap_poses(images_txt):
    """Returns {image_name: (R_cam_from_world, t)} from a COLMAP images.txt."""
    lines = [l for l in Path(images_txt).read_text().splitlines() if l and not l.startswith("#")]
    poses = {}
    for line in lines[::2]:  # every second line is the 2D-point list
        _, qw, qx, qy, qz, tx, ty, tz, _, name = line.split()
        R = Rotation.from_quat([float(qx), float(qy), float(qz), float(qw)]).as_matrix()
        poses[Path(name).name] = (R, np.array([tx, ty, tz], dtype=float))
    return poses


def umeyama(src, dst):
    """Sim(3) s,R,t minimizing ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def pose_errors(extrinsic, names, gt_poses):
    """Camera-center ATE (after Sim(3) alignment) and rotation error, in the aligned frame."""
    R_p = extrinsic[:, :, :3]
    c_p = -np.einsum("nji,nj->ni", R_p, extrinsic[:, :, 3])  # -R^T t
    R_g = np.stack([gt_poses[n][0] for n in names])
    c_g = np.stack([-gt_poses[n][0].T @ gt_poses[n][1] for n in names])
    s, R, t = umeyama(c_p, c_g)
    ate = np.linalg.norm(s * c_p @ R.T + t - c_g, axis=1)
    # predicted camera-to-world rotation expressed in the GT world frame
    R_est = np.einsum("ij,njk->nik", R, R_p.transpose(0, 2, 1))
    R_ref = R_g.transpose(0, 2, 1)
    rot = np.degrees(
        np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", R_est, R_ref)).as_rotvec(), axis=1)
    )
    scene_extent = np.linalg.norm(c_g - c_g.mean(0), axis=1).max()
    return {
        "ate_rmse": float(np.sqrt((ate**2).mean())),
        "ate_rmse_rel_to_extent": float(np.sqrt((ate**2).mean()) / scene_extent),
        "rot_err_deg_mean": float(rot.mean()),
        "rot_err_deg_max": float(rot.max()),
        "sim3_scale": float(s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="boulders")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-images", type=int, default=None, help="evenly subsample to this many views")
    ap.add_argument("--conf-percentile", type=float, default=50, help="drop points below this confidence percentile")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scene_dir = ROOT / "data/eth3d" / args.split / args.scene
    out = args.out or ROOT / "outputs" / "01_vggt" / args.scene
    out.mkdir(parents=True, exist_ok=True)

    paths = sorted((scene_dir / "images/dslr_images_undistorted").glob("*.JPG"))
    if args.max_images and len(paths) > args.max_images:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, args.max_images).round().astype(int)]
    names = [p.name for p in paths]
    print(f"{args.scene}: {len(paths)} images")

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images([str(p) for p in paths]).to(device)  # (S,3,H,W) in [0,1]
    print("model input:", tuple(images.shape))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        tokens, ps_idx = model.aggregator(images[None])
        pose_enc = model.camera_head(tokens)[-1]
        depth, depth_conf = model.depth_head(tokens, images[None], ps_idx)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    torch.cuda.synchronize()
    runtime = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"inference {runtime:.1f}s, peak GPU memory {peak_gb:.1f} GB")

    extrinsic, intrinsic = extrinsic[0].float().cpu().numpy(), intrinsic[0].float().cpu().numpy()
    depth, depth_conf = depth[0].float().cpu().numpy(), depth_conf[0].float().cpu().numpy()
    rgb = images.permute(0, 2, 3, 1).cpu().numpy()

    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)  # (S,H,W,3)
    keep = depth_conf >= np.percentile(depth_conf, args.conf_percentile)
    cloud = trimesh.PointCloud(points[keep], colors=(rgb[keep] * 255).astype(np.uint8))
    cloud.export(out / "points.ply")

    np.savez_compressed(
        out / "predictions.npz",
        names=names, extrinsic=extrinsic, intrinsic=intrinsic, depth=depth[..., 0], depth_conf=depth_conf,
    )

    metrics = {"scene": args.scene, "num_images": len(paths), "input_hw": list(images.shape[-2:]),
               "dtype": str(dtype), "runtime_s": runtime, "peak_gpu_gb": peak_gb, "num_points": int(keep.sum())}
    calib = scene_dir / "dslr_calibration_undistorted/images.txt"
    if calib.exists():
        metrics["pose_vs_colmap"] = pose_errors(extrinsic, names, load_colmap_poses(calib))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
