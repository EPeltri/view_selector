# data

Datasets live in one subfolder each. This folder is git-ignored (see ../.gitignore).

## eth3d/
ETH3D high-res multi-view benchmark, DSLR, undistorted images (COLMAP calibration).
Source: https://www.eth3d.net/datasets (multi_view_{training,test}_dslr_undistorted.7z)

- `archives/` original .7z downloads
- `training/<scene>/` 13 scenes
- `test/<scene>/` 12 scenes (no ground truth)
