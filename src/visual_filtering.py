"""Flag videos without sign language interpretation (paper, Sec. 3, step 1).

The mean/std of the RGB values of a fixed 16x16 patch inside the interpreter
overlay is estimated over validated reference videos. Every frame of each
candidate video is compared against that distribution: frames deviating by
more than 3 sigma are outliers, and videos with more than 10% outlier frames
are flagged as uninterpreted. Flagged videos are only reported, not deleted,
so they can be manually verified before removal.
"""
import json
import logging
import subprocess
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

import numpy as np

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

# Grey background in the top-right corner of the interpreter overlay of the
# 1280x720 Parlament broadcasts, out of reach of the signer's arms.
PATCH_X, PATCH_Y, PATCH_SIZE = 352, 88, 16


@lru_cache(maxsize=None)  # a reference video can also be a candidate: decode it once
def patch_means(video: Path, x: int, y: int, size: int, chunk: int = 4096) -> np.ndarray:
    """Per-frame mean RGB of the patch, shape (n_frames, 3)."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video), "-an",
        "-vf", f"crop={size}:{size}:{x}:{y}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frame_bytes = size * size * 3
    means = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while buf := proc.stdout.read(frame_bytes * chunk):
            n = len(buf) // frame_bytes
            frames = np.frombuffer(buf[: n * frame_bytes], np.uint8).reshape(n, -1, 3)
            means.append(frames.mean(axis=1))
    if proc.returncode:
        raise IOError(f"ffmpeg failed on {video}")
    return np.concatenate(means) if means else np.empty((0, 3))


def reference_stats(means: np.ndarray, min_std: float) -> dict:
    # min_std: calibration floor, a static grey patch has near-zero std and
    # compression noise alone would otherwise produce outliers
    return {"mean": means.mean(axis=0).tolist(),
            "std": np.maximum(means.std(axis=0), min_std).tolist()}


def outlier_ratio(means: np.ndarray, stats: dict, n_sigma: float) -> float:
    if len(means) == 0:
        return 1.0
    dev = np.abs(means - np.array(stats["mean"])) > n_sigma * np.array(stats["std"])
    return float(dev.any(axis=1).mean())


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", "-v", required=True, help="Directory with candidate .mp4 videos")
    ref = parser.add_mutually_exclusive_group(required=True)
    ref.add_argument("--reference", "-r", nargs="+", help="Validated interpreted videos")
    ref.add_argument("--reference-stats", help="JSON with precomputed reference stats")
    parser.add_argument("--out", "-o", required=True, help="Output JSON report")
    parser.add_argument("--n-sigma", type=float, default=3.0)
    parser.add_argument("--max-outliers", type=float, default=0.10)
    parser.add_argument("--min-std", type=float, default=1.0)
    parser.add_argument("--x", type=int, default=PATCH_X)
    parser.add_argument("--y", type=int, default=PATCH_Y)
    parser.add_argument("--size", type=int, default=PATCH_SIZE)
    args = parser.parse_args()

    if args.reference_stats:
        stats = json.load(open(args.reference_stats))["reference_stats"]
    else:
        ref_means = np.concatenate([patch_means(Path(v).resolve(), args.x, args.y, args.size) for v in args.reference])
        stats = reference_stats(ref_means, args.min_std)
    logging.info(f"Reference stats: {stats}")

    videos = {}
    for video in sorted(Path(args.videos).glob("*.mp4")):
        ratio = outlier_ratio(patch_means(video.resolve(), args.x, args.y, args.size), stats, args.n_sigma)
        flagged = ratio > args.max_outliers
        videos[video.name] = {"outlier_ratio": ratio, "uninterpreted": flagged}
        (logging.warning if flagged else logging.info)(
            f"{video.name}: {ratio:.2%} outlier frames{' -> UNINTERPRETED, verify manually' if flagged else ''}")

    report = {"reference_stats": stats, "params": vars(args), "videos": videos}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"{sum(v['uninterpreted'] for v in videos.values())}/{len(videos)} videos flagged, report in {args.out}")


if __name__ == "__main__":
    main()
