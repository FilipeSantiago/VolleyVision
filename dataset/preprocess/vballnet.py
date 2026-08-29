from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class VBallNet:

    def __init__(self):
        self.video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        self.manifest_df = pd.DataFrame()

    def build_vballnet_manifest(self, root: str | Path):
        print("build_vballnet_manifest")
        root = Path(root)

        rows = []

        stats = {
            "matches": 0,
            "csv_files": 0,
            "valid_videos": 0,
            "missing_videos": 0,
            "invalid_videos": 0,
        }

        # Find directories that actually contain annotation folders
        match_dirs = set()

        for csv_dir in root.rglob("csv"):
            if csv_dir.is_dir():
                match_dirs.add(csv_dir.parent)

        for corrected_dir in root.rglob("corrected_csv"):
            if corrected_dir.is_dir():
                match_dirs.add(corrected_dir.parent)

        match_dirs = sorted(match_dirs)

        print(f"Found {len(match_dirs)} possible match directories")

        for match_dir in match_dirs:
            stats["matches"] += 1

            corrected_dir = match_dir / "corrected_csv"
            normal_dir = match_dir / "csv"

            if corrected_dir.exists():
                annotation_dir = corrected_dir
            elif normal_dir.exists():
                annotation_dir = normal_dir
            else:
                continue

            csv_files = [
                path
                for path in annotation_dir.glob("*.csv")
                if self.is_ground_truth_csv(path)
            ]

            for csv_path in csv_files:
                stats["csv_files"] += 1

                video_path = self.find_video(
                    match_dir,
                    csv_path,
                )

                if video_path is None:
                    stats["missing_videos"] += 1

                    print(
                        f"[MISSING] {csv_path}"
                    )
                    continue

                metadata = self.get_video_metadata(
                    video_path
                )

                if metadata is None:
                    stats["invalid_videos"] += 1

                    print(
                        f"[INVALID VIDEO] {video_path}"
                    )
                    continue

                stats["valid_videos"] += 1

                annotations = pd.read_csv(csv_path)

                required = {
                    "Frame",
                    "Visibility",
                    "X",
                    "Y",
                }

                if not required.issubset(
                        annotations.columns
                ):
                    print(
                        f"[INVALID CSV] {csv_path}"
                    )
                    print(
                        f"    columns={list(annotations.columns)}"
                    )
                    continue

                # Infer train/test from path
                parts = {
                    part.lower()
                    for part in match_dir.parts
                }

                if "train" in parts:
                    split = "train"
                elif "test" in parts:
                    split = "test"
                elif "val" in parts:
                    split = "val"
                else:
                    split = "unknown"

                for annotation in annotations.itertuples(
                        index=False
                ):
                    rows.append(
                        {
                            "split": split,
                            "match_id": match_dir.name,
                            "video_id": self.annotation_stem(
                                csv_path
                            ),
                            "video_path": str(
                                video_path.resolve()
                            ),
                            "frame": int(
                                annotation.Frame
                            ),
                            "visibility": int(
                                annotation.Visibility
                            ),
                            "x": float(
                                annotation.X
                            ),
                            "y": float(
                                annotation.Y
                            ),
                            **metadata,
                        }
                    )

        manifest = pd.DataFrame(rows)

        if not manifest.empty:
            manifest["x_norm"] = (
                    manifest["x"]
                    / manifest["width"]
            )

            manifest["y_norm"] = (
                    manifest["y"]
                    / manifest["height"]
            )

        print("\nDataset summary")
        print("----------------")

        for key, value in stats.items():
            print(f"{key}: {value}")

        print(f"annotations: {len(manifest)}")
        self.manifest_df = manifest
        return self

    def split_dataset(self):
        print("split_dataset")
        rng = np.random.default_rng(42)

        videos = self.manifest_df["video_id"].unique()
        rng.shuffle(videos)

        n = len(videos)

        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_videos = videos[:n_train]
        val_videos = videos[n_train:n_train + n_val]
        test_videos = videos[n_train + n_val:]

        self.manifest_df["split"] = "test"

        self.manifest_df.loc[
            self.manifest_df["video_id"].isin(train_videos),
            "split"
        ] = "train"

        self.manifest_df.loc[
            self.manifest_df["video_id"].isin(val_videos),
            "split"
        ] = "val"

        return self

    def preprocess_frames(self, output_dir: str | Path, width: int = 512, height: int = 288) -> pd.DataFrame:
        print("preprocess_frames")
        output_dir = Path(output_dir)
        rows = []

        for video_path, video_df in tqdm(
                self.manifest_df.groupby("video_path"),
                desc="Processing videos",
        ):
            video_df = video_df.sort_values("frame")
            video_id = video_df.iloc[0]["video_id"]
            split = video_df.iloc[0]["split"]

            frame_dir = output_dir / video_id
            frame_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(video_path)

            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            scale_x = width / original_width
            scale_y = height / original_height

            annotations = {
                int(row.frame): row
                for row in video_df.itertuples(index=False)
            }

            frame_number = 0

            while True:
                success, frame = cap.read()

                if not success:
                    break

                if frame_number not in annotations:
                    frame_number += 1
                    continue

                annotation = annotations[frame_number]

                frame = cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )

                frame_path = frame_dir / f"{frame_number:06d}.jpg"

                cv2.imwrite(
                    str(frame_path),
                    frame,
                )

                visibility = int(annotation.visibility)

                if visibility:
                    x = float(annotation.x) * scale_x
                    y = float(annotation.y) * scale_y
                else:
                    x = None
                    y = None

                rows.append({
                    "video_id": video_id,
                    "frame": frame_number,
                    "frame_path": str(frame_path),
                    "visibility": visibility,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "split": split
                })

                frame_number += 1

            cap.release()

        self.manifest_df = pd.DataFrame(rows)
        return self

    def find_video(self, match_dir: Path, csv_path: Path) -> Path | None:
        wanted_stem = self.annotation_stem(csv_path)

        # Search the entire match directory rather than assuming /video
        candidates = [
            path
            for path in match_dir.rglob("*")
            if (
                    path.is_file()
                    and path.suffix.lower() in self.video_extensions
                    and path.stem == wanted_stem
            )
        ]

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            print(
                f"[WARNING] Multiple video matches for {csv_path.name}:"
            )
            for candidate in candidates:
                print(f"    {candidate}")

            return candidates[0]

        return None

    def get_manifest(self):
        return self.manifest_df

    @staticmethod
    def is_ground_truth_csv(path: Path) -> bool:
        name = path.name.lower()

        return (
                name.endswith("_ball.csv")
                and "predict" not in name
                and "prediction" not in name
                and "inference" not in name
        )

    @staticmethod
    def annotation_stem(path: Path) -> str:
        stem = path.stem

        if stem.endswith("_ball"):
            stem = stem[:-5]

        return stem

    @staticmethod
    def get_video_metadata(video_path: Path) -> dict | None:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            cap.release()
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        if (
                width <= 0
                or height <= 0
                or fps <= 0
                or frame_count <= 0
        ):
            return None

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        }
