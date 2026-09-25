# 2026 - copyright - all rights reserved - clayton thomas baber

import os
import io
import json
import zipfile
import numpy as np
import torch
from torch import tensor, float32, from_numpy
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
from collections import deque
from cube import Cube


# ==============================================================================
# 1. ARCHIVE PROCESSING & CYCLE COMPUTATION (GPU-Chunked)
# ==============================================================================

class ArchiveProcessor:
    """
    Reads raw htm4.zip paths, steps endpoints via Cube class, calculates 
    permutation cycle lengths in GPU chunks, and saves a master memmap index.
    """
    @staticmethod
    def process_archive(
        zip_filepath: str,
        output_dir: str = "master_archive_data",
        chunk_size: int = 50000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        os.makedirs(output_dir, exist_ok=True)
        print(f"=== Processing Raw Archive: {zip_filepath} ===")

        # 1. Read zip file and parse path actions
        paths = []
        with zipfile.ZipFile(zip_filepath, 'r') as zf:
            txt_files = [f for f in zf.namelist() if f.endswith('.txt') and not f.startswith('__MACOSX')]
            target_filename = txt_files[0] if txt_files else zf.namelist()[0]

            with zf.open(target_filename, 'r') as f:
                text_stream = io.TextIOWrapper(f, encoding='utf-8')
                for line in tqdm(text_stream, desc="Parsing path notation", unit="lines"):
                    line = line.strip()
                    if not line:
                        continue
                    actions = [line[i:i+2] for i in range(0, len(line), 2)]
                    path = [Cube.notation[action] for action in actions]
                    paths.append(path)

        paths_np = np.array(paths, dtype=np.uint8)
        num_paths, path_len = paths_np.shape
        print(f"Parsed {num_paths:,} paths of length {path_len}.")

        # 2. Compute final endpoints per path on CPU via Cube
        endpoints_np = np.zeros((num_paths, 54), dtype=np.uint8)
        for idx in tqdm(range(num_paths), desc="Computing path endpoints"):
            c = Cube()
            for act in paths_np[idx]:
                c.act(act)
            endpoints_np[idx] = c.state.copy()

        # 3. Vectorized GPU Chunked Permutation Cycle Calculation
        action_perms_gpu = torch.tensor(Cube.actions, device=device, dtype=torch.long)

        cycle_lengths = np.zeros(num_paths, dtype=np.int32)
        solved_state = torch.arange(54, device=device, dtype=torch.uint8)

        # Process cycle length detection in memory-friendly chunks
        for start_idx in range(0, num_paths, chunk_size):
            end_idx = min(start_idx + chunk_size, num_paths)
            chunk_paths = torch.tensor(paths_np[start_idx:end_idx], device=device, dtype=torch.long)
            b_size = chunk_paths.shape[0]

            # Compound permutation map for 1 full pass of each path
            curr_perm = torch.arange(54, device=device, dtype=torch.long).unsqueeze(0).repeat(b_size, 1)
            for step in range(path_len):
                step_actions = chunk_paths[:, step]
                step_perms = action_perms_gpu[step_actions]
                # Chain transformations forward
                curr_perm = curr_perm.gather(1, step_perms)

            # Apply compound permutation repeatedly to solved state until it returns to solved
            states = solved_state.unsqueeze(0).repeat(b_size, 1)
            cycles = torch.zeros(b_size, device=device, dtype=torch.int32)
            active_mask = torch.ones(b_size, device=device, dtype=torch.bool)

            while active_mask.any():
                cycles[active_mask] += 1
                # Advance active states by compound permutation
                states[active_mask] = states[active_mask].gather(1, curr_perm[active_mask])
                
                is_solved = (states == solved_state.unsqueeze(0)).all(dim=1)
                active_mask = active_mask & (~is_solved)

            cycle_lengths[start_idx:end_idx] = cycles.cpu().numpy()

        # 4. Save Master Memory-Mapped Archive Files
        print("Writing master memory-mapped index to disk...")
        paths_mmap = np.memmap(os.path.join(output_dir, "master_paths.dat"), dtype='uint8', mode='w+', shape=(num_paths, path_len))
        endpoints_mmap = np.memmap(os.path.join(output_dir, "master_endpoints.dat"), dtype='uint8', mode='w+', shape=(num_paths, 54))
        cycles_mmap = np.memmap(os.path.join(output_dir, "master_cycles.dat"), dtype='int32', mode='w+', shape=(num_paths,))

        paths_mmap[:] = paths_np[:]
        endpoints_mmap[:] = endpoints_np[:]
        cycles_mmap[:] = cycle_lengths[:]

        paths_mmap.flush()
        endpoints_mmap.flush()
        cycles_mmap.flush()

        # Save metadata and index offset tables
        unique_cycles, counts = np.unique(cycle_lengths, return_counts=True)
        meta = {
            "num_paths": num_paths,
            "path_len": path_len,
            "cycle_counts": {int(c): int(cnt) for c, cnt in zip(unique_cycles, counts)}
        }
        with open(os.path.join(output_dir, "master_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Done! Master archive distribution saved to {output_dir}\n")


class MasterArchiveReader:
    """Reader class to inspect master archive metadata and extract paths by cycle length."""
    def __init__(self, archive_dir: str = "master_archive_data"):
        with open(os.path.join(archive_dir, "master_meta.json"), "r") as f:
            self.meta = json.load(f)

        self.num_paths = self.meta["num_paths"]
        self.path_len = self.meta["path_len"]

        self.paths = np.memmap(os.path.join(archive_dir, "master_paths.dat"), dtype='uint8', mode='r', shape=(self.num_paths, self.path_len))
        self.endpoints = np.memmap(os.path.join(archive_dir, "master_endpoints.dat"), dtype='uint8', mode='r', shape=(self.num_paths, 54))
        self.cycles = np.memmap(os.path.join(archive_dir, "master_cycles.dat"), dtype='int32', mode='r', shape=(self.num_paths,))

    def print_metadata(self):
        print(f"Archive Total Paths: {self.num_paths:,}")
        print("Cycle Length Distribution:")
        for c, cnt in sorted(self.meta["cycle_counts"].items(), key=lambda x: int(x[0])):
            print(f"  Cycle Length {int(c):>3d}: {cnt:>8,} paths")

    def get_paths_by_cycle(self, cycle_length: int) -> np.ndarray:
        indices = np.where(self.cycles == cycle_length)[0]
        return np.array(self.paths[indices])


# ==============================================================================
# 2. DATASET BUILDER (Selection, Splits, & Offline Transformations)
# ==============================================================================

class DatasetBuilder:
    """
    Selects seed paths via Trie prefix filter and FPS, splits cleanly into Train/Val,
    applies offline transformations (shifts -> 48 action transformations -> anti-action inversions),
    computes off-path BFS distance penalization, and exports final dataset memmaps.
    """
    def __init__(self, archive_dir: str = "master_archive_data"):
        self.reader = MasterArchiveReader(archive_dir)

    def build_dataset(
        self,
        output_dir: str = "precomputed_rubiks_data",
        num_select: int = 1000,
        cycle_filter: int | None = None,
        max_shared_prefix: int = 3,
        prefix_decay: float = 0.85,
        train_split: float = 0.9,
        shift_offsets: list[int] | None = None,
        transform_indices: list[int] | None = None,
        off_path_penalty: float = 0.1,
        branch_depth_k: int = 2,
        random_seed: int | None = 42
    ):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n=== Building Dataset in {output_dir} ===")

        # 1. Filter candidate paths by cycle length
        if cycle_filter is not None:
            candidate_paths = self.reader.get_paths_by_cycle(cycle_filter)
        else:
            candidate_paths = np.array(self.reader.paths)

        print(f"Candidate pool size: {len(candidate_paths):,} paths.")

        # 2. Trie Prefix Filtering
        candidate_paths = self._filter_trie_prefixes(candidate_paths, max_shared_prefix)
        print(f" {len(candidate_paths)} Remaining paths")

        # 3. Farthest Point Sampling (FPS)
        selected_paths = self._run_farthest_point_sampling(
            candidate_paths, num_select, prefix_decay, random_seed=random_seed
        )

        # 4. Clean Train / Val Seed Split BEFORE any transformations
        if random_seed is not None:
            np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(selected_paths))
        split_idx = int(len(selected_paths) * train_split)

        train_seeds = selected_paths[shuffled_indices[:split_idx]]
        val_seeds = selected_paths[shuffled_indices[split_idx:]]

        print(f"Clean Seed Split -> Train Seeds: {len(train_seeds)}, Val Seeds: {len(val_seeds)}")

        # 5. Process Train & Val Sets Separately
        # Train defaults to all 48 action transformations (O_h group symmetry expansion)
        self._process_and_save_subset(
            train_seeds,
            os.path.join(output_dir, "train"),
            shift_offsets=shift_offsets or list(range(20)),
            transform_indices=transform_indices or list(range(48)),
            off_path_penalty=off_path_penalty,
            branch_depth_k=branch_depth_k,
            subset_name="Train"
        )

        self._process_and_save_subset(
            val_seeds,
            os.path.join(output_dir, "val"),
            shift_offsets=[0],
            transform_indices=[0],
            off_path_penalty=off_path_penalty,
            branch_depth_k=branch_depth_k,
            subset_name="Validation"
        )

        print(f"=== Dataset Pipeline Completed Successfully ===\n")

    def _filter_trie_prefixes(
        self,
        paths: np.ndarray,
        max_shared_prefix: int,
        max_paths_per_prefix: int = 1,
    ) -> np.ndarray:
        prefix_counts = {}
        unique_indices = []

        for idx, path in enumerate(
            tqdm(paths, desc="Trie prefix filtering", unit="path")
        ):
            # Hash the first max_shared_prefix moves as a tuple
            prefix = tuple(path[:max_shared_prefix])
            count = prefix_counts.get(prefix, 0)

            if count < max_paths_per_prefix:
                prefix_counts[prefix] = count + 1
                unique_indices.append(idx)

        return paths[unique_indices]

    def _run_farthest_point_sampling(
        self, candidates: np.ndarray, num_select: int, prefix_decay: float = 0.85, random_seed: int = 42
    ) -> np.ndarray:
        num_candidates, path_length = candidates.shape
        num_select = min(num_select, num_candidates)

        rng = np.random.default_rng(random_seed)
        first_idx = rng.integers(0, num_candidates)

        weights = (prefix_decay ** np.arange(path_length, dtype=np.float32))
        selected_indices = [first_idx]

        mismatches = (candidates != candidates[first_idx])
        min_distances = np.sum(mismatches * weights, axis=1)

        for _ in tqdm(range(1, num_select), desc="Farthest Point Sampling", unit="seed"):
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)

            mismatches = (candidates != candidates[next_idx])
            distances_to_latest = np.sum(mismatches * weights, axis=1)
            min_distances = np.minimum(min_distances, distances_to_latest)

        return candidates[selected_indices]

    def _process_and_save_subset(
        self,
        seed_paths: np.ndarray,
        output_dir: str,
        shift_offsets: list[int],
        transform_indices: list[int],
        off_path_penalty: float,
        branch_depth_k: int,
        subset_name: str
    ):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing {subset_name} Subset Offline Transformations...")

        # Step A: Offline Cyclic Shifts
        shifted_paths = self._apply_cyclic_shifts(seed_paths, shift_offsets)

        # Step B: Offline Action Transforms across full O_h group (48 permutations)
        action_transforms = np.array(Cube.action_transforms, dtype=np.uint8)
        transformed_paths = self._apply_action_transformations(shifted_paths, action_transforms, transform_indices)

        # Step C: Offline Anti-Action Inversions
        expanded_paths = self._apply_reversed_antiactions(transformed_paths)

        # Step D: Deduplicate expanded paths
        expanded_paths = np.unique(expanded_paths, axis=0)
        print(f"Final Expanded Unique Paths ({subset_name}): {len(expanded_paths):,}")

        # Step E: BFS Off-Path Depth-k Distance Penalization
        inputs_list, distances_list = self._compute_thickened_distances(
            expanded_paths, off_path_penalty=off_path_penalty, branch_depth_k=branch_depth_k
        )

        # Step F: Export to NumPy Memmap
        num_samples = len(inputs_list)
        inputs_mmap = np.memmap(os.path.join(output_dir, "inputs.dat"), dtype='uint8', mode='w+', shape=(num_samples, 54))
        targets_mmap = np.memmap(os.path.join(output_dir, "targets.dat"), dtype='float32', mode='w+', shape=(num_samples,))

        inputs_mmap[:] = inputs_list[:]
        targets_mmap[:] = distances_list[:]

        inputs_mmap.flush()
        targets_mmap.flush()

        # Statistics summary
        rounded_dists = np.round(distances_list).astype(int)
        unique_d, counts = np.unique(rounded_dists, return_counts=True)
        counts_dict = {int(d): int(cnt) for d, cnt in zip(unique_d, counts)}

        meta = {
            "num_samples": num_samples,
            "counts": counts_dict
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {num_samples:,} samples for {subset_name}. Distribution: {counts_dict}")

    def _apply_cyclic_shifts(self, paths: np.ndarray, shift_offsets: list[int]) -> np.ndarray:
        path_len = paths.shape[1]
        valid_shifts = list(dict.fromkeys([offset % path_len for offset in shift_offsets]))
        shifts = [np.roll(paths, -offset, axis=1) for offset in valid_shifts]
        return np.vstack(shifts)

    def _apply_action_transformations(
        self, paths: np.ndarray, transform_table: np.ndarray, transform_indices: list[int]
    ) -> np.ndarray:
        transformed_list = [transform_table[t, paths] for t in transform_indices]
        return np.vstack(transformed_list)

    def _apply_reversed_antiactions(self, paths: np.ndarray) -> np.ndarray:
        actions_mat = np.array(Cube.actions, dtype=np.int32)
        identity = np.arange(actions_mat.shape[1])
        antiaction_map = np.zeros(18, dtype=np.uint8)

        for i in range(18):
            for j in range(18):
                if np.array_equal(actions_mat[j, actions_mat[i]], identity):
                    antiaction_map[i] = j
                    break

        reversed_paths = antiaction_map[np.flip(paths, axis=1)]
        return np.vstack((paths, reversed_paths))

    def _compute_thickened_distances(
        self, paths: np.ndarray, off_path_penalty: float, branch_depth_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # Dictionary storing state bytes -> (penalized_distance, raw_distance)
        contents = {}

        # Seed with solved state
        solved_cube = Cube()
        contents[bytes(solved_cube.state)] = (0.0, 0)

        for path in tqdm(paths, desc="Thickening Off-Path BFS", unit="path"):
            cube = Cube()
            cube.reset()

            for i, action in enumerate(path, start=1):
                cube.act(action)
                state_bytes = bytes(cube.state)
                raw_d = i
                penalized_d = float(raw_d)

                if state_bytes not in contents or penalized_d < contents[state_bytes][0]:
                    contents[state_bytes] = (penalized_d, raw_d)

                # Depth-k BFS Branching off path
                frontier = deque([(cube.state, raw_d, 0)])  # (state, base_raw_d, depth)
                while frontier:
                    curr_state, base_raw_d, depth = frontier.popleft()
                    if depth >= branch_depth_k:
                        continue

                    temp_cube = Cube()
                    temp_cube.setState(curr_state)

                    for adj in temp_cube.getAdjacent():
                        adj_bytes = bytes(adj.state)
                        adj_depth = depth + 1
                        adj_raw_d = base_raw_d + adj_depth
                        additive_penalty = min(1.0, adj_depth * off_path_penalty)
                        adj_penalized_d = base_raw_d + adj_depth + additive_penalty

                        if adj_bytes not in contents or adj_penalized_d < contents[adj_bytes][0]:
                            contents[adj_bytes] = (adj_penalized_d, adj_raw_d)
                            frontier.append((adj.state, base_raw_d, adj_depth))

        num_samples = len(contents)
        inputs = np.zeros((num_samples, 54), dtype=np.uint8)
        targets = np.zeros((num_samples,), dtype=np.float32)

        for idx, (state_bytes, (pen_d, _)) in enumerate(contents.items()):
            inputs[idx] = np.frombuffer(state_bytes, dtype='uint8')
            targets[idx] = pen_d

        return inputs, targets


# ==============================================================================
# 3. PYTORCH DATASET & DATAMODULE (Online Rotations & Target Formatting)
# ==============================================================================

class RubikMmapDataset(Dataset):
    """Zero-RAM PyTorch Dataset reading from numpy memmaps with dynamic target transformation."""
    def __init__(
        self,
        subset_dir: str,
        input_type: str = "onehot",  # Options: "onehot", "bipolarhot"
        target_type: str = "gaussian",  # Options: "gaussian", "ordinal", "onehot", "scalar"
        num_classes: int = 22,
        sigma: float = 1.0
    ):
        meta_path = os.path.join(subset_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found in {subset_dir}.")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.num_samples = meta['num_samples']
        self.target_type = target_type
        self.num_classes = num_classes
        self.sigma = sigma

        self.input_type = input_type
        # Prebuild mapping lookup matrix to avoid per-item computation
        if self.input_type == "onehot":
            self.lookup_map = np.eye(6, dtype=np.float32)  # Output shape per sticker: (6,)
        elif self.input_type in ("bipolarhot", "bipolar"):
            self.lookup_map = np.array(Cube.bipolar_map)  # Output shape per sticker: (3,)
        else:
            raise ValueError(f"Invalid input_type '{input_type}'. Expected 'onehot' or 'bipolarhot'.")

        self.inputs = np.memmap(os.path.join(subset_dir, "inputs.dat"), dtype='uint8', mode='r', shape=(self.num_samples, 54))
        self.targets = np.memmap(os.path.join(subset_dir, "targets.dat"), dtype='float32', mode='r', shape=(self.num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Convert sticker indices (0..53) to color indices (0..5) matching Cube.toColor()
        state_ints = self.inputs[idx]
        color_indices = state_ints // 9

        # Map color indices to vectors and flatten (54, 6) -> 324 or (54, 3) -> 162
        x_tensor = from_numpy(self.lookup_map[color_indices].ravel())
        penalized_d = float(self.targets[idx])

        # 2. Selectable Target Formats
        if self.target_type == "gaussian":
            # Gaussian-smoothed probability density distribution
            bins = np.arange(self.num_classes, dtype=np.float32)
            y_target = np.exp(-0.5 * ((bins - penalized_d) / self.sigma) ** 2)
            y_target /= (y_target.sum() + 1e-8)  # Normalize density
            y_tensor = from_numpy(y_target.astype(np.float32))

        elif self.target_type == "ordinal":
            # Cumulative binary target vector for ordinal ranking
            bins = np.arange(self.num_classes, dtype=np.float32)
            y_target = (penalized_d >= bins).astype(np.float32)
            y_tensor = from_numpy(y_target)

        elif self.target_type == "onehot":
            # Round float distance to closest class index and clamp within [0, num_classes - 1]
            class_idx = int(np.clip(np.round(penalized_d), 0, self.num_classes - 1))
            y_target = np.zeros(self.num_classes, dtype=np.float32)
            y_target[class_idx] = 1.0
            y_tensor = from_numpy(y_target)

        else:  # "scalar"
            y_tensor = tensor(penalized_d, dtype=float32)

        return x_tensor, y_tensor


class RubikDataModule(LightningDataModule):
    """Lightning DataModule handling batching and online whole-cube rotational augmentation."""
    def __init__(
        self,
        data_dir: str = "precomputed_rubiks_data",
        input_type: str = "onehot",  # Options: "onehot", "bipolarhot"
        target_type: str = "gaussian",
        num_classes: int = 22,
        batch_size: int = 1024,
        num_workers: int = 4,
        enable_online_rotations: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.input_type = input_type
        self.target_type = target_type
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.enable_online_rotations = enable_online_rotations
        self.rotation_permutations = torch.tensor(Cube.rotations, dtype=torch.long)

        # Set feature dimension per sticker (6 for onehot, 3 for bipolarhot)
        self.feat_dim = 3 if self.input_type.lower() in ("bipolarhot", "bipolar") else 6
        self.input_dim = 54 * self.feat_dim

    def setup(self, stage=None):
        self.train_ds = RubikMmapDataset(os.path.join(self.data_dir, "train"), input_type=self.input_type, target_type=self.target_type, num_classes=self.num_classes)
        self.val_ds = RubikMmapDataset(os.path.join(self.data_dir, "val"), input_type=self.input_type, target_type=self.target_type, num_classes=self.num_classes)

    def collate_fn_train(self, batch):
        inputs, targets = zip(*batch)
        x = torch.stack(inputs)  # Shape: (B, 324)
        y = torch.stack(targets)

        if self.enable_online_rotations:
            # Apply dynamic batch-level 24 whole-cube rotations
            batch_size = x.shape[0]
            rand_rot_indices = torch.randint(0, 24, (batch_size,))
            x_reshaped = x.view(batch_size, 54, self.feat_dim)

            # Permute 54 sticker positions per randomly chosen rotation
            rot_perms = self.rotation_permutations[rand_rot_indices]  # Shape: (B, 54)
            x_rotated = torch.gather(x_reshaped, 1, rot_perms.unsqueeze(-1).expand(-1, -1, self.feat_dim))
            x = x_rotated.reshape(batch_size, self.input_dim)

        return x, y

    def collate_fn_val(self, batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=self.collate_fn_train
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            collate_fn=self.collate_fn_val
        )


# ==============================================================================
# 4. PIPELINE EXECUTION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    ARCHIVE_PATH = "assets/htm4.zip"
    MASTER_DIR = "data/master_archive_data"
    DATASET_DIR = "data/precomputed_rubiks_data"

    # --- Step 1: Pre-process Raw Zip Archive (If not generated) ---
    if not os.path.exists(os.path.join(MASTER_DIR, "master_meta.json")):
        ArchiveProcessor.process_archive(ARCHIVE_PATH, output_dir=MASTER_DIR, chunk_size=50000)

    # Inspect master archive metadata
    reader = MasterArchiveReader(MASTER_DIR)
    reader.print_metadata()

    # --- Step 2: Build Dataset (Selection, Splits, Offline Shifts/Rotations/Inversions) ---
    builder = DatasetBuilder(MASTER_DIR)
    builder.build_dataset(
        output_dir=DATASET_DIR,
        cycle_filter=2,
        num_select=18,               # Number of candidate seeds
        max_shared_prefix=3,
        shift_offsets=[0, 4, 8, 12, 16],
        transform_indices=list(range(48)), # Applies full 48 O_h symmetry transformations
        off_path_penalty=0.1,
        branch_depth_k=2,
        random_seed=None
    )

    # --- Step 3: Test DataModule & Dynamic Whole-Cube Rotations ---
    dm = RubikDataModule(
        data_dir=DATASET_DIR,
        input_type="onehot",
        target_type="onehot",
        batch_size=768,
        num_workers=4,
        num_classes=21,
        enable_online_rotations = False
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    x_batch, y_batch = next(iter(train_loader))

    print(f"Sample Batch Inputs Shape:  {x_batch.shape}")
    print(f"Sample Batch Targets Shape: {y_batch.shape}")
    print(f"First Target Density Sum:   {y_batch[0].sum().item():.4f}\n")

    x = x_batch[0]
    y = y_batch[0]
    print(f"\n\n{x}\n\n{y}")
