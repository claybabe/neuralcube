# 2026 - copyright - all rights reserved - clayton thomas baber

import os
import io
import json
import zipfile
import numpy as np
import torch
from torch import tensor, float32, from_numpy
from torch.utils.data import Dataset, DataLoader, Sampler
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
# 2. DATASET BUILDER (Global Distance Building & Stratification)
# ==============================================================================

class DatasetBuilder:
    """
    Pipeline for generating global manifold distances, train/val splits,
    meta-group bin packing, and stratified datasets.
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
        target_num_meta_groups: int = 5,
        anchors: list[int] = [0,1,2,3],
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
        print(f"{len(candidate_paths)} Remaining paths after Trie filtering.")

        # 3. Farthest Point Sampling (FPS)
        selected_paths = self._run_farthest_point_sampling(
            candidate_paths, num_select, prefix_decay, random_seed=random_seed
        )

        # 4. Fully expand all selected seed paths globally
        print("\nExpanding seed paths globally across shifts, actions, and inversions...")
        shifted_paths = self._apply_cyclic_shifts(selected_paths, shift_offsets or list(range(20)))
        action_transforms = np.array(Cube.action_transforms, dtype=np.uint8)
        transformed_paths = self._apply_action_transformations(
            shifted_paths, action_transforms, transform_indices or list(range(48))
        )
        all_expanded_paths = self._apply_reversed_antiactions(transformed_paths)
        all_expanded_paths = np.unique(all_expanded_paths, axis=0)
        print(f"Total fully expanded unique global paths: {len(all_expanded_paths):,}")

        # 5. Global Distance Table Construction with Endpoint Tracking & BFS
        print("\nBuilding global manifold distance table and tracking path endpoints...")
        distance_table, path_endpoints = self._compute_global_distance_table(
            all_expanded_paths, off_path_penalty=off_path_penalty, branch_depth_k=branch_depth_k
        )

        # 6. Train/Val Path-Level Split
        if random_seed is not None:
            np.random.seed(random_seed)

        num_paths = len(all_expanded_paths)
        shuffled_path_indices = np.random.permutation(num_paths)
        split_idx = int(num_paths * train_split)

        train_path_indices = set(shuffled_path_indices[:split_idx])
        val_path_indices = set(shuffled_path_indices[split_idx:])

        print(f"Split {num_paths:,} expanded paths -> Train Paths: {len(train_path_indices):,}, Val Paths: {len(val_path_indices):,}")

        # Extract training endpoints
        train_endpoints = [path_endpoints[i] for i in train_path_indices]
        train_endpoints_np = np.array(train_endpoints, dtype=np.uint8)

        # 7. Disjoint State Extraction: Remove validation path states from training distance lookup
        print("Separating global states into disjoint Train and Validation sets...")
        val_state_bytes = set()
        for i in val_path_indices:
            for s_bytes in self._get_path_states(all_expanded_paths[i]):
                val_state_bytes.add(s_bytes)

        train_inputs, train_penalized_targets, train_raw_targets = [], [], []
        val_inputs, val_penalized_targets, val_raw_targets = [], [], []

        for s_bytes, (pen_d, raw_d) in distance_table.items():
            state_arr = np.frombuffer(s_bytes, dtype='uint8')
            # Keep Distance 0 and 1 in training guaranteed
            if raw_d in anchors:
                train_inputs.append(state_arr)
                train_penalized_targets.append(pen_d)
                train_raw_targets.append(raw_d)
                # Also include in val for validation checking
                val_inputs.append(state_arr)
                val_penalized_targets.append(pen_d)
                val_raw_targets.append(raw_d)
            elif s_bytes in val_state_bytes:
                val_inputs.append(state_arr)
                val_penalized_targets.append(pen_d)
                val_raw_targets.append(raw_d)
            else:
                train_inputs.append(state_arr)
                train_penalized_targets.append(pen_d)
                train_raw_targets.append(raw_d)

        train_inputs = np.array(train_inputs, dtype=np.uint8)
        train_penalized_targets = np.array(train_penalized_targets, dtype=np.float32)
        train_raw_targets = np.array(train_raw_targets, dtype=np.int32)

        val_inputs = np.array(val_inputs, dtype=np.uint8)
        val_penalized_targets = np.array(val_penalized_targets, dtype=np.float32)
        val_raw_targets = np.array(val_raw_targets, dtype=np.int32)

        print(f"Extracted Dataset Split -> Train States: {len(train_inputs):,}, Val States: {len(val_inputs):,}")

        # 8. Contiguous Smooth Meta-Grouping
        print("\nConstructing Meta-Group Bins for Training Set...")
        unique_raw, raw_counts = np.unique(train_raw_targets, return_counts=True)
        raw_dist_counts = {int(d): int(cnt) for d, cnt in zip(unique_raw, raw_counts)}

        meta_groups = self._pack_meta_groups(raw_dist_counts, target_num_bins=target_num_meta_groups)

        # Assign meta-group index to each training sample
        raw_to_group = {}
        for g_idx, group in enumerate(meta_groups):
            for d in group["raw_distances"]:
                raw_to_group[d] = g_idx

        train_meta_group_ids = np.array([raw_to_group[d] for d in train_raw_targets], dtype=np.int32)

        # 9. Compute Effective Class Weights based on Sampler Intra-Group Frequency
        meta_group_counts = np.array([g["total_count"] for g in meta_groups], dtype=np.float32)
        meta_group_weights = meta_group_counts.sum() / (len(meta_groups) * meta_group_counts)

        num_classes = int(np.max(train_raw_targets)) + 1
        class_weights = np.zeros(num_classes, dtype=np.float32)

        for g in meta_groups:
            group_total = float(g["total_count"])
            for d in g["raw_distances"]:
                count_d = max(raw_dist_counts.get(d, 0), 1)
                # Class weight relative to intra-group probability distribution
                class_weights[d] = group_total / count_d

        # Smooth and clamp extreme weights to stabilize training
        class_weights = np.sqrt(class_weights)
        class_weights = np.clip(class_weights, a_min=0.1, a_max=20.0)
        active_mask = class_weights > 0
        if active_mask.any():
            class_weights[active_mask] /= class_weights[active_mask].mean()

        # Display Dataset Statistics Summary
        print("\n" + "=" * 60)
        print("RAW DISTANCE COUNTS (Training Set):")
        for d in sorted(raw_dist_counts.keys()):
            print(f"  Distance {d:>2d}: {raw_dist_counts[d]:>8,} samples")

        print("\nMETA-GROUPS DISTANCE BINS & GROUP COUNTS:")
        for g_idx, group in enumerate(meta_groups):
            d_list = sorted(group["raw_distances"])
            print(f"  Group {g_idx}: Bins {d_list} -> Total Count: {group['total_count']:>8,} | Weight: {meta_group_weights[g_idx]:.4f}")
        print("=" * 60 + "\n")

        # 10. Export Final Datasets and Endpoints to Memory Maps
        self._export_subset(
            os.path.join(output_dir, "train"),
            train_inputs,
            train_penalized_targets,
            train_raw_targets,
            meta_group_ids=train_meta_group_ids
        )

        self._export_subset(
            os.path.join(output_dir, "val"),
            val_inputs,
            val_penalized_targets,
            val_raw_targets
        )

        # Export training endpoints memmap
        endpoints_path = os.path.join(output_dir, "train_endpoints.dat")
        ep_mmap = np.memmap(endpoints_path, dtype='uint8', mode='w+', shape=train_endpoints_np.shape)
        ep_mmap[:] = train_endpoints_np[:]
        ep_mmap.flush()

        # Export metadata JSON
        meta_export = {
            "num_train_samples": len(train_inputs),
            "num_val_samples": len(val_inputs),
            "num_train_endpoints": len(train_endpoints_np),
            "raw_distance_counts": raw_dist_counts,
            "meta_groups": meta_groups,
            "meta_group_weights": meta_group_weights.tolist(),
            "class_weights": class_weights.tolist(),
            "num_classes": num_classes
        }
        with open(os.path.join(output_dir, "dataset_meta.json"), "w") as f:
            json.dump(meta_export, f, indent=2)

        print(f"=== Refactored Dataset Pipeline Completed Successfully ===\n")

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

    def _get_path_states(self, path: np.ndarray) -> list[bytes]:
        cube = Cube()
        cube.reset()
        states = [bytes(cube.state)]
        for action in path:
            cube.act(action)
            states.append(bytes(cube.state))
        return states

    def _compute_global_distance_table(
        self, paths: np.ndarray, off_path_penalty: float, branch_depth_k: int
    ) -> tuple[dict, list]:
        # Dictionary mapping state bytes -> (penalized_distance, raw_distance)
        distance_table = {}
        path_endpoints = []

        # Solved state initialization
        solved_cube = Cube()
        distance_table[bytes(solved_cube.state)] = (0.0, 0)

        for path_idx, path in enumerate(tqdm(paths, desc="Global BFS Distance Thickening", unit="path")):
            cube = Cube()
            cube.reset()

            for i, action in enumerate(path, start=1):
                cube.act(action)
                state_bytes = bytes(cube.state)
                raw_d = i
                penalized_d = float(raw_d)

                if state_bytes not in distance_table or penalized_d < distance_table[state_bytes][0]:
                    distance_table[state_bytes] = (penalized_d, raw_d)

                # Depth-k BFS Branching off-path
                frontier = deque([(cube.state, raw_d, 0)])
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

                        if adj_bytes not in distance_table or adj_penalized_d < distance_table[adj_bytes][0]:
                            distance_table[adj_bytes] = (adj_penalized_d, adj_raw_d)
                            frontier.append((adj.state, base_raw_d, adj_depth))

            # Remember final endpoint of this path as sticker indices
            path_endpoints.append(cube.state.copy())

        return distance_table, path_endpoints

    def _pack_meta_groups(self, raw_counts: dict[int, int], target_num_bins: int) -> list[dict]:
        """
        Groups contiguous distance classes into topologically smooth intervals.
        """
        sorted_distances = sorted(raw_counts.keys())
        chunks = np.array_split(sorted_distances, target_num_bins)

        bins = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            dist_list = [int(d) for d in chunk]
            total_cnt = sum(raw_counts[d] for d in dist_list)
            bins.append({
                "raw_distances": dist_list,
                "total_count": total_cnt
            })

        return bins

    def _export_subset(
        self,
        subset_dir: str,
        inputs: np.ndarray,
        penalized_targets: np.ndarray,
        raw_targets: np.ndarray,
        meta_group_ids: np.ndarray | None = None
    ):
        os.makedirs(subset_dir, exist_ok=True)
        num_samples = len(inputs)

        inputs_mmap = np.memmap(os.path.join(subset_dir, "inputs.dat"), dtype='uint8', mode='w+', shape=(num_samples, 54))
        penalized_mmap = np.memmap(os.path.join(subset_dir, "targets.dat"), dtype='float32', mode='w+', shape=(num_samples,))
        raw_mmap = np.memmap(os.path.join(subset_dir, "raw_targets.dat"), dtype='int32', mode='w+', shape=(num_samples,))

        inputs_mmap[:] = inputs[:]
        penalized_mmap[:] = penalized_targets[:]
        raw_mmap[:] = raw_targets[:]

        inputs_mmap.flush()
        penalized_mmap.flush()
        raw_mmap.flush()

        if meta_group_ids is not None:
            group_mmap = np.memmap(os.path.join(subset_dir, "meta_groups.dat"), dtype='int32', mode='w+', shape=(num_samples,))
            group_mmap[:] = meta_group_ids[:]
            group_mmap.flush()

        meta = {"num_samples": num_samples}
        with open(os.path.join(subset_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


# ==============================================================================
# 3. STRATIFIED SAMPLER & DATASET
# ==============================================================================

class StratifiedMetaGroupSampler(Sampler):
    """
    Probabilistic sampler that samples indices based on meta-group balance,
    guaranteeing uniform representation across non-contiguous distance meta-groups.
    """
    def __init__(self, meta_group_ids: np.ndarray, batch_size: int):
        self.meta_group_ids = meta_group_ids
        self.batch_size = batch_size
        self.num_samples = len(meta_group_ids)

        # Pre-group index offsets by meta-group
        self.group_indices = {}
        unique_groups = np.unique(meta_group_ids)
        for g_id in unique_groups:
            self.group_indices[g_id] = np.where(meta_group_ids == g_id)[0]

        self.num_groups = len(unique_groups)

    def __iter__(self):
        # Draw balanced batches across meta-groups
        sampled_indices = []
        samples_per_group = self.batch_size // self.num_groups

        # Shuffle indices inside each group
        shuffled_groups = {g_id: np.random.permutation(indices) for g_id, indices in self.group_indices.items()}
        group_ptrs = {g_id: 0 for g_id in self.group_indices}

        num_batches = self.num_samples // self.batch_size

        for _ in range(num_batches):
            batch_idxs = []
            for g_id in shuffled_groups:
                ptr = group_ptrs[g_id]
                g_idxs = shuffled_groups[g_id]
                
                # Wrap-around if pointer exceeds available group samples
                if ptr + samples_per_group > len(g_idxs):
                    shuffled_groups[g_id] = np.random.permutation(self.group_indices[g_id])
                    ptr = 0

                batch_idxs.extend(g_idxs[ptr : ptr + samples_per_group])
                group_ptrs[g_id] = ptr + samples_per_group

            np.random.shuffle(batch_idxs)
            sampled_indices.extend(batch_idxs)

        return iter(sampled_indices)

    def __len__(self):
        return (self.num_samples // self.batch_size) * self.batch_size


class RubikMmapDataset(Dataset):
    """Zero-RAM PyTorch Dataset reading from numpy memmaps with dynamic target transformation."""
    def __init__(
        self,
        subset_dir: str,
        input_type: str = "onehot",
        target_type: str = "gaussian",
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
        self.subset_dir = subset_dir

        self.input_type = input_type
        if self.input_type == "onehot":
            self.lookup_map = np.eye(6, dtype=np.float32)
        elif self.input_type in ("bipolarhot", "bipolar"):
            self.lookup_map = np.array(Cube.bipolar_map)
        else:
            raise ValueError(f"Invalid input_type '{input_type}'. Expected 'onehot' or 'bipolarhot'.")

        self.inputs = np.memmap(os.path.join(subset_dir, "inputs.dat"), dtype='uint8', mode='r', shape=(self.num_samples, 54))
        self.targets = np.memmap(os.path.join(subset_dir, "targets.dat"), dtype='float32', mode='r', shape=(self.num_samples,))
        self.raw_targets = np.memmap(os.path.join(subset_dir, "raw_targets.dat"), dtype='int32', mode='r', shape=(self.num_samples,))

        meta_groups_path = os.path.join(subset_dir, "meta_groups.dat")
        if os.path.exists(meta_groups_path):
            self.meta_groups = np.memmap(meta_groups_path, dtype='int32', mode='r', shape=(self.num_samples,))
        else:
            self.meta_groups = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        state_ints = self.inputs[idx]
        color_indices = state_ints // 9

        x_tensor = from_numpy(self.lookup_map[color_indices].ravel())
        penalized_d = float(self.targets[idx])

        if self.target_type == "gaussian":
            bins = np.arange(self.num_classes, dtype=np.float32)
            y_target = np.exp(-0.5 * ((bins - penalized_d) / self.sigma) ** 2)
            y_target /= (y_target.sum() + 1e-8)
            y_tensor = from_numpy(y_target.astype(np.float32))

        elif self.target_type == "ordinal":
            bins = np.arange(self.num_classes, dtype=np.float32)
            y_target = (penalized_d >= bins).astype(np.float32)
            y_tensor = from_numpy(y_target)

        elif self.target_type == "onehot":
          class_idx = int(np.clip(np.round(penalized_d), 0, self.num_classes - 1))
          y_target = np.zeros(self.num_classes, dtype=np.float32)
          y_target[class_idx] = 1.0
          y_tensor = from_numpy(y_target)

        else:  # "scalar"
            y_tensor = tensor(penalized_d, dtype=float32)

        return x_tensor, y_tensor


# ==============================================================================
# 4. PYTORCH LIGHTNING DATAMODULE
# ==============================================================================

class RubikDataModule(LightningDataModule):
    """Lightning DataModule with stratified meta-group sampling and online rotations."""
    def __init__(
        self,
        data_dir: str = "precomputed_rubiks_data",
        input_type: str = "onehot",
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

        self.feat_dim = 3 if self.input_type.lower() in ("bipolarhot", "bipolar") else 6
        self.input_dim = 54 * self.feat_dim

        # Load metadata JSON for loss function class weights
        meta_json_path = os.path.join(data_dir, "dataset_meta.json")
        if os.path.exists(meta_json_path):
            with open(meta_json_path, "r") as f:
                meta = json.load(f)
            self.class_weights = torch.tensor(meta["class_weights"], dtype=torch.float32)
            self.meta_group_weights = torch.tensor(meta["meta_group_weights"], dtype=torch.float32)
        else:
            self.class_weights = None
            self.meta_group_weights = None

    def setup(self, stage=None):
      self.train_ds = RubikMmapDataset(
          os.path.join(self.data_dir, "train"),
          input_type=self.input_type,
          target_type=self.target_type,
          num_classes=self.num_classes  # Ensure num_classes is forwarded
      )
      self.val_ds = RubikMmapDataset(
          os.path.join(self.data_dir, "val"),
          input_type=self.input_type,
          target_type=self.target_type,
          num_classes=self.num_classes  # Ensure num_classes is forwarded
      )

    def collate_fn_train(self, batch):
        inputs, targets = zip(*batch)
        x = torch.stack(inputs)
        y = torch.stack(targets)

        if self.enable_online_rotations:
            batch_size = x.shape[0]
            rand_rot_indices = torch.randint(0, 24, (batch_size,))
            x_reshaped = x.view(batch_size, 54, self.feat_dim)

            rot_perms = self.rotation_permutations[rand_rot_indices]
            x_rotated = torch.gather(x_reshaped, 1, rot_perms.unsqueeze(-1).expand(-1, -1, self.feat_dim))
            x = x_rotated.reshape(batch_size, self.input_dim)

        return x, y

    def collate_fn_val(self, batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)

    def train_dataloader(self):
        if self.train_ds.meta_groups is not None:
            sampler = StratifiedMetaGroupSampler(self.train_ds.meta_groups, batch_size=self.batch_size)
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=(self.num_workers > 0),
                collate_fn=self.collate_fn_train
            )
        else:
            return DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
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
# 5. PIPELINE EXECUTION ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    ARCHIVE_PATH = "assets/htm4.zip"
    MASTER_DIR = "data/master_archive_data"
    DATASET_DIR = "data/precomputed_rubiks_data"

    # Step 1: Pre-process Raw Zip Archive (If not generated)
    if not os.path.exists(os.path.join(MASTER_DIR, "master_meta.json")):
        ArchiveProcessor.process_archive(ARCHIVE_PATH, output_dir=MASTER_DIR, chunk_size=50000)

    reader = MasterArchiveReader(MASTER_DIR)
    reader.print_metadata()

    # Step 2: Build Global Distance Dataset with Non-Contiguous Meta-Groups
    builder = DatasetBuilder(MASTER_DIR)
    builder.build_dataset(
        output_dir=DATASET_DIR,
        cycle_filter=2,
        num_select=2,
        max_shared_prefix=3,
        shift_offsets=[0, 4, 8, 12, 16],
        transform_indices=list(range(48)),
        off_path_penalty=0.1,
        branch_depth_k=2,
        target_num_meta_groups=5,
        random_seed=42
    )

    # Step 3: Test Stratified DataModule & Batching
    dm = RubikDataModule(
        data_dir=DATASET_DIR,
        input_type="onehot",
        target_type="onehot",
        batch_size=768,
        num_workers=4,
        num_classes=23,
        enable_online_rotations=False
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    print(f"Train dataset size: {len(train_loader.dataset)} | Batches: {len(train_loader)}")
    print(f"Val dataset size:   {len(val_loader.dataset)}   | Batches: {len(val_loader)}")
    x_batch, y_batch = next(iter(train_loader))

    print(f"Sample Stratified Batch Inputs Shape:  {x_batch.shape}")
    print(f"Sample Stratified Batch Targets Shape: {y_batch.shape}")
    
    x = x_batch[0]
    y = y_batch[0]
    print(f"\n\n{x}\n\n{y}")
    if dm.class_weights is not None:
        print(f"Class Weights Loaded (Shape {dm.class_weights.shape}):\n{dm.class_weights}...")