# MUTE-DSS — Minimizing Underwater Radiated Noise in Ship Voyage Planning

MUTE-DSS is a digital-twin-based decision support system that plans ship voyages to reduce the underwater radiated noise (URN) that marine mammals are exposed to. It is built as a set of ROS 2 (Humble) packages that together simulate a ship, its source noise, sound propagation, and a population of marine mammals, and it optimizes the voyage in two stages:

1. **Route** — collision-free path planning with Batch Informed Trees (BIT\*, via OMPL) inside the shipping lanes.
2. **Speed profile** — a genetic algorithm (pymoo) that chooses speeds along the route to minimize cumulative noise exposure while meeting the voyage ETA.

The study area is the Salish Sea: voyages from the Port of Vancouver through the Strait of Georgia to the Strait of Juan de Fuca, with optimized voyages compared against baselines replayed from real AIS data.

📄 **Paper:** [Ocean Engineering, Vol. 343, 123343 (2026)](https://www.sciencedirect.com/science/article/pii/S0029801825030264) · **Preprint:** [arXiv:2508.01907](https://arxiv.org/abs/2508.01907)

---

> [!IMPORTANT]
> **The code and case studies used in the paper live on the [`JOE_mute_v1_cases`](https://github.com/akash-venkateshwaran/MUTE/tree/JOE_mute_v1_cases) branch, not on `master`.**
> `master` has diverged from the paper version (different speed-optimizer code and case-study outputs). To reproduce the paper, always check out `JOE_mute_v1_cases`.
>
> | Branch | What it is |
> |---|---|
> | [`JOE_mute_v1_cases`](https://github.com/akash-venkateshwaran/MUTE/tree/JOE_mute_v1_cases) | ✅ **Paper version.** Code, inputs and case studies used for the published results. |
> | `master` | Development snapshot; not the paper version. |
> | `mute_v1_cases`, `mute_v2` | Earlier / ongoing development branches. |

---

## Contents

- [System overview](#system-overview)
- [Repository layout](#repository-layout)
- [Case studies](#case-studies)
- [Setup](#setup)
- [Running a case study](#running-a-case-study)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## System overview

Each component is a ROS 2 package under `src/`. They talk over topics and services, so they are started as separate processes.

| Package | Executable / launch file | Role |
|---|---|---|
| `custom_interfaces` | — | Message and service definitions (`State`, `Path`, `Speed`, `SPL`, `ShipInfo`, `BoundsService`, …) |
| `environment` | `envi_exe` | Serves bathymetry (GEBCO 2023), shoreline and shipping-lane polygons |
| `path_planning` | `navig_exe` | Loads the voyage, runs BIT\* route planning, publishes waypoints and ETA |
| `mammals` | `mammals_launch.py` → `mammal_manager` | Spawns one node per mammal, moves them, computes transmission loss (TL) from the ship |
| `noise_propagation_model` | — (library) | RBF surrogate model for transmission loss, trained on 3D ray-tracing results |
| `noise_source_model` | `noise_source_exe` | JOMOPANS-ECHO semi-empirical ship source spectrum; publishes `/source_noise_spl` |
| `speed_optimization` | `opt_exe` | Genetic-algorithm speed-profile optimizer; publishes `optimized_speeds` |
| `ship` | `ship_exe` | Simulates the ship along the waypoints/speeds, publishes `/ship_state` and `/clock`, records the run to a rosbag |
| `visualisation` | `visual_launch.py` | RViz2 scene + PlotJuggler layout (optional) |

---

## Repository layout

Layout of the `JOE_mute_v1_cases` branch (main items only):

```text
MUTE/
├── .devcontainer/
│   ├── Dockerfile              # ROS 2 Humble + OMPL (built from source with Python bindings) + Python deps
│   └── devcontainer.json       # VS Code Dev Container config (X11/WSLg GUI forwarding, host networking)
├── build.sh                    # colcon build (merge-install, symlink-install)
├── imgs/                       # Figure scripts and generated plots
└── src/
    ├── custom_interfaces/
    ├── environment/
    │   ├── bathy_data/         # GEBCO 2023 NetCDF
    │   ├── ship_lanes_data/    # Shipping-lane polygons (.shp)
    │   ├── shoreline_data/     # North American shoreline (.shp)
    │   └── ssp_data/           # Sound-speed profile (Salish Sea)
    ├── mammals/
    │   ├── case_studies/       # Mammal scenarios (Case_A, Case_B) ← INPUT
    │   ├── kde_models/         # Population / depth KDE models
    │   └── launch/mammals_launch.py
    ├── noise_propagation_model/
    │   └── noise_propagation_model/models/RBF/
    │       ├── model/              # Multi-frequency RBF TL model (Git LFS, ~1.3 GB)
    │       └── model_single_freq/  # Single-frequency RBF TL model (Git LFS, ~178 MB)
    ├── noise_source_model/
    ├── path_planning/
    │   ├── case_studies/       # Voyage definitions (Case_1/2/3/case_study.csv) ← INPUT
    │   └── preprocessing_case_study/  # AIS preprocessing scripts
    ├── ship/
    │   └── case_studies/       # Ship particulars (Case_N/ship.yaml) ← INPUT
    │                           # + all simulation / optimization results ← OUTPUT
    ├── speed_optimization/
    └── visualisation/
```

---

## Case studies

A run combines **one voyage** (ship + AIS track) with **one mammal scenario**.

### Voyages

Defined across two folders, matched by name:

| Voyage | Vessel | Length (ft) | Speed bounds (kn) | AIS track points |
|---|---|---|---|---|
| `Case_1` | STAR KIRKENES | 692.3 | 8.0 – 15.9 | 720 |
| `Case_2` | STAR KILIMANJARO | 685.0 | 8.0 – 15.9 | 721 |
| `Case_3` | OKIANA | 697.2 | 8.0 – 14.5 | 583 |

- [`src/path_planning/case_studies/Case_N/case_study.csv`](https://github.com/akash-venkateshwaran/MUTE/tree/JOE_mute_v1_cases/src/path_planning/case_studies) — AIS track, no header. Columns: `index, speed_over_ground [kn], longitude, latitude`. The first and last rows are the start and goal; the ETA is derived from the track.
- [`src/ship/case_studies/Case_N/ship.yaml`](https://github.com/akash-venkateshwaran/MUTE/tree/JOE_mute_v1_cases/src/ship/case_studies) — ship particulars:

  ```yaml
  ship_info:
    ship_type: 'Other'
    ship_name: 'STAR KIRKENES'
    max_v : 15.9        # knots
    min_v : 8.0         # knots
    ship_length: 692.257 # feet
  ```

### Mammal scenarios

[`src/mammals/case_studies/`](https://github.com/akash-venkateshwaran/MUTE/tree/JOE_mute_v1_cases/src/mammals/case_studies) holds one `m<species>_<id>_params.yaml` per animal:

| Scenario | Animals | Description |
|---|---|---|
| `Case_A` | 1 | One stationary animal (`speed: 0.0`) — the simplified scenario |
| `Case_B` | 5 | Five moving animals — the more realistic dynamic scenario |

```yaml
m0_2:
  ros__parameters:
    latitude: 48.51697818452588
    longitude: -122.94053967506218
    depth: 5.56972917595162     # m
    heading: 155.3218492434733  # deg
    speed: 1.1
    topic: m0_2_state
```

### Results already in the branch

[`src/ship/case_studies/`](https://github.com/akash-venkateshwaran/MUTE/tree/JOE_mute_v1_cases/src/ship/case_studies) is also where the simulator writes its results (see [Outputs](#outputs)). On `JOE_mute_v1_cases` it already contains:

- `Case_1/` — `optimal_path.csv`, GA populations `optimal_population_<gen>.csv` (generations 1 → 2200), and the recorded run in `ship_data/`.
- `Case_A/mammal_data/` — transmission-loss recording for the Case_A animal.

---

## Setup

The supported environment is the **VS Code Dev Container** in `.devcontainer/`. It is based on `althack/ros2:humble-dev` (Ubuntu 22.04, ROS 2 Humble, Python 3.10) and adds OMPL built from source with Python bindings, Gazebo, RViz2, PlotJuggler and the Python dependencies (pymoo, scikit-learn, scipy, geopandas, pyproj, netCDF4, cartopy, …).

### 1. Prerequisites (host machine)

- [Docker](https://docs.docker.com/get-docker/) (Docker Desktop with the WSL 2 backend on Windows)
- [VS Code](https://code.visualstudio.com/) with the **Dev Containers** extension (`ms-vscode-remote.remote-containers`)
- [Git LFS](https://git-lfs.com/) — **required**. The transmission-loss models, bathymetry meshes and rosbags are stored with LFS (~1.7 GB in total). Without it the `.pkl` models are 130-byte pointer files and the simulator cannot run.
- For the GUI tools (RViz2, PlotJuggler): Windows 11 with WSLg, or Linux with X11. The container gets `DISPLAY` from the host.
- Resources: the first image build compiles OMPL's Python bindings, which takes a long time (often 1 h or more) and needs a lot of RAM — 16 GB or more is recommended.

### 2. Clone the paper branch with LFS files

```bash
git lfs install
git clone -b JOE_mute_v1_cases https://github.com/akash-venkateshwaran/MUTE.git MUTE_v1
cd MUTE_v1
git lfs pull
```

Check that the models were downloaded (they should be hundreds of MB, not ~130 bytes):

```bash
ls -lh src/noise_propagation_model/noise_propagation_model/models/RBF/model*/
```

> [!TIP]
> Cloning into a folder named **`MUTE_v1`** puts the workspace at `/workspaces/MUTE_v1` inside the container, which matches the absolute paths hard-coded in some helper and plotting scripts (`imgs/*.py`, `speed_optimization/pop_cost_eval.py`). The ROS nodes themselves locate `src/` automatically and work from any folder name.

> [!NOTE]
> Git LFS has to be installed on the **host**. The Dockerfile's `pip3 install git-lfs` installs an unrelated Python package, not the `git-lfs` binary, so you can't pull LFS files from inside the container unless you install it there yourself (instructions are at the end of the Dockerfile).

### 3. Open in the Dev Container

1. Open the `MUTE_v1` folder in VS Code.
2. Run **Dev Containers: Reopen in Container** from the Command Palette.
3. Wait for the image build to finish (slow the first time; see above).

The container runs as user `ros` with host networking. `devcontainer.json` mounts `/tmp/.X11-unix` and `/mnt/wslg` for GUI forwarding.

### 4. Build the workspace

Inside the container terminal, from the workspace root:

```bash
cd /workspaces/MUTE_v1
bash build.sh               # colcon build --merge-install --symlink-install
source install/setup.bash   # new terminals do this automatically via ~/.bashrc
```

VS Code's default build task (**Ctrl+Shift+B**, "Build ROS2 Workspace") runs the same script.

Check that all packages registered:

```bash
ros2 pkg list | grep -E 'environment|path_planning|mammals|ship|speed_optimization|noise_source_model|noise_propagation_model|visualisation|custom_interfaces'
```

---

## Running a case study

### Step 1 — Select the voyage and mammal scenario

There are no launch arguments for this. The case names are hard-coded in four files (default: voyage `Case_3`, mammals `Case_B`):

| File | Line(s) | Sets |
|---|---|---|
| `src/path_planning/path_planning/py_navigation_system.py` | 49–50 | voyage CSV and export folder |
| `src/speed_optimization/speed_optimization/py_speed_optimizer.py` | 71, 74 | voyage; warm-start population file |
| `src/ship/ship/py_ship.py` | 44, 45 | voyage; mammal scenario |
| `src/mammals/launch/mammals_launch.py` | 33 | mammal scenario |

**All voyage names must match each other, and both mammal names must match each other.** To set them all at once, save this as `set_case.sh` in the workspace root:

```bash
#!/usr/bin/env bash
# Usage: bash set_case.sh <voyage 1|2|3> <mammals A|B>     e.g.  bash set_case.sh 1 A
set -euo pipefail
VOYAGE="Case_$1"; MAMMALS="Case_$2"
sed -i -E "s#'case_studies' / 'Case_[0-9]+'#'case_studies' / '${VOYAGE}'#g" \
  src/path_planning/path_planning/py_navigation_system.py \
  src/speed_optimization/speed_optimization/py_speed_optimizer.py \
  src/ship/ship/py_ship.py
sed -i -E "s#self.case_study = 'Case_[0-9]+'#self.case_study = '${VOYAGE}'#" \
  src/speed_optimization/speed_optimization/py_speed_optimizer.py
sed -i -E "s#'mammals' / 'case_studies' / 'Case_[A-Z]'#'mammals' / 'case_studies' / '${MAMMALS}'#" \
  src/ship/ship/py_ship.py
sed -i -E "s#^case_directory = 'case_studies/Case_[A-Z]'#case_directory = 'case_studies/${MAMMALS}'#" \
  src/mammals/launch/mammals_launch.py
```

Then rebuild (`bash build.sh`) to be safe.

### Step 2 — Start the nodes

Every node runs in its own terminal, from the **workspace root** (plots are written to the relative folder `imgs/`). The nodes ask questions on the terminal, so start them by hand rather than all together. They wait for each other's services and topics, so the order below is a guide rather than a strict requirement.

```bash
# Terminal 1 — environment services (shoreline, shipping lanes, bathymetry)
ros2 run environment envi_exe

# Terminal 2 — route planner (loads the voyage; menu: see table below)
ros2 run path_planning navig_exe

# Terminal 3 — mammals (needs the planner's bounds_service)
ros2 launch mammals mammals_launch.py

# Terminal 4 — speed optimizer (menu: see table below)
ros2 run speed_optimization opt_exe

# Terminal 5 — ship source-noise model
ros2 run noise_source_model noise_source_exe

# Terminal 6 — ship simulator (starts the clock and the rosbag recording)
ros2 run ship ship_exe

# Terminal 7 (optional) — RViz2 + PlotJuggler
ros2 launch visualisation visual_launch.py
```

### Step 3 — Answer the prompts

**Route planner (`navig_exe`)** and **speed optimizer (`opt_exe`)** each show a menu:

| Choice | `navig_exe` (route) | `opt_exe` (speed) |
|---|---|---|
| `1` | Replay the AIS route | Replay the AIS speeds |
| `2` | Optimize the route with BIT\* (re-planned during the voyage) | Optimize speeds with the GA |
| `3` | Optimize the route with BIT\* **once** | Warm-start the GA from a saved population (`optimal_population_2200.csv`, line 74) |

- **AIS baseline:** `navig_exe` → `1`, `opt_exe` → `1`
- **MUTE-DSS optimized voyage:** `navig_exe` → `3` (or `2`), `opt_exe` → `2`

When optimizing, `navig_exe` and `opt_exe` then list the mammal topics they found and ask `Do you want to process these topics? (y/n)` → answer **`y`** once all mammals are up (Case_A: 1 topic, Case_B: 5).

**Mammals launch** asks two questions:

1. `The directory …/src/mammals/case_studies/Case_X is not empty. Do you want to clear it?` → answer **`n`**.
   ⚠️ `y` deletes the mammal parameter YAMLs, and the run then starts with zero mammals. If that happens, restore them with `git checkout -- src/mammals/case_studies`.
2. `The directory …/src/ship/case_studies/Case_X/mammal_data already exists… delete it?` → `y` for a clean run.

**Ship (`ship_exe`)**: if `src/ship/case_studies/Case_N/ship_data` already exists it asks to delete it. Answer `y` to continue; `n` stops the node. Copy any results you want to keep first (this includes the `Case_1/ship_data` recording that ships with the branch).

> [!NOTE]
> The GA default is 1000 individuals × 2210 generations (`pop_size`, `num_gen` in `py_speed_optimizer.py`), and BIT\* runs for 500 s (`ompl_runtime` in `py_navigation_system.py`). A full optimized run is computationally expensive. Option `3` in `opt_exe` lets you resume from a saved population. On this branch only `Case_1` includes saved populations, so update line 74 when using it for another voyage.

---

## Outputs

All results go into `src/ship/case_studies/` (git-ignored by default) and `imgs/`:

| Location | Contents |
|---|---|
| `src/ship/case_studies/Case_N/ship_data/` | rosbag2 (sqlite3) of `/ship_state`, `/clock`, `/source_noise_spl` and every mammal state topic; `simulation_results.txt` with arrival time vs. ETA |
| `src/ship/case_studies/Case_N/optimal_path.csv` | BIT\* route waypoints |
| `src/ship/case_studies/Case_N/optimal_population_<gen>.csv` | GA population, saved every 200 generations |
| `src/ship/case_studies/Case_X/mammal_data/<id>_TL/` | Per-animal transmission-loss recording (rosbag) + `simulation_info.txt` |
| `imgs/OMPL_*.png`, `imgs/OMPL_convergence.png` | Route-planning plots and BIT\* convergence |
| `imgs/speed_opt_convergence.png`, `imgs/optimized_speed_and_positions.png` | GA convergence and optimized speed profile |

Inspect a recording with:

```bash
ros2 bag info src/ship/case_studies/Case_1/ship_data
```

> [!WARNING]
> Mammal TL recordings are stored by **mammal scenario** (`Case_A`/`Case_B`), not by voyage, and ship recordings by **voyage** only. Running a baseline and an optimized voyage — or two voyages with the same mammal scenario — overwrites the previous outputs. Copy `src/ship/case_studies/` somewhere else after each run.

The figure scripts in `imgs/` (e.g. `plotting_case_studies.py`, `plotting_mammal_distribution.py`) and `speed_optimization/pop_cost_eval.py` (GA convergence analysis) use absolute `/workspaces/MUTE_v1/...` paths; edit them if your workspace path is different.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `_pickle.UnpicklingError` / `invalid load key 'v'` when loading the TL model | LFS files are missing. Run `git lfs pull` on the host. |
| Image build stops at `msttcorefonts` (EULA) | Add `RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" \| debconf-set-selections` before that line in the Dockerfile. |
| OMPL build runs out of memory or takes very long | Give Docker more RAM, or close other workloads. The build already uses only half the CPU cores to save memory. |
| Container won't start on macOS / non-WSL Linux because of `/mnt/wslg` | Remove the `--volume=/mnt/wslg:/mnt/wslg` entry from `runArgs` in `devcontainer.json`. |
| RViz2 / PlotJuggler won't open | Make sure the host has an X server/WSLg, `DISPLAY` is set, and on Linux run `xhost +local:` on the host. |
| Error loading pickled models after a fresh image build | The Dockerfile installs the latest `scikit-learn`/`scipy`/`pymoo`. If a newer version can't read the saved models, pin those packages to versions from around the paper's development period. |
| `ship_exe` keeps printing `Mismatch between number of mammal state topics …` | The mammal scenario in `py_ship.py` doesn't match the one in `mammals_launch.py`, or the mammals launch hasn't started yet. |
| `FileNotFoundError` for `imgs/…png` | Start the nodes from the workspace root. |

---

## Citation

If you use MUTE-DSS or this code in your research, please cite:

> A. Venkateshwaran, I. K. Deo, and R. K. Jaiman, "MUTE-DSS: A digital-twin-based decision support system for minimizing underwater radiated noise in ship voyage planning," *Ocean Engineering*, vol. 343, p. 123343, 2026. doi: [10.1016/j.oceaneng.2025.123343](https://doi.org/10.1016/j.oceaneng.2025.123343)

```bibtex
@article{venkateshwaran2026mutedss,
  title   = {{MUTE-DSS}: A digital-twin-based decision support system for minimizing underwater radiated noise in ship voyage planning},
  author  = {Venkateshwaran, Akash and Deo, Indu Kant and Jaiman, Rajeev K.},
  journal = {Ocean Engineering},
  volume  = {343},
  pages   = {123343},
  year    = {2026},
  doi     = {10.1016/j.oceaneng.2025.123343},
  url     = {https://www.sciencedirect.com/science/article/pii/S0029801825030264}
}
```

Preprint: [arXiv:2508.01907](https://arxiv.org/abs/2508.01907)

---

## Authors

- **Akash Venkateshwaran** — akash.venkateshwaran@gmail.com — [@akash-venkateshwaran](https://github.com/akash-venkateshwaran)

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
