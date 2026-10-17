# MUTE — Minimizing Underwater Radiated Noise (MUTE-DSS)

MUTE-DSS is a digital-twin-based decision support system for minimizing underwater radiated noise (URN) during ship voyage planning. The project combines near-field acoustic modeling, 3D ray tracing for far-field propagation, whale distribution tracking, and a two-stage optimization framework built on ROS2 for modular, distributed, and real-time operation.

This repository implements a decision-support pipeline that plans collision-free vessel routes and adaptive speed profiles to reduce cumulative underwater noise exposure to marine mammals while respecting voyage constraints and operational feasibility.

## Abstract

We present a novel MUTE-DSS, a digital-twin-based decision support system for minimizing underwater radiated noise (URN) during ship voyage planning. The system is built on the Robot Operating System (ROS2), an open-source middleware framework designed for modular, distributed, and real-time communication between software components. The framework integrates near-field acoustic modeling using a semi-empirical spectrum with 3D ray tracing for far-field propagation, enabling real-time ship noise prediction and whale distribution tracking. The proposed DSS performs a two-stage optimization pipeline: Batch Informed Trees (BIT*) for collision-free ship routing and a genetic algorithm for adaptive ship speed profiling under voyage constraints that minimizes cumulative URN exposure to marine mammals. The effectiveness of MUTE-DSS is demonstrated through case studies of ships operating between the Strait of Georgia and the Strait of Juan de Fuca, comparing optimized voyages against baseline trajectories derived from automatic identification system (AIS) data. Results show substantial reductions in noise exposure levels, up to 7.14 dB, corresponding to an approximately 80.68% reduction in a simplified scenario, and an average 4.90 dB reduction, corresponding to an approximately 67.6% reduction in a more realistic dynamic setting. These results illustrate the adaptability and practical utility of the proposed decision support system.

## Project Goals

- Reduce underwater radiated noise during ship voyage planning
- Enable real-time noise prediction using digital-twin-style environmental modeling
- Combine routing and speed optimization into a practical decision-support workflow
- Support marine mammal conservation by minimizing noise exposure in sensitive regions
- Provide a modular ROS2-based architecture that can be extended and adapted

## Key Features

- ROS2-based modular software architecture
- Semi-empirical near-field acoustic model
- 3D ray-tracing-based far-field propagation model
- Whale distribution and marine mammal exposure tracking
- Batch Informed Trees (BIT*) for route planning
- Genetic algorithm for speed profile optimization
- AIS-driven baseline comparison
- Output visualizations for routes, noise exposure, and performance metrics

## Repository Structure

A typical repository structure for this project is:

```text
MUTE/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── package.xml
├── src/
│   ├── mute_dss/
│   │   ├── acoustic_model/
│   │   ├── ray_tracer/
│   │   ├── whale_tracker/
│   │   ├── route_planner/
│   │   ├── speed_optimizer/
│   │   ├── data_ingest/
│   │   ├── visualization/
│   │   └── utils/
│   └── ...
├── scripts/
│   ├── run_optimization.py
│   ├── preprocess_data.py
│   └── plot_results.py
├── config/
│   ├── vessel.yaml
│   ├── optimizer.yaml
│   ├── environment.yaml
│   └── routes.yaml
├── data/
│   ├── ais_example.csv
│   ├── bathymetry/
│   ├── whale_distribution/
│   └── sample_outputs/
├── results/
│   └── example_run/
├── tests/
│   └── ...
├── .github/
│   └── workflows/
└── docs/
    └── ...
```

If your repo layout differs, adapt the commands below to the actual package names and paths.

## System Requirements

### Recommended Environment

- Ubuntu 22.04 LTS or other ROS2-supported Linux distribution
- ROS2 Humble (recommended)
- Python 3.10+
- Git
- CMake / colcon
- pip for Python dependencies

### Python Dependencies

Install Python dependencies from the project requirements file:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the repo uses ROS2 Python packages or custom libraries, install them according to your workspace setup.

## ROS2 Setup Instructions

If this project is implemented as a ROS2 workspace, follow the steps below.

### 1. Install ROS2

Follow the official ROS2 installation guide for your OS:

- ROS2 documentation: https://docs.ros.org

For Ubuntu 22.04, a common installation is ROS2 Humble:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install -y curl
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-humble-desktop
```

Then source ROS2:

```bash
source /opt/ros/humble/setup.bash
```

### 2. Create a ROS2 Workspace

```bash
mkdir -p ~/mute_ws/src
cd ~/mute_ws
```

### 3. Clone the Repository

```bash
cd ~/mute_ws/src
git clone https://github.com/akash-venkateshwaran/MUTE.git
```

### 4. Install System Dependencies

```bash
cd ~/mute_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 5. Build the Workspace

```bash
cd ~/mute_ws
colcon build
source install/setup.bash
```

### 6. Verify the Workspace

```bash
ros2 pkg list | grep mute
```

If the package names are correct, you are ready to run the project.

## Running the Project

Common ROS2 workflow:

### Build

```bash
cd ~/mute_ws
colcon build
source install/setup.bash
```

### Run a Launch File

```bash
ros2 launch <package_name> <launch_file>.launch.py
```

Example:

```bash
ros2 launch mute_dss core.launch.py
```

If your launch file uses parameters, pass them explicitly:

```bash
ros2 launch mute_dss core.launch.py ais_csv:=data/ais_example.csv vessel_config:=config/vessel.yaml
```

### Run a Node Directly

```bash
ros2 run <package_name> <node_executable>
```

### Inspect Topics

```bash
ros2 topic list
ros2 topic echo /some_topic
```

### Inspect Nodes

```bash
ros2 node list
ros2 node info /some_node
```

### View Parameters

```bash
ros2 param list
ros2 param get /some_node some_parameter
```

### Start RViz

```bash
rviz2
```

## Typical Workflow

1. Prepare AIS and environment input files
2. Configure vessel and optimizer parameters
3. Launch the ROS2 system
4. Run the route planner
5. Run the speed optimization step
6. Produce output route and exposure metrics
7. Visualize results and compare with baseline trajectory

## Input Data

The project usually expects the following types of inputs:

- AIS data (e.g., vessel positions, timestamps, speed, heading)
- Bathymetry or environmental fields
- Vessel configuration (length, draft, speed-power model, constraints)
- Whale distribution or marine mammal density data
- Voyage constraints and operational limits

Typical file examples:

```text
data/
├── ais_example.csv
├── vessel.yaml
├── bathymetry/
├── whale_distribution/
└── config/
```

## Output Data

The project may generate:

- Optimized route coordinates
- Adaptive speed profile over time
- Noise maps / exposure maps
- Comparison between optimized and baseline trajectories
- Summary metrics (dB reduction, cumulative exposure levels, etc.)

Outputs are commonly stored in:

```text
results/
├── route.csv
├── speed_profile.csv
├── noise_map.png
├── comparison_plot.png
└── summary.json
```

## Configuration

Project behavior can usually be customized through config files such as:

- `config/vessel.yaml`
- `config/optimizer.yaml`
- `config/environment.yaml`
- `config/routes.yaml`

Before running experiments, verify:

- vessel parameters are correct
- environmental inputs exist
- output directory is writable
- route constraints are valid

## Development Workflow

### Recommended Branching Model

- `main` — stable production-ready branch
- `develop` — active development
- `feature/*` — new functionality
- `bugfix/*` — fixes and patches

### Recommended Commit Style

Use clear, descriptive commit messages such as:

```bash
git checkout -b feature/route-optimizer
git add .
git commit -m "Add BIT* route planner integration"
```

### Testing

If the project uses Python tests, run:

```bash
pytest -q
```

If ROS2 packages include unit tests or launch-based tests, use the appropriate ROS2 test runner:

```bash
colcon test
colcon test-result --all
```

### Linting

Common standards:

```bash
black .
flake8 .
```

## Best Practices for a Research Repository

- Keep README up to date
- Include install and usage instructions
- Document dependencies and assumptions
- Use example data and example commands
- Explain expected outputs and data formats
- Provide test instructions
- Include a citation for the associated paper
- Document limitations and known issues

## Citation

If you use this project or its methods in research, please cite the associated paper:

- Article: https://www.sciencedirect.com/science/article/pii/S0029801825030264

Suggested BibTeX:

```bibtex
@article{venkateshwaran2025mute,
  title = {MUTE-DSS: A digital-twin-based decision support system for minimizing underwater radiated noise during ship voyage planning},
  author = {Venkateshwaran, Akash and others},
  journal = {Journal Name},
  year = {2025},
  url = {https://www.sciencedirect.com/science/article/pii/S0029801825030264}
}
```

## Authors

- Akash Venkateshwaran — akash.venkateshwaran@gmail.com — [@akash-venkateshwaran](https://github.com/akash-venkateshwaran)

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, usage support, or collaboration inquiries, contact:

- Akash Venkateshwaran
- Email: akash.venkateshwaran@gmail.com

## Notes

This README provides a standard research-repository structure and usage guidance. If the repository contains real package names, launch files, or scripts, replace the example commands above with the actual commands from the repo before running the project.

If you want, I can also generate:
- a shorter project README version
- a more academic paper-style README
- a GitHub-ready README tailored exactly to your repo structure after you share the actual project folders and package names
