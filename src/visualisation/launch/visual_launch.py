import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

from pathlib import Path as Path2

def find_src_dir(start_path: Path2) -> Path2:
    current = start_path.resolve()
    while current != current.parent:
        candidate = current / "src"
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError("Could not locate a 'src' directory.")


def generate_launch_description():
    src_dir = find_src_dir(Path2(__file__))
    layout_file = src_dir / 'visualisation' / 'visualisation'/ 'config'/ 'plotjuggler_conf.xml'
    rviz_config = src_dir / 'visualisation' / 'visualisation'/ 'config'/ 'rviz_config.rviz'


    if not os.path.isfile(layout_file):
        raise FileNotFoundError(f"Layout file not found: {layout_file}")
    
    if not os.path.isfile(rviz_config):
        raise FileNotFoundError(f"RViz configuration file not found: {rviz_config}")

    # Define the PlotJuggler node with the layout argument
    plotjuggler_node = Node(
        package='plotjuggler',
        executable='plotjuggler',  # Use 'PlotJuggler' as the executable name
        name='plotjuggler_with_layout',
        arguments=['--layout', str(layout_file)],  # Pass the layout file as an argument
        output='screen'
    )

    # Define the RViz node
    rviz_node = Node(
        package='visualisation',
        executable='visual_exe',
        name='rviz_node',
        output='screen'
    )

    rviz2_process = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', str(rviz_config)], 
        output='screen'
    )

    return LaunchDescription([
        plotjuggler_node,
        rviz_node,
        rviz2_process
    ])
