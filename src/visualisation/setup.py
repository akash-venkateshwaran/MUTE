from setuptools import find_packages, setup
import os

package_name = 'visualisation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name + '/launch', ['launch/visual_launch.py']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/mesh', 
            [os.path.join('mesh', f) for f in os.listdir('mesh') if os.path.isfile(os.path.join('mesh', f))]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akashv22',
    maintainer_email='akashv22@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_exe = visualisation.rviz_nodes:main',
        ],
    },
)
