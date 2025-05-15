from setuptools import find_packages, setup
import os

package_name = 'environment'

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/bathy_data', 
            [os.path.join('bathy_data', f) for f in os.listdir('bathy_data') if os.path.isfile(os.path.join('bathy_data', f))]),
        ('share/' + package_name + '/ship_lanes_data', 
            [os.path.join('ship_lanes_data', f) for f in os.listdir('ship_lanes_data') if os.path.isfile(os.path.join('ship_lanes_data', f))]),
        ('share/' + package_name + '/shoreline_data', 
            [os.path.join('shoreline_data', f) for f in os.listdir('shoreline_data') if os.path.isfile(os.path.join('shoreline_data', f))]),
        ('share/' + package_name + '/ssp_data', 
            [os.path.join('ssp_data', f) for f in os.listdir('ssp_data') if os.path.isfile(os.path.join('ssp_data', f))]),
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
            'envi_exe = environment.py_envi:main',
        ],
    },
)
