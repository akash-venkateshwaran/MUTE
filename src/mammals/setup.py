from setuptools import find_packages, setup
import os

package_name = 'mammals'

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                paths.append((os.path.join('share', package_name, path), [file_path]))
    return paths

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/mammals_launch.py']),  # Including launch files
    ] + package_files('case_studies')
      + package_files('data')
      + package_files('kde_models')
      + package_files('plot'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akashv22',
    maintainer_email='akashv22@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mammal_exe = mammals.py_mammal:main',
        ],
    },
)
