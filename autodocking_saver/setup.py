from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autodocking_saver'

setup(
    name=package_name,
    version='0.0.0',
    #packages=find_packages(exclude=['test']),
    packages=[f'{package_name}', f'{package_name}.open_waters', f'{package_name}.open_waters.wasr', f'{package_name}.open_waters.weights', 'launch', 'resource'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    package_data={
        # Include all files in weights folder
        package_name: [f'{package_name}/open_waters/weights/*.pth'],
    },
    include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_saver = ' + package_name + '.image_saver_node:main',
        ],
    },
)
