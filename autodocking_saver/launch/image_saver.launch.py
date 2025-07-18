from ament_index_python.resources import has_resource

from launch.actions import DeclareLaunchArgument
from launch.launch_description import LaunchDescription
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


pkg_share = get_package_share_directory('lifecycle_manager')
launch_file = os.path.join(pkg_share, 'launch', 'lifecycle_manager.launch.py')


def generate_launch_description():

    # parameter to choose if the images (processed by the CV algorithm) should be saved or not
    save_image_arg = DeclareLaunchArgument(
        'save_image',
        default_value='False',
        description='Enable or disable image saving'
    )
    
    # the node calling the CV algorithm
    image_saver_node = Node(
        package='autodocking_saver',
        executable='image_saver',
        name='image_saver_node',
        output='screen',
        parameters=[{
            'save_image': LaunchConfiguration('save_image')
        }],
    )

    '''return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_file),
            # Optionally, pass launch arguments as a dictionary
            # launch_arguments={'arg_name': 'value'}.items()
        ),
        # Other actions or nodes can be added here
        save_image_arg,
        image_saver_node,
    ])'''

    return LaunchDescription([
        save_image_arg,
        image_saver_node,
    ])
