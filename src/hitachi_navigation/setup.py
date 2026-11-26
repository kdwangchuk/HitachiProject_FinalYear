from setuptools import find_packages, setup

package_name = 'hitachi_navigation'

setup(
    name=package_name,
    version='0.0.1',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
        (
            'share/' + package_name + '/config',
            ['config/aruco_locations.csv']
        ),
        (
            'share/' + package_name + '/web',
            ['hitachi_navigation/gui.html']
        ),
    ],

    install_requires=[
        'setuptools'
    ],

    zip_safe=True,

    maintainer='orangepi',
    maintainer_email='orangepi@example.com',

    description='Hardware navigation system for TurtleBot3',

    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'aruco_detector = hitachi_navigation.aruco_detector:main',
	        'yellow_line_detector = hitachi_navigation.yellow_line_detector:main',
            'line_follower_test = hitachi_navigation.line_follower_test:main',
            'robot_localizer = hitachi_navigation.robot_localizer:main',
            'localization_validator = hitachi_navigation.localization_validator:main',
            'open_dashboard = hitachi_navigation.gui_launcher:main',
        ],
    },
)
