#!/usr/bin/env python3

import argparse
from pathlib import Path
from urllib.parse import urlencode
import webbrowser


def find_dashboard():
    source_dashboard = Path(__file__).with_name('gui.html')
    if source_dashboard.is_file():
        return source_dashboard

    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError as exc:
        raise RuntimeError(
            'Could not find gui.html beside the launcher. Build and source the '
            'ROS workspace before using the installed launcher.'
        ) from exc

    installed_dashboard = (
        Path(get_package_share_directory('hitachi_navigation')) / 'web' / 'gui.html'
    )
    if not installed_dashboard.is_file():
        raise FileNotFoundError(f'Dashboard HTML not found: {installed_dashboard}')
    return installed_dashboard


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Open the Hitachi robot dashboard in your web browser.'
    )
    parser.add_argument(
        '--rosbridge',
        help='Rosbridge WebSocket URL (for example ws://192.168.1.20:9090).',
    )
    options = parser.parse_args(args)

    dashboard_url = find_dashboard().resolve().as_uri()
    if options.rosbridge:
        dashboard_url = f'{dashboard_url}?{urlencode({"rosbridge": options.rosbridge})}'

    if not webbrowser.open(dashboard_url, new=2):
        raise RuntimeError('Could not open a browser for the robot dashboard.')

    print(f'Opened robot dashboard: {dashboard_url}')


if __name__ == '__main__':
    main()
