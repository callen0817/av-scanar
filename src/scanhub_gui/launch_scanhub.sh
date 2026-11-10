#!/usr/bin/env bash
# scanHUB Launcher Script
set -e

# Source ROS 2 environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
elif [ -f "/opt/ros/foxy/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
fi

# Source workspace setup if built
if [ -f "/home/scanar/scanarMini/install/setup.bash" ]; then
    source /home/scanar/scanarMini/install/setup.bash
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "=================================================="
echo "Launching scanHUB OS v1.0 (scanR Platform)..."
echo "=================================================="

python3 "${SCRIPT_DIR}/scanhub_app.py" "$@"
