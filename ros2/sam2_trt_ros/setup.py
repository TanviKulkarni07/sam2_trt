from setuptools import setup

package_name = "sam2_trt_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sam2_trt",
    maintainer_email="you@example.com",
    description="ROS2 node for SAM2 TensorRT realtime inference from image topics",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "realtime_node = sam2_trt_ros.realtime_node:main",
        ],
    },
)
