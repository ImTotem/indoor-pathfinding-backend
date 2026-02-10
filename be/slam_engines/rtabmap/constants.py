# slam_engines/rtabmap/constants.py

SLAM_TIMEOUT_SECONDS = 600

DATABASE_FILENAME = "rtabmap.db"

DEFAULT_PARAMS = {
    "Mem/IncrementalMemory": "true",
    "Mem/InitWMWithAllNodes": "false",
    "Mem/DepthAsMask": "false",

    "Kp/DetectorStrategy": "6",
    "Vis/FeatureType": "6",
    "BRIEF/Bytes": "64",
    "Kp/MaxFeatures": "1000",
    "Vis/MinInliers": "3",
    "Vis/DepthAsMask": "false",
    "Vis/MaxDepth": "4.0",
    "Vis/MinDepth": "0.3",

    "OdomF2M/ValidDepthRatio": "0.1",

    "Rtabmap/LoopThr": "0.01",
    "Rtabmap/LoopRatio": "0",
    "Rtabmap/ImagesAlreadyRectified": "true",

    "RGBD/LinearUpdate": "0.02",
    "RGBD/AngularUpdate": "0.03",
    "RGBD/OptimizeFromGraphEnd": "false",
    "Optimizer/Strategy": "1",

    "Grid/3D": "true",
    "Grid/RayTracing": "true",
    "Grid/CellSize": "0.05",
    "Grid/RangeMax": "4.0",
    "Grid/RangeMin": "0.3",
    "Grid/DepthDecimation": "4",
    "Grid/NoiseFilteringRadius": "0.05",
    "Grid/NoiseFilteringMinNeighbors": "5",
    "Grid/MinClusterSize": "10",
    "Grid/PreVoxelFiltering": "true",
}
