from slam_interface.factory import SLAMEngineFactory

from .rtabmap import RTABMapEngine

SLAMEngineFactory.register("rtabmap", RTABMapEngine)

__all__ = ["RTABMapEngine"]
