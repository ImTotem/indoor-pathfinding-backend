# slam_engines/__init__.py
from slam_interface.factory import SLAMEngineFactory

from .rtabmap import RTABMapEngine

# 팩토리에 엔진 등록
SLAMEngineFactory.register('rtabmap', RTABMapEngine)

__all__ = ['RTABMapEngine']

