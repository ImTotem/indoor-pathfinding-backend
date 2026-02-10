# slam_interface/factory.py
from .base import SLAMEngineBase

class SLAMEngineFactory:
    """SLAM 엔진 팩토리"""
    
    _engines = {}
    
    @classmethod
    def register(cls, name: str, engine_class):
        """엔진 등록"""
        cls._engines[name] = engine_class
    
    @classmethod
    def create(cls, engine_type: str) -> SLAMEngineBase:
        """엔진 생성"""
        engine_class = cls._engines.get(engine_type)
        
        if engine_class is None:
            raise ValueError(f"Unknown SLAM engine type: '{engine_type}'. Available engines: {cls.list_engines()}")
        
        return engine_class()
    
    @classmethod
    def list_engines(cls):
        """사용 가능한 엔진 목록"""
        return list(cls._engines.keys())

