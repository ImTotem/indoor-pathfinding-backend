from enum import StrEnum


class POISource(StrEnum):
    TRACK_LOCK = "track_lock"
    MANUAL = "manual"


class DetectionSource(StrEnum):
    ON_DEVICE = "on_device"
    SERVER = "server"  # 후속 스프린트 R-3용. 본 스프린트에서는 write 안 함.


class BuildState(StrEnum):
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    RUNNING = "running"
    BUILT = "built"
    FAILED = "failed"
