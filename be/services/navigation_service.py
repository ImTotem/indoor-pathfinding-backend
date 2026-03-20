import math
import uuid
from typing import Dict, List, Optional


def linear_path(start: List[float], end: List[float], steps: int = 40) -> List[List[float]]:
    if steps < 2:
        return [start, end]
    path = []
    for i in range(steps):
        t = i / (steps - 1)
        path.append([
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
            start[2] + (end[2] - start[2]) * t,
        ])
    return path


class NavigationSession:
    def __init__(self, session_id: str, map_id: str, start: List[float], goal: List[float], path: List[List[float]]):
        self.session_id = session_id
        self.map_id = map_id
        self.start = start
        self.goal = goal
        self.path = path
        self.current_index = 0
        self.status = "running"

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "map_id": self.map_id,
            "start": self.start,
            "goal": self.goal,
            "path": self.path,
            "current_index": self.current_index,
            "status": self.status,
        }


class NavigationService:
    def __init__(self):
        self.sessions: Dict[str, NavigationSession] = {}

    def _distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _find_closest_index(self, session: NavigationSession, position: List[float]) -> int:
        closest_index = session.current_index
        closest_dist = float('inf')
        window_start = max(0, session.current_index - 5)
        window_end = min(len(session.path), session.current_index + 10)

        for i in range(window_start, window_end):
            dist = self._distance(position, session.path[i])
            if dist < closest_dist:
                closest_dist = dist
                closest_index = i

        # full-scan fallback (sparse map or first frame)
        if closest_dist > 10.0:
            for i, p in enumerate(session.path):
                dist = self._distance(position, p)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_index = i

        return closest_index

    def start_session(self, map_id: str, start: List[float], goal: List[float]) -> NavigationSession:
        session_id = str(uuid.uuid4())
        path = linear_path(start, goal, steps=40)
        session = NavigationSession(session_id=session_id, map_id=map_id, start=start, goal=goal, path=path)
        self.sessions[session_id] = session
        return session

    def close_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def update_position(self, session_id: str, position: List[float]) -> Dict:
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")

        session = self.sessions[session_id]
        if session.status != "running":
            return {"status": session.status}

        closest_idx = self._find_closest_index(session, position)
        session.current_index = closest_idx

        path_point = session.path[closest_idx]
        deviation_distance = self._distance(position, path_point)

        on_path = deviation_distance <= 3.0

        if self._distance(position, session.goal) <= 1.5:
            session.status = "completed"
            arrival = True
        else:
            arrival = False

        # 재경로: 현재 위치가 경로에서 많이 벗어났을 때
        replan_needed = False
        new_path = None
        if not on_path and deviation_distance > 5.0:
            replan_needed = True
            new_path = linear_path(position, session.goal, steps=max(20, len(session.path) - closest_idx))
            session.path = [position] + new_path
            session.current_index = 0

        remaining_path = session.path[session.current_index + 1 :]

        return {
            "session_id": session_id,
            "position": position,
            "path_index": session.current_index,
            "on_path": on_path,
            "deviation_distance": deviation_distance,
            "next_instruction": "keep going" if not arrival else "arrived",
            "remaining_path": remaining_path,
            "status": session.status,
            "arrival": arrival,
            "replan": replan_needed,
            "new_path": new_path,
        }
