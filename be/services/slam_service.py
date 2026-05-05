# services/slam_service.py
from storage.storage_manager import StorageManager

storage = StorageManager()

async def process_slam_async(session_id: str, slam_engine):
    """비동기 SLAM 처리"""
    
    try:
        print(f"\n{'='*50}")
        print(f"SLAM 처리 시작: {session_id}")
        print(f"{'='*50}\n")
        
        await storage.update_status(session_id, "processing", progress=0)
        
        frames_data = await storage.load_session_data(session_id)
        print(f"로드된 프레임: {len(frames_data['poses'])}개")
        
        async def progress_callback(progress: float):
            await storage.update_progress(session_id, progress)
            print(f"진행률: {progress}%")
        
        map_data = await slam_engine.process(
            session_id=session_id,
            frames_data=frames_data,
            progress_callback=progress_callback
        )
        
        map_id = await storage.save_map(session_id, map_data, slam_engine)
        print(f"맵 저장 완료: {map_id}")
        
        await storage.update_status(
            session_id, 
            "completed", 
            map_id=map_id,
            progress=100
        )
        
        print(f"\n{'='*50}")
        print(f"SLAM 처리 완료: {session_id}")
        print(f"맵 ID: {map_id}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"SLAM 처리 실패: {session_id}")
        print(f"에러: {e}")
        print(f"{'='*50}\n")
        
        await storage.update_status(
            session_id, 
            "failed", 
            error=str(e)
        )

