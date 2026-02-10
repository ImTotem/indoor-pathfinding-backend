# utils/logger.py
import logging
import sys
from datetime import datetime
from pathlib import Path
from config.settings import settings

def setup_logger(name: str = "slam_backend") -> logging.Logger:
    """로거 설정"""
    
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 있으면 스킵
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    
    # 포맷터
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    log_dir = settings.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# 전역 로거 인스턴스
logger = setup_logger()


def log_request(method: str, path: str, body: dict = None):
    """요청 로깅"""
    logger.info(f"→ {method} {path}")
    if body:
        # 이미지 데이터는 길이만 표시
        safe_body = _sanitize_body(body)
        logger.debug(f"  Body: {safe_body}")


def log_response(status: int, detail: str = None):
    """응답 로깅"""
    if status >= 500:
        logger.error(f"← {status} {detail}")
    elif status >= 400:
        logger.warning(f"← {status} {detail}")
    else:
        logger.info(f"← {status} {detail or 'OK'}")


def log_error(error: Exception, context: str = None):
    """에러 로깅 (traceback 포함)"""
    if context:
        logger.error(f"[{context}] {type(error).__name__}: {error}", exc_info=True)
    else:
        logger.error(f"{type(error).__name__}: {error}", exc_info=True)


def _sanitize_body(body: dict) -> dict:
    """로깅용 body 정제 (이미지 데이터 제거)"""
    if not isinstance(body, dict):
        return body
    
    sanitized = {}
    for key, value in body.items():
        if key in ('image', 'imageBase64', 'image_base64'):
            sanitized[key] = f"<base64, {len(value)} chars>" if isinstance(value, str) else "<binary>"
        elif key == 'frames' and isinstance(value, list):
            sanitized[key] = f"<{len(value)} frames>"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_body(value)
        else:
            sanitized[key] = value
    
    return sanitized
