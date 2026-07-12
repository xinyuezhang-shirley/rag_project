import base64
from datetime import datetime


def encode_cursor(created_at: datetime, id_: int) -> str:
    """将 (created_at, id) 编码为不透明的 base64 游标字符串。"""
    raw = f"{created_at.isoformat()}|{id_}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, int]:
    """解码游标字符串，返回 (created_at, id)。解码失败抛出 ValueError。"""
    raw = base64.urlsafe_b64decode(cursor.encode()).decode()
    created_at_str, id_str = raw.split("|", maxsplit=1)
    return datetime.fromisoformat(created_at_str), int(id_str)
