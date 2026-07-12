from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt
from cryptography.fernet import Fernet

from app.config.settings import get_settings

# ── 密码哈希 ──


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


# ── JWT Token ──

def create_access_token(user_id: int) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int | None:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        user_id = int(payload.get("sub", 0))
        return user_id if user_id else None
    except (JWTError, ValueError):
        return None


# ── 对称加密（用于数据库密码） ──

def encrypt_value(plain_text: str) -> str:
    settings = get_settings()
    f = Fernet(settings.encryption_key.encode())
    return f.encrypt(plain_text.encode()).decode()


def decrypt_value(encrypted_text: str) -> str:
    settings = get_settings()
    f = Fernet(settings.encryption_key.encode())
    return f.decrypt(encrypted_text.encode()).decode()