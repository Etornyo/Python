from passlib.context import CryptContext

password_context=CryptContext(schemes=["bcrypt"])


def hash_password(password: str)-> str:
    return password_context.hash(secret=password)


def match_password_hash(plain: str, hashed: str)->bool:
    return password_context.verify(secret=plain, hash=hashed)