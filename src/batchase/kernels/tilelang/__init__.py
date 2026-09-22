def is_tilelang_available() -> bool:
    try:
        import tilelang  # noqa: F401
        return True
    except ImportError:
        return False

__all__ = ["is_tilelang_available"]
