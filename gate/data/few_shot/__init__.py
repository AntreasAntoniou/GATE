def bytes_to_string(x):
    return (
        x.decode("utf-8").lower() if isinstance(x, bytes) else str(x).lower()
    )
