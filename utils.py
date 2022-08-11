from pathlib import Path


from pathlib import Path

def root() -> Path:
    return Path(__file__).parent.absolute()