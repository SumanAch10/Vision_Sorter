"""Generate a local secret without printing it or replacing an existing key."""

import os
import secrets
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / ".secrets"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / "api_key"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print("Existing API key retained at .secrets/api_key")
        return
    with os.fdopen(fd, "w") as output:
        output.write(secrets.token_urlsafe(32) + "\n")
    print("Created .secrets/api_key (not printed; ignored by Git and Docker context)")


if __name__ == "__main__":
    main()
