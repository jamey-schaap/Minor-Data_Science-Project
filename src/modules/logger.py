from src.configs.data import OUT_PATH
import os


def log_error(message: str) -> None:
    """
    Logs an error message.
    :param message: str, Message to be logged.
    """
    path = os.path.join(OUT_PATH, "error.log")
    with open(path, "a") as file:
        file.write(message + "\n")
