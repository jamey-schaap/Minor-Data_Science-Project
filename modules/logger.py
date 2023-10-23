def log_error(message: str):
    with open("error.log", "a") as file:
        file.write(message + "\n")
