def create_label(column_name: str, label: str) -> str:
    return label if label != "" else column_name
