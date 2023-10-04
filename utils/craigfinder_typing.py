from pathlib import Path
from typing import Any, Protocol, Union

FilePath = Union[str, Path]

var_sem_uso = "qualquer coisa aqui"


class FileParser(Protocol):
    def read_file(self, file_path: FilePath) -> Any:
        ...

    def write_line(self, file_path: FilePath) -> None:
        ...
