from pathlib import Any
from abc import abstractmethod
from craigfinder_typing import FileParser, FilePath


class FileParser:
    @abstractmethod
    def read_file(self, file_path: FilePath) -> Any:
        raise NotImplementedError("read_file method not implemented")

    @abstractmethod
    def write_line(self, file_path: FilePath) -> None:
        raise NotImplementedError("write_file method not implemented")
