import logging
from pathlib import Path

logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)


class IO:
    @classmethod
    def read_file(self, dir: Path) -> bytes:
        try:
            file = open(dir, "rb").read()
            return file
        except Exception as e:
            logging.error("Failed with exception:", str(e))
            raise e

    @classmethod
    def write_file(self, file: str, dir: Path):
        try:
            with open(dir, "w") as fout:
                fout.write(file)
        except Exception as e:
            logging.error("Failed with exception:", str(e))
            raise e
