import csv
import logging
import os
from datetime import datetime

_LOG = logging.getLogger(__name__)


class Logger:
    def __init__(self, directory=".", filename_prefix="log", filename=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = directory
        if filename is None:
            self.filename = f"{filename_prefix}_{timestamp}.csv"
        else:
            self.filename = f"{filename}.csv"
        self.full_path = os.path.join(directory, self.filename)
        os.makedirs(os.path.dirname(self.full_path), exist_ok=True)
        self.headers_written = False
        self.headers = []
        _LOG.info(f"Logging to {os.path.abspath(self.full_path)}")

    def append_row(self, data: dict):
        data["timestamp"] = datetime.now().isoformat()
        if not self.headers_written:
            self.headers = list(data.keys())
            with open(self.full_path, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.headers)
                writer.writeheader()
                writer.writerow(data)
            self.headers_written = True
        else:
            if set(data.keys()) != set(self.headers):
                raise ValueError(
                    "Keys of the new data must match the original headers. The following keys are missing: "
                    + str(set(self.headers) - set(data.keys()))
                )
            with open(self.full_path, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.headers)
                writer.writerow(data)

    def load_existing_data(self):
        if not os.path.exists(self.full_path):
            return []
        with open(self.full_path, newline="") as file:
            reader = csv.DictReader(file)
            self.headers_written = True
            self.headers = list(reader.fieldnames)
            return [row for row in reader]
