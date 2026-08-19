"""
StorageBackend abstraction: local filesystem for development,
S3 for production. Methods: save(), get_url(), delete(), exists()
"""
import os
import shutil
from abc import ABC, abstractmethod
from app.config import settings


class StorageBackend(ABC):
    @abstractmethod
    def save(self, local_tmp_path: str, dest_relative_path: str) -> str:
        ...

    @abstractmethod
    def get_url(self, dest_relative_path: str) -> str:
        ...

    @abstractmethod
    def delete(self, dest_relative_path: str) -> None:
        ...

    @abstractmethod
    def exists(self, dest_relative_path: str) -> bool:
        ...


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or settings.local_data_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _full_path(self, rel_path: str) -> str:
        return os.path.join(self.base_dir, rel_path)

    def save(self, local_tmp_path: str, dest_relative_path: str) -> str:
        full_path = self._full_path(dest_relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if os.path.abspath(local_tmp_path) != os.path.abspath(full_path):
            shutil.move(local_tmp_path, full_path)
        return full_path

    def get_url(self, dest_relative_path: str) -> str:
        return self._full_path(dest_relative_path)

    def delete(self, dest_relative_path: str) -> None:
        full_path = self._full_path(dest_relative_path)
        if os.path.exists(full_path):
            os.remove(full_path)

    def exists(self, dest_relative_path: str) -> bool:
        return os.path.exists(self._full_path(dest_relative_path))


class S3StorageBackend(StorageBackend):
    def __init__(self, bucket: str = None, region: str = None):
        import boto3
        self.bucket = bucket or settings.aws_s3_bucket
        self.client = boto3.client("s3", region_name=region or settings.aws_region)

    def save(self, local_tmp_path: str, dest_relative_path: str) -> str:
        self.client.upload_file(local_tmp_path, self.bucket, dest_relative_path)
        return dest_relative_path

    def get_url(self, dest_relative_path: str) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": dest_relative_path},
            ExpiresIn=3600,
        )

    def delete(self, dest_relative_path: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=dest_relative_path)

    def exists(self, dest_relative_path: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.head_object(Bucket=self.bucket, Key=dest_relative_path)
            return True
        except ClientError:
            return False


def get_storage_backend() -> StorageBackend:
    if settings.storage_backend == "s3":
        return S3StorageBackend()
    return LocalStorageBackend()
