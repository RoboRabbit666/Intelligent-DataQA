# coding: utf-8
import os
from datetime import timedelta
from io import BytesIO
from typing import BinaryIO, Dict, List, Optional, Union

from minio import Minio
from minio.error import S3Error

from czce_ai.utils.log import logger
from czce_ai.vectordb.exceptions import MinioOperationError


class MinioClient:
    """MinIO 工具类封装

    功能包括:
    - 文件上传/下载/删除
    - 生成预签名URL
    - 存储桶管理
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ):
        """初始化MinIO客户端

        Args:
            endpoint (str): MinIO服务器地址
            access_key (str): 账户。
            secret_key (str): #这些参数在实际使用中通常从配置文件或环境变量中获取,这里为了简化示例直接作为参数传递
            secure (bool, optional): 是否启用HTTPS. Defaults to False.
        """
        try:
            self._client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )
        except Exception:
            logger.exception("Fail to connect {}".format(endpoint))

    def bucket_exists(self, bucket_name: str) -> bool:
        """检查存储桶是否存在"""
        try:
            return self._client.bucket_exists(bucket_name)
        except S3Error as e:
            raise MinioOperationError(
                f"检查存储桶失败:{e}", operation="bucket_exists", bucket=bucket_name
            )

    def make_bucket(self, bucket_name: str) -> None:
        """创建存储桶"""
        try:
            if not self.bucket_exists(bucket_name):
                self._client.make_bucket(bucket_name)
        except S3Error as e:
            raise MinioOperationError(f"创建存储桶失败:{e}")

    def list_buckets(self) -> List[str]:
        """列出所有存储桶"""
        try:
            return [bucket.name for bucket in self._client.list_buckets()]
        except S3Error as e:
            raise MinioOperationError(f"列出存储桶失败:{e}")

    def remove_bucket(self, bucket_name: str) -> None:
        """删除存储桶"""
        try:
            if self._client.bucket_exists(bucket_name):
                self._client.remove_bucket(bucket_name)
        except S3Error as e:
            raise MinioOperationError(f"删除存储桶失败:{e}")

    def upload_data(
        self,
        bucket_name: str,
        object_name: str,
        data: Union[bytes, BinaryIO],
        length: int,
        metadata: Optional[Dict] = None,
    ) -> None:
        """上传bytes数据到MinIO"""
        try:
            self.make_bucket(bucket_name)
            if isinstance(data, bytes):
                length = len(data)
                data = BytesIO(data)
                data.seek(0)

            self._client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=data,
                length=length,
                metadata=metadata,
            )
        except (S3Error, Exception) as e:
            raise MinioOperationError(f"上传数据失败:{e}", bucket=bucket_name)

    def get_object(self, bucket_name: str, object_name: str) -> Optional[bytes]:
        """获取对象"""
        response = None
        try:
            response = self._client.get_object(bucket_name, object_name)
            return response.read()
        except (S3Error, Exception) as e:
            raise MinioOperationError(f"{object_name}数据读取失败:{e}", bucket=bucket_name)
        finally:
            if response:
                response.close()
                response.release_conn()

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """上传文件到MinIO

        Args:
            bucket_name (str): 存储桶名称
            object_name (str): 对象名称(包含路径)
            file_path (str): 本地文件路径
            metadata (Optional[Dict], optional): 元数据字典. Defaults to None.
        """
        try:
            if not os.path.exists(file_path):
                raise FileExistsError(f"本地文件不存在:{file_path}")
            # 若bucket不存在,自动创建
            self.make_bucket(bucket_name)
            self._client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=file_path,
                metadata=metadata,
            )
        except (S3Error, Exception) as e:
            raise MinioOperationError(f"上传文件失败: {e}", bucket=bucket_name)

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        """删除MinIO中的对象"""
        try:
            self._client.remove_object(bucket_name=bucket_name, object_name=object_name)
        except S3Error as e:
            raise MinioOperationError(f"删除对象失败:{e}", bucket_name=bucket_name)

    def presigned_get_object(
        self, bucket_name: str, object_name: str, expires: timedelta = timedelta(days=3)
    ) -> str:
        """生成预签名的下载链接

        Args:
            bucket_name (str): 存储桶名称
            object_name (str): 对象名称
            expires (timedelta, optional): URL过期时间. Defaults to timedelta(days=3).

        Raises:
            MinioOperationError: _description_

        Returns:
            str: _description_
        """
        try:
            return self._client.presigned_get_object(
                bucket_name=bucket_name, object_name=object_name, expires=expires
            )
        except S3Error as e:
            raise MinioOperationError(
                f"生成预签名URL失败: {e}", bucket_name=bucket_name
            )