from app.core.components import minio
from czce_ai.document import DocTable
from czce_ai.document.reader import ReaderRegistry


def parse(doc_uuid: str, doc_bucket: str, doc_original_name: str) -> str:
    """解析并提取文件信息
    TODO 增加OCR功能解析文档中的图片

    Args:
        doc_uuid (str):文档uuid
        doc_bucket (str):文档存储桶
        doc_original_name (str):文档原始名称,包含后缀名

    Raises:
        ValueError:无法获取正确后缀名

    Returns:
        str:文档内容
    """
    try:
        doc_name, doc_type = doc_original_name.rsplit(".", 1)
    except ValueError:
        raise ValueError(f"无效的文件名格式:{doc_original_name}")

    file_obj = minio.get_object(doc_bucket, doc_uuid)
    doc_reader = ReaderRegistry.get_reader(doc_type.lower())
    if doc_type.lower() == "docx":
        # 创建阅读器实例
        doc = doc_reader.read(
            file=file_obj, doc_id=doc_uuid, doc_name=doc_name, in_order=True
        )
        # 处理表格内容
        article_content = ""
        for c in doc.content:
            # 如果是表格,需要将表格内容转换为字符串
            if isinstance(c, DocTable):
                article_content += c.content + "\n"
            else:
                article_content += c + "\n"
    else:
        # 创建阅读器实例
        doc = doc_reader.read(file=file_obj, doc_id=doc_uuid, doc_name=doc_name)
        article_content = doc.content

    return article_content