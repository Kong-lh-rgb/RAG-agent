"""
Layer 4: Milvus 向量存储
========================
管理 Milvus 连接、Collection 创建、数据插入与索引构建。
"""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config.settings import settings


class MilvusStore:
    """Milvus 向量存储管理器。

    负责 Collection 的生命周期：连接 → 建表 → 插入 → 建索引 → 加载。
    """

    def __init__(
        self,
        host: str = settings.milvus_host,
        port: str = settings.milvus_port,
        collection_name: str = settings.collection_name,
        dim: int = settings.embedding_dim,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self._collection: Collection | None = None

    # ── 公开方法 ────────────────────────────────

    def connect(self) -> None:
        """建立 Milvus 连接。"""
        connections.connect("default", host=self.host, port=self.port)
        print(f"🔗 已连接 Milvus: {self.host}:{self.port}")

    def ensure_connected(self) -> None:
        """确保已连接，未连接则自动连接。"""
        try:
            connections.get_connection_addr("default")
        except Exception:
            self.connect()
            return
        # 连接地址存在但可能未真正建立
        if not connections.has_connection("default"):
            self.connect()

    def get_or_create_collection(self) -> Collection:
        """获取已有 Collection，不存在时新建。

        与 recreate_collection 不同，此方法不会删除已有数据。
        适用于 query 场景。
        """
        self.ensure_connected()

        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            self._collection.load()
            print(f"📦 已加载已有 Collection: {self.collection_name}")
        else:
            schema = self._build_schema()
            self._collection = Collection(name=self.collection_name, schema=schema)
            print(f"📦 已创建新 Collection: {self.collection_name}")

        return self._collection

    def recreate_collection(self) -> Collection:
        """重建 Collection（删旧建新）。"""
        self.ensure_connected()

        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"🗑️  已删除旧 Collection: {self.collection_name}")

        schema = self._build_schema()
        self._collection = Collection(name=self.collection_name, schema=schema)
        print(f"📦 已创建 Collection: {self.collection_name}")
        return self._collection

    def insert(self, chunks: list[str], embeddings: list[list[float]]) -> int:
        """插入文本和嵌入数据。

        Args:
            chunks:     文本块列表。
            embeddings: 对应的嵌入向量列表。

        Returns:
            插入的记录数。
        """
        if self._collection is None:
            self.recreate_collection()

        result = self._collection.insert([chunks, embeddings])
        self._collection.flush()
        print(f"💾 已插入 {result.insert_count} 条记录")
        return result.insert_count

    def build_index(self) -> None:
        """创建 IVF_FLAT 索引并加载到内存。"""
        if self._collection is None:
            raise RuntimeError("Collection 尚未创建，请先调用 recreate_collection()")

        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(field_name="embedding", index_params=index_params)
        print("🏗️  已创建索引: IVF_FLAT (L2)")

        self._collection.load()
        print("✅ Collection 已加载到内存，可供检索")

    def get_collection(self) -> Collection:
        """获取当前 Collection 实例。"""
        self.ensure_connected()
        if self._collection is None:
            self._collection = Collection(self.collection_name)
            self._collection.load()
        return self._collection

    # ── 私有方法 ────────────────────────────────

    def _build_schema(self) -> CollectionSchema:
        """构建 Collection Schema。"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        return CollectionSchema(fields, description="RAG document collection")
