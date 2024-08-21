from __future__ import annotations

import json
import os
from abc import abstractmethod
from typing import Dict, Generic, List, Optional, Type, TypeVar, Union
from typing_extensions import override, TypeAlias

from torch._inductor import config


if config.is_fbcode():
    from rfe.scubadata.scubadata_py3 import (  # type: ignore[import-not-found]
        Sample as Sample_,
    )

    Sample: TypeAlias = Sample_
else:
    Sample: TypeAlias = Type[object]  # type: ignore[misc,no-redef]


class RemoteCacheBackend:
    """
    A backend implementation for accessing a remote/distributed cache.  Only
    works with bytes in/out.  For structured data use a RemoteCache.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        pass

    @abstractmethod
    def put(self, key: str, data: bytes) -> None:
        pass


_T = TypeVar("_T")


class RemoteCacheSerde(Generic[_T]):
    @abstractmethod
    def encode(self, data: _T) -> bytes:
        pass

    @abstractmethod
    def decode(self, data: bytes) -> _T:
        pass


JsonDataTy = Optional[
    Union[int, float, str, bool, Dict[str, "JsonDataTy"], List["JsonDataTy"]]
]


class RemoteCacheJsonSerde(RemoteCacheSerde[JsonDataTy]):
    def encode(self, data: JsonDataTy) -> bytes:
        return bytes(json.dumps(data), "ascii")

    def decode(self, data: bytes) -> JsonDataTy:
        return json.loads(data)


class RemoteCache(Generic[_T]):
    def __init__(
        self, backend: RemoteCacheBackend, serde: RemoteCacheSerde[_T]
    ) -> None:
        self.backend = backend
        self.serde = serde

    def get(self, key: str) -> Optional[_T]:
        sample = self._create_sample()
        result = self._get(key, sample)
        self._log_sample(sample)
        return result

    def put(self, key: str, value: _T) -> None:
        sample = self._create_sample()
        self._put(key, value, sample)
        self._log_sample(sample)

    def _decode(self, data: bytes, sample: Optional[Sample]) -> _T:
        return self.serde.decode(data)

    def _encode(self, value: _T, sample: Optional[Sample]) -> bytes:
        return self.serde.encode(value)

    def _get(self, key: str, sample: Optional[Sample]) -> Optional[_T]:
        if data := self.backend.get(key):
            return self._decode(data, sample)
        return None

    def _put(self, key: str, value: _T, sample: Optional[Sample]) -> None:
        data = self._encode(value, sample)
        self.backend.put(key, data)

    def _create_sample(self) -> Optional[Sample]:
        return None

    def _log_sample(self, sample: Optional[Sample]) -> None:
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend):
    """
    A Redis implementation of a remote/distributed cache.
    """

    def __init__(self, cache_id: str) -> None:
        import redis

        self._key_fmt = f"pt2:{cache_id}:{{key}}"
        self._redis = redis.Redis(
            host=os.environ.get("TORCHINDUCTOR_REDIS_HOST", "localhost"),
            port=int(os.environ.get("TORCHINDUCTOR_REDIS_PORT", 6379)),
        )

    def __get_key(self, key: str) -> str:
        return self._key_fmt.format(key=key)

    @override
    def get(self, key: str) -> Optional[bytes]:
        value = self._redis.get(self.__get_key(key))
        # In theory redis.get() can return an Awaitable as well...
        assert value is None or isinstance(value, bytes)
        return value

    @override
    def put(self, key: str, data: bytes) -> None:
        self._redis.set(self.__get_key(key), data)
