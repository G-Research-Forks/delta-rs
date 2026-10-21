import pytest
from arro3.core import Array, DataType, Table

from deltalake import DeltaTable, write_deltalake
from deltalake._internal import _UNSIGNED_INTEGERS
from deltalake.exceptions import DeltaError


@pytest.mark.skipif(
    _UNSIGNED_INTEGERS, reason="unsigned integers are not cast to signed"
)
def test_unsafe_cast(tmp_path):
    tbl = Table.from_pydict({"foo": Array([1, 2, 3, 200], DataType.uint8())})

    with pytest.raises(
        DeltaError,
        match="Cast error: Failed to convert into Arrow schema: Cast error: Failed to cast foo from Int8 to UInt8: Can't cast value 200 to type Int8",
    ):
        write_deltalake(tmp_path, tbl)


def test_safe_cast(tmp_path):
    tbl = Table.from_pydict({"foo": Array([1, 2, 3, 4], DataType.uint8())})

    write_deltalake(tmp_path, tbl)


@pytest.mark.pyarrow
@pytest.mark.skipif(
    not _UNSIGNED_INTEGERS, reason="unsigned integer support not enabled"
)
def test_no_cast_with_unsigned_integers(tmp_path):
    tbl = Table.from_pydict({"foo": Array([1, 2, 3, 200], DataType.uint8())})

    write_deltalake(tmp_path, tbl)

    dt = DeltaTable(tmp_path)
    result = dt.to_pyarrow_table()
    assert str(result.schema.field("foo").type) == "uint8"
    assert result["foo"].to_pylist() == [1, 2, 3, 200]
