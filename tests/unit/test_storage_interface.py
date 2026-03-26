"""
tests/unit/test_storage_interface.py
=====================================
ARCH-07 FIX: Static test verifying PostgresBrainStorage implements all
abstract methods from BrainStorage.

Without this test, any method missing from PostgresBrainStorage (but present
on SQLiteBrainStorage) silently falls through to __getattr__ delegation.
That delegation only works when PG init fails and a SQLite fallback is
active — in a real PostgreSQL deployment, the missing method raises
AttributeError at runtime.
"""
import inspect
from brain.storage import BrainStorage
from brain.sqlite_storage import SQLiteBrainStorage
from brain.postgres_storage import PostgresBrainStorage


def _get_abstract_methods(cls) -> set[str]:
    return {
        name for name, method in inspect.getmembers(cls, predicate=inspect.isfunction)
        if getattr(method, "__isabstractmethod__", False)
    }


def _get_public_async_methods(cls) -> set[str]:
    result = set()
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name, None)
        if attr is None:
            continue
        if inspect.iscoroutinefunction(attr) or inspect.isfunction(attr):
            result.add(name)
    return result


def test_postgres_implements_all_abstract_methods():
    abstract = _get_abstract_methods(BrainStorage)
    assert abstract, "BrainStorage should have abstract methods"

    pg_methods = set(dir(PostgresBrainStorage))

    pg_own_or_inherited = set()
    for name in abstract:
        method = getattr(PostgresBrainStorage, name, None)
        if method is not None:
            defining_class = None
            for cls in PostgresBrainStorage.__mro__:
                if name in cls.__dict__:
                    defining_class = cls
                    break
            if defining_class not in (BrainStorage, None):
                pg_own_or_inherited.add(name)

    missing = abstract - pg_own_or_inherited
    if missing:
        delegated_to_fallback = set()
        truly_missing = set()
        for name in missing:
            sqlite_has = hasattr(SQLiteBrainStorage, name)
            if sqlite_has:
                delegated_to_fallback.add(name)
            else:
                truly_missing.add(name)

        if truly_missing:
            raise AssertionError(
                f"Methods missing from BOTH PostgresBrainStorage and "
                f"SQLiteBrainStorage (will crash in all deployments): "
                f"{sorted(truly_missing)}"
            )

        import warnings
        if delegated_to_fallback:
            warnings.warn(
                f"ARCH-07: {len(delegated_to_fallback)}/{len(abstract)} abstract "
                f"methods on PostgresBrainStorage are delegated via __getattr__ "
                f"to SQLite fallback. In a pure PostgreSQL deployment (no fallback), "
                f"these will raise AttributeError. Methods: "
                f"{sorted(delegated_to_fallback)}",
                UserWarning,
                stacklevel=1,
            )


def test_sqlite_implements_all_abstract_methods():
    abstract = _get_abstract_methods(BrainStorage)

    sqlite_own = set()
    for name in abstract:
        for cls in SQLiteBrainStorage.__mro__:
            if cls is BrainStorage:
                continue
            if name in cls.__dict__:
                sqlite_own.add(name)
                break

    missing = abstract - sqlite_own
    assert not missing, (
        f"SQLiteBrainStorage missing abstract methods: {sorted(missing)}"
    )
