"""Product types: the kind of task a pipeline performs.

The built-in products are members of :class:`ProductType`. Downstream harnesses
can add their own with :func:`register_product_type`; a registered name is
accepted anywhere a ``ProductType`` field is declared and behaves like an enum
member for the common operations (``==`` against the string, ``.value``,
``.name``, ``str()``).
"""

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BeforeValidator, GetCoreSchemaHandler
from pydantic_core import core_schema


class ProductType(StrEnum):
    PARSE = "parse"
    LAYOUT_DETECTION = "layout_detection"
    EXTRACT = "extract"


class ExtensionProductType(str):
    """A product type registered by an extension.

    A ``str`` subclass so it compares equal to its name and serializes as a
    plain string, with ``value``/``name`` so code written against
    :class:`ProductType` keeps working.
    """

    __slots__ = ()

    @property
    def value(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return str(self).upper()

    def __repr__(self) -> str:
        return f"ExtensionProductType({str(self)!r})"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        # Validate like a string, wrap into this type, serialize as the plain string.
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
            serialization=core_schema.to_string_ser_schema(),
        )


_EXTENSION_PRODUCT_TYPES: dict[str, ExtensionProductType] = {}


def register_product_type(name: str) -> ExtensionProductType:
    """Register an additional product type name.

    Idempotent. Returns the object to use in pipeline specs and comparisons.
    Raises ``ValueError`` if ``name`` collides with a built-in product.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Product type name must be a non-empty string")
    if name in ProductType.__members__.values():
        raise ValueError(f"{name!r} is a built-in ProductType")
    existing = _EXTENSION_PRODUCT_TYPES.get(name)
    if existing is not None:
        return existing
    product = ExtensionProductType(name)
    _EXTENSION_PRODUCT_TYPES[name] = product
    return product


def registered_product_types() -> list[str]:
    """Every accepted product type name: built-ins first, then extensions."""
    return [member.value for member in ProductType] + list(_EXTENSION_PRODUCT_TYPES)


def coerce_product_type(value: Any) -> ProductType | ExtensionProductType:
    """Validate a product type from an enum member, extension object or string."""
    if isinstance(value, ProductType | ExtensionProductType):
        return value
    if isinstance(value, str):
        try:
            return ProductType(value)
        except ValueError:
            pass
        extension = _EXTENSION_PRODUCT_TYPES.get(value)
        if extension is not None:
            return extension
        raise ValueError(f"Unknown product type {value!r}; known: {registered_product_types()}")
    raise TypeError(f"Product type must be a string, got {type(value).__name__}")


ProductTypeName = Annotated[ProductType | ExtensionProductType, BeforeValidator(coerce_product_type)]
"""Field annotation accepting built-in and registered product types."""
