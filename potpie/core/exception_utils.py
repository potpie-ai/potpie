"""Exception translation utilities for PotpieRuntime library.

This module provides utilities to translate FastAPI HTTPExceptions and other
app-specific exceptions to library-specific exceptions, ensuring the library
is decoupled from web framework dependencies.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar

from potpie.exceptions import (
    AgentNotFoundError,
    DatabaseError,
    Neo4jError,
    PotpieError,
    ProjectNotFoundError,
    UserNotFoundError,
)

if TYPE_CHECKING:
    from fastapi import HTTPException

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ExceptionTranslator:
    """Translates HTTP and app exceptions to library exceptions.

    This class handles the conversion of FastAPI HTTPExceptions and other
    application-specific exceptions to PotpieRuntime library exceptions.
    """

    # Map of HTTP status codes to exception types (for 404s, primarily)
    NOT_FOUND_KEYWORDS = ["not found", "does not exist", "no such", "unknown"]
    ACCESS_DENIED_KEYWORDS = [
        "access denied",
        "forbidden",
        "unauthorized",
        "permission",
    ]

    @classmethod
    def translate_http_exception(
        cls,
        http_exception: HTTPException,
        target_error_class: Type[PotpieError] = PotpieError,
        not_found_class: Optional[Type[PotpieError]] = None,
    ) -> PotpieError:
        """Translate an HTTPException to a library exception.

        Args:
            http_exception: The FastAPI HTTPException to translate
            target_error_class: Default exception class for non-404 errors
            not_found_class: Optional class for 404 errors

        Returns:
            Appropriate PotpieError subclass
        """
        status_code = http_exception.status_code
        detail = (
            str(http_exception.detail) if http_exception.detail else "Unknown error"
        )

        if status_code == 404:
            if not_found_class:
                return not_found_class(detail)
            return cls._infer_not_found_type(detail)

        if status_code in (401, 403):
            return PotpieError(f"Access denied: {detail}")

        if status_code == 500:
            return target_error_class(f"Internal error: {detail}")

        return target_error_class(detail)

    @classmethod
    def _infer_not_found_type(cls, detail: str) -> PotpieError:
        """Infer the correct not-found exception type from the detail message.

        Args:
            detail: Error detail message

        Returns:
            Appropriate not-found exception
        """
        detail_lower = detail.lower()

        if "project" in detail_lower:
            return ProjectNotFoundError(detail)
        if "agent" in detail_lower:
            return AgentNotFoundError(detail)
        if "user" in detail_lower:
            return UserNotFoundError(detail)

        return PotpieError(f"Not found: {detail}")

    @classmethod
    def translate_exception(
        cls,
        exception: Exception,
        target_error_class: Type[PotpieError] = PotpieError,
        not_found_class: Optional[Type[PotpieError]] = None,
    ) -> PotpieError:
        """Translate any exception to a library exception.

        Args:
            exception: Any exception to translate
            target_error_class: Default exception class
            not_found_class: Optional class for 404-like errors

        Returns:
            Appropriate PotpieError subclass
        """
        # Already a PotpieError - return as is
        if isinstance(exception, PotpieError):
            return exception

        # Check for HTTPException
        try:
            from fastapi import HTTPException

            if isinstance(exception, HTTPException):
                return cls.translate_http_exception(
                    exception, target_error_class, not_found_class
                )
        except ImportError:
            pass

        # Check for SQLAlchemy exceptions
        try:
            from sqlalchemy.exc import SQLAlchemyError

            if isinstance(exception, SQLAlchemyError):
                error = DatabaseError(f"Database error: {exception}")
                error.__cause__ = exception
                return error
        except ImportError:
            pass

        # Check for Neo4j exceptions
        try:
            from neo4j.exceptions import Neo4jError as Neo4jDriverError

            if isinstance(exception, Neo4jDriverError):
                error = Neo4jError(f"Neo4j error: {exception}")
                error.__cause__ = exception
                return error
        except ImportError:
            pass

        # Default: wrap in target error class
        return target_error_class(str(exception))


def translate_exceptions(
    target_error_class: Type[PotpieError] = PotpieError,
    not_found_class: Optional[Type[PotpieError]] = None,
    reraise_potpie_errors: bool = True,
):
    """Decorator that translates exceptions to library exceptions.

    Args:
        target_error_class: Default exception class for errors
        not_found_class: Optional class for 404-like errors
        reraise_potpie_errors: If True, PotpieError subclasses pass through unchanged

    Returns:
        Decorated function that translates exceptions

    Example:
        @translate_exceptions(ProjectError, ProjectNotFoundError)
        async def get_project(project_id: str):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if reraise_potpie_errors and isinstance(e, PotpieError):
                    raise
                translated = ExceptionTranslator.translate_exception(
                    e, target_error_class, not_found_class
                )
                raise translated from e

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if reraise_potpie_errors and isinstance(e, PotpieError):
                    raise
                translated = ExceptionTranslator.translate_exception(
                    e, target_error_class, not_found_class
                )
                raise translated from e

        if _is_async(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def _is_async(func: Callable) -> bool:
    """Check if a function is async."""
    import asyncio

    return asyncio.iscoroutinefunction(func)


def wrap_http_exception(
    func: Callable[..., T],
    target_error_class: Type[PotpieError] = PotpieError,
    not_found_class: Optional[Type[PotpieError]] = None,
) -> Callable[..., T]:
    """Wrap a function to translate HTTPExceptions.

    This is a simpler alternative to the decorator for wrapping existing functions.

    Args:
        func: Function to wrap
        target_error_class: Default exception class
        not_found_class: Optional class for 404 errors

    Returns:
        Wrapped function
    """
    return translate_exceptions(target_error_class, not_found_class)(func)


class ExceptionContext:
    """Context manager for exception translation.

    Example:
        async with ExceptionContext(ProjectError, ProjectNotFoundError):
            result = await project_service.get(id)
    """

    def __init__(
        self,
        target_error_class: Type[PotpieError] = PotpieError,
        not_found_class: Optional[Type[PotpieError]] = None,
        reraise_potpie_errors: bool = True,
    ):
        self.target_error_class = target_error_class
        self.not_found_class = not_found_class
        self.reraise_potpie_errors = reraise_potpie_errors

    async def __aenter__(self) -> ExceptionContext:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is None:
            return False

        if self.reraise_potpie_errors and isinstance(exc_val, PotpieError):
            return False

        translated = ExceptionTranslator.translate_exception(
            exc_val, self.target_error_class, self.not_found_class
        )
        raise translated from exc_val

    def __enter__(self) -> ExceptionContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is None:
            return False

        if self.reraise_potpie_errors and isinstance(exc_val, PotpieError):
            return False

        translated = ExceptionTranslator.translate_exception(
            exc_val, self.target_error_class, self.not_found_class
        )
        raise translated from exc_val
