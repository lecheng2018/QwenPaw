# -*- coding: utf-8 -*-
"""Tests for provider model ordering."""

from __future__ import annotations

import pytest

from qwenpaw.providers.provider import (
    ModelInfo,
    Provider,
)


class FakeProvider(Provider):
    """Minimal concrete provider for ordering tests."""

    async def check_connection(self, timeout: float = 5) -> tuple[bool, str]:
        return True, ""

    async def fetch_models(self, timeout: float = 5) -> list[ModelInfo]:
        return []

    async def check_model_connection(
        self,
        model_id: str,
        timeout: float = 5,
    ) -> tuple[bool, str]:
        return True, ""

    def get_chat_model_instance(self, model_id: str):
        return None


def make_provider() -> FakeProvider:
    return FakeProvider(
        id="fake",
        name="Fake",
        models=[
            ModelInfo(id="a", name="A", sort_order=2),
            ModelInfo(id="b", name="B", sort_order=0),
            ModelInfo(id="c", name="C", sort_order=1),
        ],
        extra_models=[
            ModelInfo(id="x", name="X", sort_order=1),
            ModelInfo(id="y", name="Y", sort_order=0),
        ],
    )


@pytest.mark.asyncio
async def test_get_info_sorts_by_sort_order_then_original_index() -> None:
    provider = make_provider()
    info = await provider.get_info()

    model_ids = [m.id for m in info.models]
    extra_ids = [m.id for m in info.extra_models]

    assert model_ids == ["b", "c", "a"]
    assert extra_ids == ["y", "x"]


@pytest.mark.asyncio
async def test_add_model_places_new_model_at_end() -> None:
    provider = make_provider()
    new_model = ModelInfo(id="z", name="Z")
    ok, _ = await provider.add_model(new_model)
    assert ok is True

    info = await provider.get_info()
    assert info.extra_models[-1].id == "z"
    assert info.extra_models[-1].sort_order == 3


@pytest.mark.asyncio
async def test_reorder_models_updates_sort_order() -> None:
    provider = make_provider()
    provider.reorder_models(["c", "a", "b"])

    info = await provider.get_info()
    assert [m.id for m in info.models] == ["c", "a", "b"]

    # Extra models keep their previous order.
    assert [m.id for m in info.extra_models] == ["y", "x"]


@pytest.mark.asyncio
async def test_reorder_models_ignores_unknown_ids() -> None:
    provider = make_provider()
    provider.reorder_models(["unknown", "b", "a"])

    info = await provider.get_info()
    # Unknown IDs still consume an index; known IDs are assigned the index at
    # which they appear in the requested order.
    assert [m.id for m in info.models] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_reorder_models_across_both_lists() -> None:
    provider = make_provider()
    provider.reorder_models(["y", "a", "x", "b", "c"])

    # Cross-list reorder only affects the sort_order of each list; the
    # two lists are still returned separately.
    info = await provider.get_info()
    assert [m.id for m in info.models] == ["a", "b", "c"]
    assert [m.id for m in info.extra_models] == ["y", "x"]
