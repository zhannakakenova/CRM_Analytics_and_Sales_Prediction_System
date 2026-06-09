from __future__ import annotations

from app.dashboard.gradio_app import (
    customer_details,
    dropdown_values,
    product_details,
    regression_product_details,
    sales_territory,
    update_states_and_territory,
    update_subcategories,
    update_territory,
)


def _choice_values(dropdown) -> list[str]:
    return [value for _, value in dropdown.choices]


def test_country_updates_state_choices() -> None:
    states, territory = update_states_and_territory("Australia")

    assert "Queensland" in _choice_values(states)
    assert "California" not in _choice_values(states)
    assert territory == "Australia"


def test_country_choices_exclude_placeholder_values() -> None:
    countries = dropdown_values("country_region")

    assert "[Not Applicable]" not in countries
    assert "Unknown" not in countries
    assert countries == ["Australia", "Canada", "France", "Germany", "United Kingdom", "United States"]


def test_state_updates_sales_territory() -> None:
    assert sales_territory("United States", "California") == "Southwest"
    assert update_territory("United States", "Washington") == "Northwest"


def test_category_updates_subcategory_choices() -> None:
    subcategories = update_subcategories("Bikes")

    assert _choice_values(subcategories) == ["Mountain Bikes", "Road Bikes", "Touring Bikes"]


def test_customer_selection_fills_location_and_channel() -> None:
    country, state, region, channel = customer_details(11000)

    assert (country, state.value, region, channel) == ("Australia", "Queensland", "Australia", "Internet")


def test_product_selection_fills_product_attributes() -> None:
    category, subcategory, color, list_price = product_details(214)

    assert (category, subcategory.value, color, list_price) == ("Accessories", "Helmets", "Red", 34.99)
    assert regression_product_details(214)[-2:] == (34.99, 34.99)
