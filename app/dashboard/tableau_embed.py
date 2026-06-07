from __future__ import annotations

TABLEAU_PUBLIC_URL = "https://public.tableau.com/views/Dashboard_Sales_17808249802880/Dashboard?:showVizHome=no&:embed=true"


def tableau_iframe() -> str:
    return f"""
    <iframe
        src="{TABLEAU_PUBLIC_URL}"
        width="100%"
        height="900"
        frameborder="0"
        allowfullscreen>
    </iframe>
    """
