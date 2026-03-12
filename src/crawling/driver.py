from __future__ import annotations

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


def build_chrome_driver(
    headless: bool = True,
    window_size: str = "1400,1000",
    driver_path: str | None = None,
) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument(f"--window-size={window_size}")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=ko-KR")
    options.add_argument("--blink-settings=imagesEnabled=false")

    if driver_path:
        service = Service(driver_path)
        return webdriver.Chrome(service=service, options=options)

    return webdriver.Chrome(options=options)