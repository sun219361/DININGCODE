from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


@dataclass
class ReviewRecord:
    res_name: str
    user_name: str
    rating: float | None
    review: str
    review_date: str
    url: str


class DiningCodeReviewCrawler:
    def __init__(
        self,
        driver: WebDriver,
        wait_sec: int = 10,
        sleep_sec: float = 0.5,
        click_more_limit: int = 10,
    ) -> None:
        self.driver = driver
        self.wait = WebDriverWait(driver, wait_sec)
        self.sleep_sec = sleep_sec
        self.click_more_limit = click_more_limit

    def load_page(self, url: str) -> None:
        self.driver.get(url)
        time.sleep(self.sleep_sec)

    def try_expand_reviews(self) -> None:
        for _ in range(self.click_more_limit):
            try:
                buttons = self.driver.find_elements(By.XPATH, "//button[contains(., '더보기')]")
                if not buttons:
                    break
                buttons[0].click()
                time.sleep(self.sleep_sec)
            except Exception:
                break

    def get_restaurant_name(self) -> str:
        try:
            return self.driver.find_element(By.CSS_SELECTOR, "h1").text.strip()
        except Exception:
            return ""

    def parse_rating_from_block(self, text: str) -> float | None:
        for token in text.split():
            try:
                value = float(token)
                if 0 <= value <= 5:
                    return value
            except Exception:
                continue
        return None

    def extract_reviews(self, url: str) -> list[ReviewRecord]:
        self.load_page(url)

        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        except Exception:
            pass

        self.try_expand_reviews()

        restaurant_name = self.get_restaurant_name()
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        text_blocks = soup.get_text("\n", strip=True).split("\n")

        records: list[ReviewRecord] = []

        # DiningCode 구조가 자주 바뀌기 때문에,
        # CSS selector 고정보다 블록 기반 보수적 파싱으로 처리
        possible_review_items = self.driver.find_elements(By.CSS_SELECTOR, "div, li")

        for block in possible_review_items:
            txt = block.text.strip()
            if len(txt) < 8:
                continue

            if restaurant_name and restaurant_name in txt and txt.count("\n") < 1:
                continue

            lines = [line.strip() for line in txt.split("\n") if line.strip()]
            if len(lines) < 2:
                continue

            user_name = lines[0]
            rating = self.parse_rating_from_block(txt)
            review_date = ""
            review = ""

            for line in lines[1:]:
                if any(char.isdigit() for char in line) and ("." in line or "-" in line):
                    if not review_date:
                        review_date = line
                else:
                    if len(line) >= 5:
                        review = line
                        break

            if not review:
                continue

            records.append(
                ReviewRecord(
                    res_name=restaurant_name,
                    user_name=user_name,
                    rating=rating,
                    review=review,
                    review_date=review_date,
                    url=url,
                )
            )

        dedup_df = pd.DataFrame([asdict(r) for r in records]).drop_duplicates()
        if dedup_df.empty:
            return []

        return [ReviewRecord(**row) for row in dedup_df.to_dict(orient="records")]

    def crawl_reviews(self, restaurant_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []

        for _, row in restaurant_df.iterrows():
            url = row["url"]

            try:
                review_records = self.extract_reviews(url)
                for record in review_records:
                    rows.append(asdict(record))
            except Exception as e:
                rows.append({
                    "res_name": row.get("name", ""),
                    "user_name": "",
                    "rating": None,
                    "review": "",
                    "review_date": "",
                    "url": url,
                    "crawl_error": str(e),
                })

        return pd.DataFrame(rows)