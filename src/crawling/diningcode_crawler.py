from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict

import pandas as pd
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


@dataclass
class RestaurantRecord:
    name: str
    address: str
    category: str
    main_mn: str
    price: str
    opng_tm: str
    rating: float | None
    rvw_cnt: int | None
    tags: str
    url: str


SEOUL_DISTRICTS = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구",
]


class DiningCodeRestaurantCrawler:
    def __init__(
        self,
        driver: WebDriver,
        wait_sec: int = 10,
        sleep_sec: float = 0.5,
    ) -> None:
        self.driver = driver
        self.wait = WebDriverWait(driver, wait_sec)
        self.sleep_sec = sleep_sec

    def get_search_url(self, district: str) -> str:
        return f"https://www.diningcode.com/list.dc?query={district}"

    def load_page(self, url: str) -> None:
        self.driver.get(url)
        time.sleep(self.sleep_sec)

    def safe_text(self, by: By, selector: str, default: str = "") -> str:
        try:
            return self.driver.find_element(by, selector).text.strip()
        except Exception:
            return default

    def safe_attr(self, by: By, selector: str, attr: str, default: str = "") -> str:
        try:
            return self.driver.find_element(by, selector).get_attribute(attr) or default
        except Exception:
            return default

    def extract_links_from_search_page(self) -> list[str]:
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        links = []

        for a_tag in soup.select("a[href]"):
            href = a_tag.get("href", "")
            if "/profile.php?rid=" in href:
                full_url = href if href.startswith("http") else f"https://www.diningcode.com{href}"
                links.append(full_url)

        return sorted(set(links))

    def collect_restaurant_links(self, districts: list[str] | None = None) -> pd.DataFrame:
        districts = districts or SEOUL_DISTRICTS
        rows: list[dict] = []

        for district in districts:
            search_url = self.get_search_url(district)
            self.load_page(search_url)
            links = self.extract_links_from_search_page()

            for link in links:
                rows.append({
                    "district": district,
                    "url": link,
                })

        return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

    def parse_rating(self, text: str) -> float | None:
        text = text.strip()
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def parse_review_count(self, text: str) -> int | None:
        text = text.replace(",", "").strip()
        match = re.search(r"(\d+)", text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def extract_restaurant_detail(self, url: str) -> RestaurantRecord:
        self.load_page(url)

        try:
            self.wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception:
            pass

        name = self.safe_text(By.CSS_SELECTOR, "h1")
        if not name:
            name = self.safe_text(By.CSS_SELECTOR, "div.tit-point")

        address = ""
        category = ""
        main_mn = ""
        price = ""
        opng_tm = ""
        rating = None
        rvw_cnt = None
        tags = ""

        page_text = BeautifulSoup(self.driver.page_source, "html.parser").get_text(" ", strip=True)

        try:
            info_blocks = self.driver.find_elements(By.CSS_SELECTOR, "li")
            for block in info_blocks:
                txt = block.text.strip()
                if "주소" in txt and not address:
                    address = txt
                elif "음식종류" in txt and not category:
                    category = txt
                elif "대표메뉴" in txt and not main_mn:
                    main_mn = txt
                elif "가격대" in txt and not price:
                    price = txt
                elif ("영업시간" in txt or "영업" in txt) and not opng_tm:
                    opng_tm = txt
        except Exception:
            pass

        try:
            score_candidates = self.driver.find_elements(By.CSS_SELECTOR, "span")
            for el in score_candidates:
                txt = el.text.strip()
                if rating is None and re.search(r"\d+\.\d+", txt):
                    rating = self.parse_rating(txt)
                if rvw_cnt is None and ("리뷰" in txt or "평가" in txt):
                    rvw_cnt = self.parse_review_count(txt)
        except Exception:
            pass

        tag_candidates = re.findall(r"#\S+", page_text)
        tags = ",".join(sorted(set(tag_candidates)))

        return RestaurantRecord(
            name=name,
            address=address,
            category=category,
            main_mn=main_mn,
            price=price,
            opng_tm=opng_tm,
            rating=rating,
            rvw_cnt=rvw_cnt,
            tags=tags,
            url=url,
        )

    def crawl_restaurants(self, link_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []

        for _, row in link_df.iterrows():
            url = row["url"]
            district = row.get("district", "")

            try:
                record = self.extract_restaurant_detail(url)
                item = asdict(record)
                item["district"] = district
                rows.append(item)
            except Exception as e:
                rows.append({
                    "name": "",
                    "address": "",
                    "category": "",
                    "main_mn": "",
                    "price": "",
                    "opng_tm": "",
                    "rating": None,
                    "rvw_cnt": None,
                    "tags": "",
                    "url": url,
                    "district": district,
                    "crawl_error": str(e),
                })

        return pd.DataFrame(rows)