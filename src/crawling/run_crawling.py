from __future__ import annotations

from src.crawling.driver import build_chrome_driver
from src.crawling.diningcode_crawler import DiningCodeRestaurantCrawler, SEOUL_DISTRICTS
from src.crawling.review_crawler import DiningCodeReviewCrawler
from src.utils.io import write_csv
from src.utils.paths import RAW_DATA_DIR


def run() -> None:
    driver = build_chrome_driver(headless=True)

    try:
        restaurant_crawler = DiningCodeRestaurantCrawler(driver=driver, wait_sec=10, sleep_sec=0.7)
        link_df = restaurant_crawler.collect_restaurant_links(SEOUL_DISTRICTS)
        write_csv(link_df, RAW_DATA_DIR / "DiningCode_links.csv")

        restaurant_df = restaurant_crawler.crawl_restaurants(link_df)
        write_csv(restaurant_df, RAW_DATA_DIR / "DiningCode_df.csv")

        review_crawler = DiningCodeReviewCrawler(driver=driver, wait_sec=10, sleep_sec=0.7, click_more_limit=10)
        review_df = review_crawler.crawl_reviews(restaurant_df)
        write_csv(review_df, RAW_DATA_DIR / "DiningCode_review_df.csv")

    finally:
        driver.quit()


if __name__ == "__main__":
    run()