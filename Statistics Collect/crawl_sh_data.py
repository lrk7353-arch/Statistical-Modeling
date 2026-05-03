import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import WebDriverException, NoSuchElementException


BASE_URL = "https://data.sh.gov.cn"


EXPECTED_LABELS = [
    "数据标签",
    "关键词",
    "更新频率",
    "国家主题分类",
    "数据量（条）",
    "数据量(条)",
    "数据大小",
    "首次发布日期",
    "更新日期",
    "空间范围",
    "时间范围",
    "联系方式",
    "最新数据集",
    "开放条件",
    "开放条件补充",
    "数据协议",
]


FORMAT_WORDS = ["csv", "json", "rdf", "xlsx", "xls", "xml", "api"]


def setup_driver(browser: str = "edge", headless: bool = False):
    if browser.lower() == "chrome":
        options = ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1600,1200")
        options.add_argument("--disable-blink-features=AutomationControlled")
        return webdriver.Chrome(options=options)

    options = EdgeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1600,1200")
    options.add_argument("--disable-blink-features=AutomationControlled")
    return webdriver.Edge(options=options)


def normalize_space(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def safe_int(x):
    if x is None:
        return None
    x = str(x).strip()
    if x in ["", "-", "--", "—", "无"]:
        return None
    x = re.sub(r"[^\d]", "", x)
    if not x:
        return None
    return int(x)


def scroll_full_page(driver, pause: float = 0.5):
    """
    让页面中懒加载内容尽量加载出来。
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    y = 0
    step = 900

    while y < last_height:
        driver.execute_script(f"window.scrollTo(0, {y});")
        time.sleep(pause)
        y += step
        last_height = driver.execute_script("return document.body.scrollHeight")

    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.3)


def get_inner_text(driver) -> str:
    try:
        return driver.execute_script("return document.body.innerText;")
    except WebDriverException:
        return ""


def parse_usage_counts(text: str):
    """
    解析：
    访问/下载次数：249558/173090
    访问/下载次数/调用次数：1/-/-
    """
    text = normalize_space(text)

    view_count = download_count = call_count = None

    m = re.search(
        r"访问\s*/\s*下载次数\s*/\s*调用次数\s*[:：]\s*([0-9\-—]+)\s*/\s*([0-9\-—]+)\s*/\s*([0-9\-—]+)",
        text,
    )
    if m:
        view_count = safe_int(m.group(1))
        download_count = safe_int(m.group(2))
        call_count = safe_int(m.group(3))
        return view_count, download_count, call_count

    m = re.search(
        r"访问\s*/\s*下载次数\s*[:：]\s*([0-9\-—]+)\s*/\s*([0-9\-—]+)",
        text,
    )
    if m:
        view_count = safe_int(m.group(1))
        download_count = safe_int(m.group(2))
        return view_count, download_count, call_count

    return view_count, download_count, call_count


def parse_title_and_department(text: str):
    """
    根据页面可见文本粗略解析标题与数据来源部门。
    """
    lines = [normalize_space(x) for x in text.splitlines() if normalize_space(x)]

    title = ""
    for i, line in enumerate(lines):
        if "首页" in line and "数据资源" in line and "数据接口详情" in line:
            if i + 1 < len(lines):
                title = lines[i + 1]
                break
        if "首页" in line and "数据资源" in line and "数据产品详情" in line:
            if i + 1 < len(lines):
                title = lines[i + 1]
                break

    if not title:
        # 兜底：找第一个看起来像数据集标题的长中文行
        for line in lines:
            if (
                len(line) >= 6
                and "上海市公共数据开放平台" not in line
                and "首页" not in line
                and "数据资源" not in line
                and "DATA" not in line
            ):
                title = line
                break

    source_department = ""
    m = re.search(r"数据来源部门\s*[:：]\s*([^\n\r]+)", text)
    if m:
        source_department = normalize_space(m.group(1))

    return title, source_department


def parse_summary(text: str):
    m = re.search(r"摘要\s*[:：]\s*(.+?)(?:\n|收藏|订阅|纠错|分享|数据标签)", text, flags=re.S)
    if m:
        return normalize_space(m.group(1))
    return ""


def parse_metadata_table(soup: BeautifulSoup):
    """
    解析详情页中的固定元数据表。
    逻辑：找到包含 label 的 td/th，然后取它右侧相邻单元格作为 value。
    """
    result = {}

    for label in EXPECTED_LABELS:
        result[label] = ""

    cells = soup.find_all(["td", "th"])
    for idx, cell in enumerate(cells):
        label_text = normalize_space(cell.get_text(" ", strip=True))
        if label_text in EXPECTED_LABELS and idx + 1 < len(cells):
            value_text = normalize_space(cells[idx + 1].get_text(" ", strip=True))
            result[label_text] = value_text

    # 统一字段名
    normalized = {
        "tags": result.get("数据标签", ""),
        "keywords": result.get("关键词", ""),
        "update_frequency": result.get("更新频率", ""),
        "theme_category": result.get("国家主题分类", ""),
        "data_quantity": result.get("数据量（条）") or result.get("数据量(条)", ""),
        "data_size": result.get("数据大小", ""),
        "first_publish_date": result.get("首次发布日期", ""),
        "update_date": result.get("更新日期", ""),
        "spatial_scope": result.get("空间范围", ""),
        "time_scope": result.get("时间范围", ""),
        "contact": result.get("联系方式", ""),
        "latest_dataset": result.get("最新数据集", ""),
        "open_condition": result.get("开放条件", ""),
        "open_condition_supplement": result.get("开放条件补充", ""),
        "data_agreement": result.get("数据协议", ""),
    }
    return normalized


def extract_download_formats(text: str):
    lower = text.lower()
    found = []
    for fmt in FORMAT_WORDS:
        if re.search(rf"\b{fmt}\b", lower):
            found.append(fmt)

    # API 说明文档里的 API-1(JSON) 也会命中 json，所以这里保留但后续单独解释
    return sorted(set(found))


def parse_api_info(text: str):
    has_api_doc = "API说明文档" in text or "API DOCUMENTATION" in text
    has_api_json = "API-1(JSON)" in text
    has_api_xml = "API-2(XML)" in text

    service_urls = re.findall(r"https://data\.sh\.gov\.cn/interface/[^\s]+", text)
    service_urls = sorted(set(service_urls))

    api_block_hidden = "请先申请使用" in text and "接口服务地址" in text

    return {
        "has_api_doc": int(has_api_doc),
        "has_api_json": int(has_api_json),
        "has_api_xml": int(has_api_xml),
        "api_service_url_count": len(service_urls),
        "api_service_urls": ";".join(service_urls),
        "api_service_hidden_or_need_apply": int(api_block_hidden),
    }


def extract_tables(soup: BeautifulSoup):
    """
    抽取页面中的所有表格，后续用于识别字段名。
    """
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [normalize_space(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def extract_api_fields_from_tables(tables):
    """
    从 API 参数说明 / 返回值说明表中抽取字段。
    常见列：参数、参数描述、参数类型、字段大小。
    """
    fields = []
    descriptions = []

    ignore = {"token", "offset", "limit", "参数", "参数描述", "参数类型", "字段大小"}

    for rows in tables:
        for row in rows:
            if len(row) >= 2:
                name = row[0]
                desc = row[1]
                if name and name not in ignore and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$|^[\u4e00-\u9fffA-Za-z0-9_]+$", name):
                    fields.append(name)
                    if desc:
                        descriptions.append(desc)

    fields = sorted(set(fields))
    return fields, descriptions


def extract_sample_headers_from_tables(tables):
    """
    尝试从数据样例表中抽取字段名。
    排除明显的元数据表和 API 文档表。
    """
    candidates = []

    bad_words = {
        "数据标签",
        "关键词",
        "更新频率",
        "国家主题分类",
        "数据大小",
        "开放条件",
        "参数",
        "参数描述",
        "参数类型",
        "字段大小",
    }

    for rows in tables:
        if not rows:
            continue

        first_row = rows[0]
        joined = " ".join(first_row)

        if any(w in joined for w in bad_words):
            continue

        # 数据样例表通常列数较多
        if len(first_row) >= 3:
            candidates.append(first_row)

    if candidates:
        # 选字段最多的一行
        headers = max(candidates, key=len)
        headers = [h for h in headers if h and not h.startswith("Unnamed")]
        return headers

    return []


def infer_field_features(fields, text: str):
    joined = " ".join(fields) + " " + text

    time_patterns = [
        "时间",
        "日期",
        "年份",
        "年度",
        "年月",
        "year",
        "date",
        "time",
        "update",
        "jpt_update_time",
    ]
    geo_patterns = [
        "地址",
        "位置",
        "经纬度",
        "经度",
        "纬度",
        "街道",
        "空间",
        "区",
        "location",
        "lng",
        "lat",
        "longitude",
        "latitude",
    ]

    has_time_field = any(p.lower() in joined.lower() for p in time_patterns)
    has_geo_field = any(p.lower() in joined.lower() for p in geo_patterns)

    unnamed_count = sum(1 for f in fields if str(f).lower().startswith("unnamed"))

    return {
        "field_count_est": len(set(fields)),
        "has_time_field": int(has_time_field),
        "has_geo_field": int(has_geo_field),
        "unnamed_field_count": unnamed_count,
    }


def count_recommended_datasets(text: str):
    """
    粗略估计数据集推荐数量。
    推荐卡片通常包含“查看/调用”和“了解详情”。
    """
    if "数据集推荐" not in text:
        return 0

    after = text.split("数据集推荐", 1)[-1]
    # 防止把后面整页都算进去，截一段
    after = after[:5000]

    count1 = after.count("了解详情")
    count2 = len(re.findall(r"查看\s*/\s*调用", after))

    return max(count1, count2)


def compute_simple_scores(record):
    """
    初步可计算分数，不作为最终论文模型，只作为字段审计阶段的 MVP。
    """
    # ActualUse：先做原始 log 输入，后续全样本里再统一标准化
    view = record.get("view_count")
    download = record.get("download_count")
    call = record.get("call_count")

    usage_parts = []
    if view is not None:
        usage_parts.append(("view", view))
    if download is not None:
        usage_parts.append(("download", download))
    if call is not None:
        usage_parts.append(("call", call))

    record["actual_use_raw_available_channels"] = len(usage_parts)
    record["actual_use_raw_sum_log"] = sum(
        [0 if v is None else __import__("math").log1p(v) for _, v in usage_parts]
    )

    # PotentialUse 初版：字段审计用的简化版
    potential = 0

    if record.get("summary"):
        potential += 1
    if record.get("theme_category"):
        potential += 1
    if record.get("keywords"):
        potential += 1
    if record.get("update_frequency"):
        potential += 1
    if record.get("update_date"):
        potential += 1
    if record.get("field_count_est", 0) >= 5:
        potential += 1
    if record.get("has_time_field") == 1:
        potential += 1
    if record.get("has_geo_field") == 1:
        potential += 1
    if record.get("has_api_doc") == 1:
        potential += 1
    if record.get("download_format_count", 0) >= 2:
        potential += 1
    if record.get("recommended_dataset_count", 0) >= 1:
        potential += 1

    record["potential_use_audit_score_0_11"] = potential

    return record


def parse_detail_page(driver, url: str):
    driver.get(url)
    time.sleep(2)
    scroll_full_page(driver, pause=0.25)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    text = get_inner_text(driver)

    title, source_department = parse_title_and_department(text)
    summary = parse_summary(text)
    view_count, download_count, call_count = parse_usage_counts(text)

    metadata = parse_metadata_table(soup)
    formats = extract_download_formats(text)
    api_info = parse_api_info(text)

    tables = extract_tables(soup)
    api_fields, api_descs = extract_api_fields_from_tables(tables)
    sample_headers = extract_sample_headers_from_tables(tables)

    all_fields = sorted(set(api_fields + sample_headers))
    field_features = infer_field_features(all_fields, text)

    record = {
        "dataset_name": title,
        "dataset_url": url,
        "source_department": source_department,
        "summary": summary,
        "view_count": view_count,
        "download_count": download_count,
        "call_count": call_count,
        "download_formats": ";".join(formats),
        "download_format_count": len(formats),
        "has_csv": int("csv" in formats),
        "has_json": int("json" in formats),
        "has_rdf": int("rdf" in formats),
        "has_xlsx": int("xlsx" in formats or "xls" in formats),
        "has_xml": int("xml" in formats),
        "has_data_sample": int("数据样例" in text or "DATA SAMPLE" in text),
        "recommended_dataset_count": count_recommended_datasets(text),
        "user_comment_section": int("用户评分及评论" in text),
        "api_fields": ";".join(api_fields),
        "sample_headers": ";".join(sample_headers),
        "all_fields_est": ";".join(all_fields),
        "field_description_count_est": len(api_descs),
        "page_text_length": len(text),
    }

    record.update(metadata)
    record.update(api_info)
    record.update(field_features)
    record = compute_simple_scores(record)

    return record


def looks_like_detail_url(href: str):
    if not href:
        return False

    href_lower = href.lower()

    keywords = [
        "detail",
        "dataresource",
        "dataproduct",
        "resource",
        "interface",
    ]

    if any(k in href_lower for k in keywords):
        return True

    return False


def collect_links_from_current_page(driver):
    """
    从当前列表页粗略收集详情链接。
    如果网站链接结构和预期不同，至少可以把候选链接导出来检查。
    """
    links = set()

    for a in driver.find_elements(By.TAG_NAME, "a"):
        try:
            href = a.get_attribute("href")
            text = normalize_space(a.text)
        except Exception:
            continue

        if not href:
            continue

        full = urljoin(BASE_URL, href)

        if "data.sh.gov.cn" not in urlparse(full).netloc:
            continue

        if looks_like_detail_url(full) or "了解详情" in text or "数据产品详情" in text or "数据接口详情" in text:
            links.add(full)

    return sorted(links)


def click_next_page(driver):
    """
    尝试点击下一页。不同页面结构可能不一样，所以做多个 XPath 尝试。
    """
    xpaths = [
        "//a[contains(text(), '下一页')]",
        "//button[contains(text(), '下一页')]",
        "//*[contains(@class, 'next')]",
        "//*[contains(@title, '下一页')]",
    ]

    for xp in xpaths:
        try:
            elems = driver.find_elements(By.XPATH, xp)
            for elem in elems:
                if elem.is_displayed() and elem.is_enabled():
                    driver.execute_script("arguments[0].click();", elem)
                    time.sleep(2)
                    return True
        except Exception:
            continue

    return False


def collect_detail_links(driver, start_url: str, max_links: int = 50, manual: bool = False):
    if manual:
        print("\n浏览器已经打开。请你手动进入数据资源列表页 / 搜索结果页。")
        print("进入后回到终端按 Enter，脚本会从当前页面收集详情链接。")
        input("准备好后按 Enter：")
    else:
        driver.get(start_url)
        time.sleep(3)

    links = []
    visited_pages = 0

    while len(links) < max_links and visited_pages < 20:
        scroll_full_page(driver, pause=0.2)
        page_links = collect_links_from_current_page(driver)

        for link in page_links:
            if link not in links:
                links.append(link)
                print(f"[link] {len(links)} {link}")
                if len(links) >= max_links:
                    break

        visited_pages += 1

        if len(links) >= max_links:
            break

        moved = click_next_page(driver)
        if not moved:
            break

    return links[:max_links]


def read_urls_file(path: str):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=BASE_URL, help="数据资源列表页或搜索结果页 URL")
    parser.add_argument("--urls", default="", help="如果你已有详情页 URL 列表，放在 txt 文件里")
    parser.add_argument("--max", type=int, default=50, help="最多抓取多少个详情页")
    parser.add_argument("--out", default="sh_data_audit.xlsx", help="输出 Excel 文件")
    parser.add_argument("--browser", default="edge", choices=["edge", "chrome"], help="浏览器")
    parser.add_argument("--headless", action="store_true", help="无头模式，不建议第一次使用")
    parser.add_argument("--manual", action="store_true", help="手动打开列表页后再让脚本收集链接")
    parser.add_argument("--delay", type=float, default=1.5, help="每个详情页之间的等待秒数")
    args = parser.parse_args()

    out_path = Path(args.out)

    driver = setup_driver(args.browser, args.headless)

    try:
        if args.urls:
            detail_links = read_urls_file(args.urls)[: args.max]
        else:
            detail_links = collect_detail_links(
                driver=driver,
                start_url=args.start,
                max_links=args.max,
                manual=args.manual,
            )

        if not detail_links:
            print("没有收集到详情页链接。")
            print("建议使用 --manual 模式，或者手动把详情页 URL 放到 urls.txt 后用 --urls urls.txt。")
            return

        print(f"\n共准备抓取 {len(detail_links)} 个详情页。")

        records = []
        failed = []

        for url in tqdm(detail_links, desc="Scraping detail pages"):
            try:
                record = parse_detail_page(driver, url)
                records.append(record)
                time.sleep(args.delay)
            except Exception as e:
                failed.append({"url": url, "error": repr(e)})
                print(f"[failed] {url} {e}")

        df = pd.DataFrame(records)

        # 字段顺序优化
        preferred_cols = [
            "dataset_name",
            "dataset_url",
            "source_department",
            "theme_category",
            "tags",
            "keywords",
            "summary",
            "view_count",
            "download_count",
            "call_count",
            "actual_use_raw_available_channels",
            "actual_use_raw_sum_log",
            "potential_use_audit_score_0_11",
            "update_frequency",
            "data_quantity",
            "data_size",
            "first_publish_date",
            "update_date",
            "spatial_scope",
            "time_scope",
            "open_condition",
            "open_condition_supplement",
            "download_formats",
            "download_format_count",
            "has_csv",
            "has_json",
            "has_rdf",
            "has_xlsx",
            "has_xml",
            "has_api_doc",
            "has_api_json",
            "has_api_xml",
            "api_service_url_count",
            "api_service_hidden_or_need_apply",
            "has_data_sample",
            "field_count_est",
            "field_description_count_est",
            "has_time_field",
            "has_geo_field",
            "recommended_dataset_count",
            "user_comment_section",
            "all_fields_est",
            "sample_headers",
            "api_fields",
            "contact",
            "data_agreement",
            "latest_dataset",
            "page_text_length",
        ]

        cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
        df = df[cols]

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="field_audit")
            pd.DataFrame({"url": detail_links}).to_excel(writer, index=False, sheet_name="detail_links")
            if failed:
                pd.DataFrame(failed).to_excel(writer, index=False, sheet_name="failed")

        print(f"\n完成，已输出：{out_path.resolve()}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()