# website_audit_tool_with_summaries.py
# Advanced Website Comparison & SEO Audit Tool with per-step summaries and final recommendation
#
# Usage:
#   pip install streamlit requests beautifulsoup4 scikit-learn lxml
#   streamlit run website_audit_tool_with_summaries.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import urlparse, urlunparse
from collections import Counter

# TF-IDF / cosine similarity — try sklearn but provide a lightweight fallback if missing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def similarity_score(text1, text2):
        try:
            vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
            tfidf = vec.fit_transform([text1 or " ", text2 or " "])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(sim * 100)
        except Exception:
            return 0.0
except Exception:
    # fallback: simple Jaccard on word sets (not as good, but avoids crash)
    def similarity_score(text1, text2):
        a = set((text1 or "").lower().split())
        b = set((text2 or "").lower().split())
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b) or 1
        return float((inter / union) * 100)


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SeoGapBot/1.0; +https://example.com/bot)"
}


# -------------------------
# URL normalization helpers
# -------------------------
def normalize_input_url(raw_url):
    """
    Clean user's URL input:
      - strip leading/trailing whitespace
      - if empty -> return ''
      - if missing scheme, prepend http://
      - return cleaned URL string
    """
    if not raw_url:
        return ""
    u = raw_url.strip()
    if u == "":
        return ""
    # if user pasted multiple URLs separated by spaces/newlines accidentally, take first token
    if "\n" in u or " " in u:
        # split on whitespace and take first non-empty token
        tokens = [t for t in re.split(r"\s+", u) if t.strip()]
        if tokens:
            u = tokens[0]
    # if missing scheme, add http:// so urlparse works consistently
    parsed = urlparse(u)
    if not parsed.scheme:
        u = "http://" + u
    return u


def safe_get_domain(raw_url):
    """
    Return the cleaned domain (netloc) for display. Accepts raw user input (with possible spaces).
    """
    u = normalize_input_url(raw_url)
    try:
        netloc = urlparse(u).netloc
        netloc = re.sub(r'^.*@', '', netloc)
        netloc = re.sub(r':80$|:443$', '', netloc)
        return netloc or u
    except Exception:
        return u


# -------------------------
# Network & parsing
# -------------------------
def fetch_html(url, timeout=15):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return ""


def parse_page(html):
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Meta description
    meta_desc = ""
    m = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
    if m and m.get("content"):
        meta_desc = m["content"].strip()
    else:
        m = soup.find("meta", attrs={"property": re.compile(r"og:description", re.I)})
        if m and m.get("content"):
            meta_desc = m["content"].strip()
        else:
            m = soup.find("meta", attrs={"itemprop": re.compile(r"description", re.I)})
            if m and m.get("content"):
                meta_desc = m["content"].strip()

    # Headings
    h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]
    h2s = [h.get_text(strip=True) for h in soup.find_all("h2")]
    h3s = [h.get_text(strip=True) for h in soup.find_all("h3")]

    # Images (alt + src)
    images = []
    for img in soup.find_all("img"):
        alt = img.get("alt") or ""
        src = img.get("src") or ""
        images.append({"alt": alt.strip(), "src": src})

    # Schema JSON-LD
    schema_data = []
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            parsed = json.loads(script.string)
            schema_data.append(parsed)
        except Exception:
            # best-effort: try to clean and parse
            txt = re.sub(r'<!--.*?-->', '', script.string, flags=re.S)
            try:
                parsed = json.loads(txt)
                schema_data.append(parsed)
            except Exception:
                # skip if not parseable
                pass

    # visible text
    for s in soup(['script', 'style', 'noscript', 'iframe']):
        s.extract()
    visible_text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))

    return {
        "title": title,
        "meta_desc": meta_desc,
        "h1s": h1s,
        "h2s": h2s,
        "h3s": h3s,
        "images": images,
        "schema": schema_data,
        "text": visible_text,
    }


# Keyword helpers
def count_keyword_in_text(text, kw):
    # whole-word, case-insensitive
    return len(re.findall(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE))


def keywords_presence_map(text, keywords):
    return {kw: count_keyword_in_text(text, kw) for kw in keywords}


def keywords_in_list_of_texts(list_of_texts, keywords):
    # return map keyword -> total occurrences across items and which items contain it
    result = {kw: {"count": 0, "in_items": []} for kw in keywords}
    for i, t in enumerate(list_of_texts, start=1):
        for kw in keywords:
            cnt = count_keyword_in_text(t, kw)
            if cnt > 0:
                result[kw]["count"] += cnt
                result[kw]["in_items"].append({"index": i, "text": t, "count": cnt})
    return result


# Simple SEO hints
def simple_seo_scores(parsed):
    title = parsed.get("title", "")
    meta = parsed.get("meta_desc", "")
    h1s = parsed.get("h1s", [])

    # title length scoring (0-10)
    t_len = len(title)
    if t_len == 0:
        tlen_score = 0
    elif 50 <= t_len <= 60:
        tlen_score = 10
    else:
        # partial: subtract 1 point for every ~5 chars away from 55
        tlen_score = max(0, round(10 - (abs(55 - t_len) / 5), 2))

    # meta length scoring (0-10)
    m_len = len(meta)
    if m_len == 0:
        mlen_score = 0
    elif 120 <= m_len <= 160:
        mlen_score = 10
    else:
        mlen_score = max(0, round(10 - (abs(140 - m_len) / 6), 2))

    # H1 scoring (0-15)
    h_count = len(h1s)
    if h_count == 0:
        h1_score = 0
    elif h_count == 1:
        h1_score = 15
    else:
        # multiple H1s penalized slightly
        h1_score = max(0, 15 - (h_count - 1) * 4)

    return {
        "title_len_score": float(tlen_score),
        "meta_len_score": float(mlen_score),
        "h1_score": float(h1_score),
    }


def compute_overall_score(parsed, keywords):
    """
    Weighted scoring:
      - title length: 10
      - title keywords: 10
      - meta length: 10
      - meta keywords: 10
      - h1: 15
      - content keyword coverage: 25
      - image alt ratio: 10
      - schema presence: 10
    Normalizes final score to 0-100 taking into account absent keywords by scaling max_possible.
    """
    weights = {
        "title_len": 10,
        "title_kw": 10,
        "meta_len": 10,
        "meta_kw": 10,
        "h1": 15,
        "content_kw": 25,
        "image_alt": 10,
        "schema": 10,
    }

    seo = simple_seo_scores(parsed)
    raw = 0.0
    max_possible = 0.0

    # title length
    raw += (seo["title_len_score"] / 10.0) * weights["title_len"]
    max_possible += weights["title_len"]

    # title keywords
    total_keywords = len(keywords)
    if total_keywords > 0:
        # proportion of keywords present in title
        title_kw_counts = sum(1 for kw in keywords if count_keyword_in_text(parsed.get("title", ""), kw) > 0)
        raw += (title_kw_counts / total_keywords) * weights["title_kw"]
        max_possible += weights["title_kw"]

    # meta length
    raw += (seo["meta_len_score"] / 10.0) * weights["meta_len"]
    max_possible += weights["meta_len"]

    # meta keywords
    if total_keywords > 0:
        meta_kw_counts = sum(1 for kw in keywords if count_keyword_in_text(parsed.get("meta_desc", ""), kw) > 0)
        raw += (meta_kw_counts / total_keywords) * weights["meta_kw"]
        max_possible += weights["meta_kw"]

    # h1
    raw += (seo["h1_score"] / 15.0) * weights["h1"]
    max_possible += weights["h1"]

    # content keyword coverage
    if total_keywords > 0:
        present_kw_in_text = sum(1 for kw in keywords if count_keyword_in_text(parsed.get("text", ""), kw) > 0)
        raw += (present_kw_in_text / total_keywords) * weights["content_kw"]
        max_possible += weights["content_kw"]

    # image alt ratio
    images = parsed.get("images", [])
    if images:
        with_alt = sum(1 for img in images if img.get("alt"))
        alt_ratio = with_alt / len(images)
        raw += alt_ratio * weights["image_alt"]
    else:
        # no images -> don't penalize. treat as full points for image-alt component
        raw += weights["image_alt"]
    max_possible += weights["image_alt"]

    # schema presence
    if parsed.get("schema"):
        raw += weights["schema"]
    max_possible += weights["schema"]

    # normalize to 0-100
    score = (raw / max_possible) * 100 if max_possible > 0 else 0.0
    return round(score, 2), {"raw": raw, "max_possible": max_possible}


def extract_schema_types(schema_blocks):
    types = []
    def collect(obj):
        if isinstance(obj, dict):
            if '@type' in obj:
                t = obj.get('@type')
                if isinstance(t, list):
                    types.extend(t)
                else:
                    types.append(t)
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)
    collect(schema_blocks)
    # normalize and unique
    return sorted(list({str(t) for t in types if t}))


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Website Comparator & SEO Audit", layout="wide")
st.title("Website Comparator & SEO Audit — with per-step summaries")

with st.form("compare_form"):
    col1, col2 = st.columns(2)
    with col1:
        raw_url1 = st.text_input("Enter First Website URL (your site)", value="")
    with col2:
        raw_url2 = st.text_input("Enter Second Website URL (competitor)", value="")

    keywords_input = st.text_area(
        "Enter keywords (comma-separated). Example: 'pain relief, detox candy, reset oil'",
        value="",
        height=100,
    )

    submitted = st.form_submit_button("Compare")

if submitted:
    # normalize user-entered URLs (trim spaces/newlines and ensure scheme)
    url1 = normalize_input_url(raw_url1)
    url2 = normalize_input_url(raw_url2)

    if not url1 or not url2:
        st.warning("Please enter both URLs to compare.")
    else:
        domain1 = safe_get_domain(raw_url1)
        domain2 = safe_get_domain(raw_url2)
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

        with st.spinner("Fetching pages..."):
            html1 = fetch_html(url1)
            html2 = fetch_html(url2)

        if not html1 or not html2:
            st.error("One or both pages could not be fetched. Check the URLs and try again.")
        else:
            with st.spinner("Parsing pages and running analysis..."):
                site1 = parse_page(html1)
                site2 = parse_page(html2)

                # top summary and similarity
                sim_score = similarity_score(site1["text"], site2["text"])
                st.markdown(f"### Comparison: **{domain1}**  ⇄  **{domain2}**")
                top_col1, top_col2, top_col3 = st.columns([1, 1, 1])
                with top_col1:
                    st.markdown(f"**{domain1}**")
                    score1, score1_info = compute_overall_score(site1, keywords)
                    st.write(f"Overall score: **{score1} / 100**")
                with top_col2:
                    st.markdown("**Content Similarity**")
                    st.metric("Similarity", f"{sim_score:.2f} %")
                with top_col3:
                    st.markdown(f"**{domain2}**")
                    score2, score2_info = compute_overall_score(site2, keywords)
                    st.write(f"Overall score: **{score2} / 100**")

                st.markdown("---")

                # --- META Title & Description + summary ---
                st.header("Meta Title & Meta Description")
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(domain1)
                    st.write("**Title:**")
                    st.code(site1["title"] or "(missing)")
                    st.write(f"Length: {len(site1['title'])}")
                    st.write("**Meta Description:**")
                    st.code(site1["meta_desc"] or "(missing)")
                    st.write(f"Length: {len(site1['meta_desc'])}")
                with c2:
                    st.subheader(domain2)
                    st.write("**Title:**")
                    st.code(site2["title"] or "(missing)")
                    st.write(f"Length: {len(site2['title'])}")
                    st.write("**Meta Description:**")
                    st.code(site2["meta_desc"] or "(missing)")
                    st.write(f"Length: {len(site2['meta_desc'])}")

                # Meta summary: keyword mentions in title/meta
                st.subheader("Meta Summary")
                ms1 = []
                ms2 = []
                if keywords:
                    title_map1 = keywords_presence_map(site1["title"], keywords)
                    meta_map1 = keywords_presence_map(site1["meta_desc"], keywords)
                    title_map2 = keywords_presence_map(site2["title"], keywords)
                    meta_map2 = keywords_presence_map(site2["meta_desc"], keywords)

                    ms1.append(f"Keywords in title: {', '.join([k for k, v in title_map1.items() if v>0]) or '(none)'}")
                    ms1.append(f"Keywords in meta description: {', '.join([k for k, v in meta_map1.items() if v>0]) or '(none)'}")

                    ms2.append(f"Keywords in title: {', '.join([k for k, v in title_map2.items() if v>0]) or '(none)'}")
                    ms2.append(f"Keywords in meta description: {', '.join([k for k, v in meta_map2.items() if v>0]) or '(none)'}")
                else:
                    ms1.append("No keywords provided — keyword checks skipped.")
                    ms2.append("No keywords provided — keyword checks skipped.")

                col_ms1, col_ms2 = st.columns(2)
                with col_ms1:
                    st.info("\n".join(ms1))
                with col_ms2:
                    st.info("\n".join(ms2))

                st.markdown("---")

                # --- Headings (H1/H2/H3) + summary ---
                st.header("Headings (H1 / H2 / H3)")
                hcol1, hcol2 = st.columns(2)
                with hcol1:
                    st.subheader(domain1)
                    st.write("H1s:")
                    st.write(site1["h1s"] or ["(none)"])
                    st.write("H2s:")
                    st.write(site1["h2s"] or ["(none)"])
                    st.write("H3s:")
                    st.write(site1["h3s"] or ["(none)"])
                with hcol2:
                    st.subheader(domain2)
                    st.write("H1s:")
                    st.write(site2["h1s"] or ["(none)"])
                    st.write("H2s:")
                    st.write(site2["h2s"] or ["(none)"])
                    st.write("H3s:")
                    st.write(site2["h3s"] or ["(none)"])

                # Headings summary: which headings contain keywords
                st.subheader("Headings Summary")
                hs1 = []
                hs2 = []
                if keywords:
                    for kw in keywords:
                        k_in_h1_1 = sum(count_keyword_in_text(h, kw) for h in site1["h1s"])
                        k_in_h1_2 = sum(count_keyword_in_text(h, kw) for h in site2["h1s"])
                        k_in_h2_1 = sum(count_keyword_in_text(h, kw) for h in site1["h2s"])
                        k_in_h2_2 = sum(count_keyword_in_text(h, kw) for h in site2["h2s"])
                        hs1.append(f"'{kw}': H1={k_in_h1_1}, H2={k_in_h2_1}")
                        hs2.append(f"'{kw}': H1={k_in_h1_2}, H2={k_in_h2_2}")
                else:
                    hs1.append("No keywords provided — heading keyword checks skipped.")
                    hs2.append("No keywords provided — heading keyword checks skipped.")

                hcol_sum1, hcol_sum2 = st.columns(2)
                with hcol_sum1:
                    st.info("\n".join(hs1))
                with hcol_sum2:
                    st.info("\n".join(hs2))

                st.markdown("---")

                # --- Images & ALT Tags + summary ---
                st.header("Images & ALT Tags")
                icol1, icol2 = st.columns(2)
                with icol1:
                    st.subheader(domain1)
                    if site1["images"]:
                        st.table([{"alt": img["alt"] or "(missing)", "src": img["src"]} for img in site1["images"]])
                    else:
                        st.write("No images found.")
                with icol2:
                    st.subheader(domain2)
                    if site2["images"]:
                        st.table([{"alt": img["alt"] or "(missing)", "src": img["src"]} for img in site2["images"]])
                    else:
                        st.write("No images found.")

                # images summary
                st.subheader("Images Summary")
                img_summ_1 = []
                img_summ_2 = []
                imgs1 = site1["images"]
                imgs2 = site2["images"]
                if imgs1:
                    missing1 = sum(1 for i in imgs1 if not i["alt"])
                    img_summ_1.append(f"Total images: {len(imgs1)}")
                    img_summ_1.append(f"Images missing alt: {missing1}")
                    if keywords:
                        imgs_alt_texts = " ".join(i["alt"] for i in imgs1 if i["alt"])
                        present = [kw for kw in keywords if count_keyword_in_text(imgs_alt_texts, kw) > 0]
                        img_summ_1.append(f"Keywords mentioned in alt text: {', '.join(present) or '(none)'}")
                else:
                    img_summ_1.append("No images found.")

                if imgs2:
                    missing2 = sum(1 for i in imgs2 if not i["alt"])
                    img_summ_2.append(f"Total images: {len(imgs2)}")
                    img_summ_2.append(f"Images missing alt: {missing2}")
                    if keywords:
                        imgs_alt_texts2 = " ".join(i["alt"] for i in imgs2 if i["alt"])
                        present2 = [kw for kw in keywords if count_keyword_in_text(imgs_alt_texts2, kw) > 0]
                        img_summ_2.append(f"Keywords mentioned in alt text: {', '.join(present2) or '(none)'}")
                else:
                    img_summ_2.append("No images found.")

                icol_sum1, icol_sum2 = st.columns(2)
                with icol_sum1:
                    st.info("\n".join(img_summ_1))
                with icol_sum2:
                    st.info("\n".join(img_summ_2))

                st.markdown("---")

                # --- Schema Markup + summary ---
                st.header("Schema Markup (JSON-LD detected)")
                scol1, scol2 = st.columns(2)
                with scol1:
                    st.subheader(domain1)
                    if site1["schema"]:
                        for idx, sdata in enumerate(site1["schema"], 1):
                            with st.expander(f"JSON-LD block #{idx}"):
                                st.json(sdata)
                    else:
                        st.write("No JSON-LD schema found.")
                with scol2:
                    st.subheader(domain2)
                    if site2["schema"]:
                        for idx, sdata in enumerate(site2["schema"], 1):
                            with st.expander(f"JSON-LD block #{idx}"):
                                st.json(sdata)
                    else:
                        st.write("No JSON-LD schema found.")

                # schema summary
                st.subheader("Schema Summary")
                schema_s1 = []
                schema_s2 = []
                types1 = extract_schema_types(site1["schema"]) if site1["schema"] else []
                types2 = extract_schema_types(site2["schema"]) if site2["schema"] else []
                schema_s1.append(f"Detected types: {types1 or '(none)'}")
                schema_s2.append(f"Detected types: {types2 or '(none)'}")
                if keywords:
                    # look for keywords inside JSON-LD (stringify)
                    s1text = json.dumps(site1["schema"], ensure_ascii=False) if site1["schema"] else ""
                    s2text = json.dumps(site2["schema"], ensure_ascii=False) if site2["schema"] else ""
                    present1 = [kw for kw in keywords if count_keyword_in_text(s1text, kw) > 0]
                    present2 = [kw for kw in keywords if count_keyword_in_text(s2text, kw) > 0]
                    schema_s1.append(f"Keywords in schema: {', '.join(present1) or '(none)'}")
                    schema_s2.append(f"Keywords in schema: {', '.join(present2) or '(none)'}")
                st.columns(2)[0].info("\n".join(schema_s1))
                st.columns(2)[1].info("\n".join(schema_s2))

                st.markdown("---")

                # --- Keyword Analysis (Counts & Density) + summary ---
                st.header("Keyword Analysis (Counts & Density)")
                if keywords:
                    kcol1, kcol2 = st.columns(2)
                    kstats1 = []
                    kstats2 = []
                    wordcount1 = len(site1["text"].split()) or 1
                    wordcount2 = len(site2["text"].split()) or 1
                    for kw in keywords:
                        c1 = count_keyword_in_text(site1["text"], kw)
                        c2 = count_keyword_in_text(site2["text"], kw)
                        d1 = round((c1 / wordcount1) * 100, 4)
                        d2 = round((c2 / wordcount2) * 100, 4)
                        kstats1.append({"keyword": kw, "count": c1, "density_%": d1})
                        kstats2.append({"keyword": kw, "count": c2, "density_%": d2})
                    with kcol1:
                        st.subheader(domain1)
                        st.table(kstats1)
                    with kcol2:
                        st.subheader(domain2)
                        st.table(kstats2)

                    # quick summary comparing which site uses each keyword more
                    comp_lines = []
                    for i, kw in enumerate(keywords):
                        if kstats1[i]["count"] > kstats2[i]["count"]:
                            comp_lines.append(f"'{kw}': **{domain1}** uses it more ({kstats1[i]['count']} vs {kstats2[i]['count']})")
                        elif kstats1[i]["count"] < kstats2[i]["count"]:
                            comp_lines.append(f"'{kw}': **{domain2}** uses it more ({kstats2[i]['count']} vs {kstats1[i]['count']})")
                        else:
                            comp_lines.append(f"'{kw}': Both sites use it equally ({kstats1[i]['count']})")
                    st.subheader("Keyword Comparison Summary")
                    for line in comp_lines:
                        st.write("- " + line)
                else:
                    st.info("No keywords provided. Add comma-separated keywords to see counts & density.")

                st.markdown("---")

                # --- Visible Text sample ---
                st.header("Sample Extracted Visible Text")
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    st.subheader(domain1)
                    with st.expander("Show extracted text (first 20k chars)"):
                        excerpt = site1["text"][:20000] + ("..." if len(site1["text"]) > 20000 else "")
                        st.write(excerpt or "(no visible text)")
                with tcol2:
                    st.subheader(domain2)
                    with st.expander("Show extracted text (first 20k chars)"):
                        excerpt = site2["text"][:20000] + ("..." if len(site2["text"]) > 20000 else "")
                        st.write(excerpt or "(no visible text)")

                st.markdown("---")

                # --- Final Recommendations & Full Summary ---
                st.header("Full Summary & Recommendation")

                # Scores and explanations
                s1_score, s1_info = compute_overall_score(site1, keywords)
                s2_score, s2_info = compute_overall_score(site2, keywords)

                # create bullet points of key strengths/weaknesses
                def key_points(domain, parsed, score):
                    pts = []
                    # title
                    tl = parsed.get("title", "")
                    if not tl:
                        pts.append("Missing title.")
                    else:
                        if 50 <= len(tl) <= 60:
                            pts.append("Title length is good.")
                        elif len(tl) < 50:
                            pts.append("Title is short.")
                        else:
                            pts.append("Title is long (may be truncated).")
                    # meta
                    md = parsed.get("meta_desc", "")
                    if not md:
                        pts.append("Missing meta description.")
                    else:
                        if 120 <= len(md) <= 160:
                            pts.append("Meta description length is good.")
                        elif len(md) < 120:
                            pts.append("Meta description is short.")
                        else:
                            pts.append("Meta description is long (may be truncated).")
                    # h1
                    if len(parsed.get("h1s", [])) == 0:
                        pts.append("No H1 found.")
                    elif len(parsed.get("h1s", [])) == 1:
                        pts.append("Single H1 present (good).")
                    else:
                        pts.append(f"{len(parsed.get('h1s'))} H1s found (multiple).")
                    # images
                    imgs = parsed.get("images", [])
                    if imgs:
                        missing = sum(1 for i in imgs if not i["alt"])
                        if missing:
                            pts.append(f"{missing} images missing alt text.")
                        else:
                            pts.append("All images have alt text.")
                    else:
                        pts.append("No images found — nothing to check for alt text.")
                    # schema
                    if parsed.get("schema"):
                        pts.append("Schema (JSON-LD) present.")
                    else:
                        pts.append("No JSON-LD schema detected.")
                    # keywords
                    if keywords:
                        present = [kw for kw in keywords if count_keyword_in_text(parsed.get("text", ""), kw) > 0]
                        if present:
                            pts.append(f"Keywords present in page text: {', '.join(present)}")
                        else:
                            pts.append("None of the provided keywords appear in the page text.")
                    else:
                        pts.append("No keywords provided.")
                    pts.append(f"Computed score: {score}/100")
                    return pts

                s1_points = key_points(domain1, site1, s1_score)
                s2_points = key_points(domain2, site2, s2_score)

                # Show side-by-side final summary
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    st.subheader(domain1)
                    for p in s1_points:
                        st.write("- " + p)
                with fcol2:
                    st.subheader(domain2)
                    for p in s2_points:
                        st.write("- " + p)

                st.markdown("---")

                # Decide winner and list top differences
                winner = None
                if s1_score > s2_score:
                    winner = domain1
                elif s2_score > s1_score:
                    winner = domain2
                else:
                    winner = "Tie"

                st.markdown("### Recommendation / Verdict")
                if winner == "Tie":
                    st.info(f"Scores tie at {s1_score}. Both pages are similar overall — choose improvements based on business goals.")
                else:
                    st.success(f"**{winner}** scores higher ({s1_score if winner==domain1 else s2_score}) and is the better on-page SEO match according to this quick audit.")

                # highlight top 5 actionable points to improve (based on differences)
                actionable = []
                # Titles & metas
                if len(site1["title"]) == 0:
                    actionable.append(f"{domain1}: Add a descriptive title (50-60 chars) containing primary keyword.")
                if len(site2["title"]) == 0:
                    actionable.append(f"{domain2}: Add a descriptive title (50-60 chars) containing primary keyword.")
                if len(site1["meta_desc"]) == 0:
                    actionable.append(f"{domain1}: Add a meta description (120-160 chars).")
                if len(site2["meta_desc"]) == 0:
                    actionable.append(f"{domain2}: Add a meta description (120-160 chars).")

                # image alt
                miss1 = sum(1 for i in site1["images"] if not i["alt"])
                miss2 = sum(1 for i in site2["images"] if not i["alt"])
                if miss1:
                    actionable.append(f"{domain1}: Add alt text to {miss1} image(s).")
                if miss2:
                    actionable.append(f"{domain2}: Add alt text to {miss2} image(s).")

                # schema
                if not site1["schema"]:
                    actionable.append(f"{domain1}: Consider adding JSON-LD schema for important entities (Organization, Product, Article).")
                if not site2["schema"]:
                    actionable.append(f"{domain2}: Consider adding JSON-LD schema for important entities (Organization, Product, Article).")

                # keywords
                if keywords:
                    for kw in keywords:
                        c1 = count_keyword_in_text(site1["text"], kw)
                        c2 = count_keyword_in_text(site2["text"], kw)
                        if c1 == 0 and c2 == 0:
                            actionable.append(f"Content: Consider adding content for keyword '{kw}' on both pages.")
                        elif c1 == 0:
                            actionable.append(f"{domain1}: No usage of '{kw}' — consider adding a section or sentence covering it.")
                        elif c2 == 0:
                            actionable.append(f"{domain2}: No usage of '{kw}' — consider adding a section or sentence covering it.")

                # dedupe and show top 7 actionables
                actionable = list(dict.fromkeys(actionable))[:7]
                if actionable:
                    st.subheader("Top Actionable Items")
                    for a in actionable:
                        st.write("- " + a)
                else:
                    st.success("No immediate action required from this quick audit.")

                st.markdown("---")
                st.caption("Note: This is a quick on-page comparison. For full SEO decisions, combine this with backlink analysis, traffic data, and business relevance.")
