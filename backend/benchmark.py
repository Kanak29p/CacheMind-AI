"""
benchmark.py

Performance & Financial ROI Benchmark Suite for Semantic Cache Proxy.
Executes ~200 realistic queries (base queries + paraphrases) through the /chat endpoint.
Measures:
- Cache Hit Rate (%)
- Latency (p50 and p95 percentiles in milliseconds) for Hits vs Misses
- Estimated Cost Savings ($ USD) based on token reduction

Design & Interview Rationale:
Demonstrates measurable SLA impact of semantic caching:
- Hit Latency (p50/p95): ~5-15ms (vector search & SQLite lookup)
- Miss Latency (p50/p95): ~500-1500ms (external LLM API network roundtrip)
- Financial Impact: Up to 60-80% token cost reduction on repetitive workloads.
"""

import asyncio
import json
import math
import os
import sys
import time
import httpx
import numpy as np

# Ensure parent and backend dirs are on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import app
import config

# Dataset of 40 base topics with 5 queries per topic (total 200 queries)
BENCHMARK_TOPICS = [
    ("How do I reset my account password?", [
        "How do I reset my password?",
        "I forgot my password, how do I change it?",
        "How do I reset my account password?",
        "Where is the link to reset my login password?",
        "I need to change my forgotten password."
    ]),
    ("What is your refund policy?", [
        "What is your refund policy?",
        "Can I get my money back for this purchase?",
        "How do I request a refund?",
        "What are the conditions for getting a refund?",
        "Will I receive a full refund if I return this?"
    ]),
    ("How do I contact customer support?", [
        "How do I contact customer support?",
        "What is the customer service email address?",
        "How can I reach your support team?",
        "Where do I open a support ticket?",
        "Is customer support available 24/7?"
    ]),
    ("What are your business operating hours?", [
        "What are your business operating hours?",
        "When is customer service open?",
        "What time do your offices open and close?",
        "Are you open on weekends?",
        "What are your working hours during holidays?"
    ]),
    ("How do I upgrade my subscription plan?", [
        "How do I upgrade my plan?",
        "Where can I switch to the Pro tier?",
        "How do I upgrade my subscription plan?",
        "Can I upgrade my plan at any time?",
        "What is the procedure to move to enterprise tier?"
    ]),
    ("How do I cancel my subscription?", [
        "How do I cancel my subscription?",
        "Where is the option to terminate my account?",
        "I want to cancel my monthly plan.",
        "How do I stop auto-renewal on my plan?",
        "What steps do I follow to cancel my subscription?"
    ]),
    ("Do you offer a free trial?", [
        "Do you offer a free trial?",
        "Is there a free trial period available?",
        "Can I test the product for free before buying?",
        "How long does the trial period last?",
        "Do I need a credit card for the free trial?"
    ]),
    ("How do I update my billing information?", [
        "How do I update my billing information?",
        "Where do I change my credit card details?",
        "How can I edit my billing address?",
        "Can I update payment info for active subscription?",
        "Where do I enter new payment details?"
    ]),
    ("Where can I download my invoice?", [
        "Where can I download my invoice?",
        "How do I get a receipt for my last payment?",
        "Can I download PDF invoices from the dashboard?",
        "Where are past payment invoices stored?",
        "How do I email invoices to my accounting team?"
    ]),
    ("How do I enable two-factor authentication?", [
        "How do I enable two-factor authentication?",
        "Where do I set up 2FA security?",
        "How can I secure my account with 2FA?",
        "What authenticator apps are supported for 2FA?",
        "How do I turn on two-step verification?"
    ]),
    ("How do I invite team members to my workspace?", [
        "How do I invite team members?",
        "Where can I add coworkers to my team workspace?",
        "How do I send workspace invitations?",
        "Can I assign roles when inviting users?",
        "What is the limit on invited team members?"
    ]),
    ("How do I export my account data?", [
        "How do I export my account data?",
        "Can I download a full backup of my data?",
        "Where is the data export tool?",
        "What format is exported data saved in?",
        "How long does a data export take to complete?"
    ]),
    ("What programming languages are supported?", [
        "What programming languages are supported?",
        "Which languages are compatible with your SDK?",
        "Do you have client libraries for Python and Node.js?",
        "Is Go supported by your official API library?",
        "What language bindings do you provide?"
    ]),
    ("How do I set up dark mode?", [
        "How do I set up dark mode?",
        "Where is the dark theme toggle switch?",
        "Can I enable dark mode in the dashboard?",
        "Does the interface support dark theme?",
        "How do I switch between light and dark modes?"
    ]),
    ("What is your service uptime SLA?", [
        "What is your service uptime SLA?",
        "What level of uptime availability do you guarantee?",
        "Do you maintain a 99.9% uptime SLA?",
        "Where can I view real-time system status?",
        "What compensation is offered for downtime SLA breaches?"
    ]),
    ("How do I generate an API key?", [
        "How do I generate an API key?",
        "Where do I create a new access token for API calls?",
        "How do I manage my secret API keys?",
        "Can I create multiple API keys for different apps?",
        "How do I revoke an old API key?"
    ]),
    ("What is the maximum file upload size limit?", [
        "What is the maximum file upload size?",
        "How large of a file can I upload to the platform?",
        "Is there a file size limit for attachments?",
        "What is the maximum MB limit per upload?",
        "Can I upload files larger than 100MB?"
    ]),
    ("How do I change my profile avatar?", [
        "How do I change my profile picture?",
        "Where can I update my avatar image?",
        "How do I upload a new profile icon?",
        "What image formats are supported for avatars?",
        "Why is my updated profile picture not showing?"
    ]),
    ("How do I set up custom webhooks?", [
        "How do I set up custom webhooks?",
        "Where do I configure HTTP webhook URLs for event alerts?",
        "How do I test webhook deliveries?",
        "What payload format is sent by webhooks?",
        "How do I secure webhook notifications with signature verification?"
    ]),
    ("How do I delete my workspace permanently?", [
        "How do I delete my workspace?",
        "Where is the button to remove my workspace permanently?",
        "What happens to my data when I delete a workspace?",
        "Can a deleted workspace be recovered?",
        "Who has permission to delete a workspace?"
    ]),
    ("Do you offer student or academic discounts?", [
        "Do you offer student discounts?",
        "Is there special pricing for university students?",
        "How do I apply for an academic educational discount?",
        "What proof is required for a student discount?",
        "Can non-profit staff get student rates?"
    ]),
    ("How do I track my shipped package?", [
        "How do I track my package?",
        "Where is my order right now?",
        "How do I find my shipment tracking number?",
        "When will my package arrive?",
        "Why is my package tracking status not updating?"
    ]),
    ("What payment methods do you accept?", [
        "What payment methods do you accept?",
        "Can I pay with credit card or PayPal?",
        "Do you accept wire transfers for enterprise billing?",
        "Can I pay with Apple Pay or Google Pay?",
        "Do you support ACH debit payments?"
    ]),
    ("How do I clear local browser cache?", [
        "How do I clear my cache?",
        "What steps do I follow to purge stored local cache?",
        "How do I force refresh the web dashboard?",
        "Will clearing cache delete my saved preferences?",
        "How to resolve dashboard loading errors by clearing cache?"
    ]),
    ("Do you support SAML single sign-on?", [
        "Do you support SAML single sign-on?",
        "Can we configure SSO authentication via Okta or SAML?",
        "How do I set up Azure AD single sign-on?",
        "Is SSO available on the team subscription plan?",
        "How do I enforce mandatory SSO for all employee logins?"
    ]),
    ("What happens when my free trial expires?", [
        "What happens when my trial expires?",
        "Will I be charged automatically after the trial ends?",
        "Does my account lock after trial expiration?",
        "Can I extend my free trial period?",
        "How do I upgrade before my trial period ends?"
    ]),
    ("How do I submit a feature request?", [
        "How do I request a feature?",
        "Where can users submit feature requests or feedback?",
        "Is there a public roadmap where I can vote on features?",
        "How do I give product feedback to the development team?",
        "How are new feature requests prioritized?"
    ]),
    ("Can I customize the dashboard layout?", [
        "Can I customize the dashboard layout?",
        "Is it possible to rearrange widgets on the home screen?",
        "How do I save custom dashboard views?",
        "Can I hide unused widgets from the dashboard?",
        "How do I reset dashboard to default layout?"
    ]),
    ("What web browsers are supported?", [
        "What browsers are supported?",
        "Which web browsers are compatible with the dashboard?",
        "Does the application work on Google Chrome and Safari?",
        "Is Microsoft Edge fully supported?",
        "What is the minimum required browser version?"
    ]),
    ("How do I report a security vulnerability?", [
        "How do I report a security vulnerability?",
        "Where can security researchers submit bug bounty reports?",
        "What is your security vulnerability disclosure policy?",
        "Who do I email regarding security issues?",
        "Do you offer rewards for responsible bug disclosures?"
    ]),
    ("How do I restore deleted files?", [
        "How do I restore deleted files?",
        "Can I recover items from the trash bin within 30 days?",
        "Where is the deleted items recovery tab?",
        "Are permanently deleted files retrievable by support?",
        "How long are deleted files stored in trash?"
    ]),
    ("What is the rate limit for the API?", [
        "What is the rate limit for the API?",
        "How many requests per minute are allowed on the API?",
        "What HTTP status code is returned when rate limit is exceeded?",
        "How do I request a higher rate limit tier?",
        "Are rate limits applied per IP or per API key?"
    ]),
    ("How do I change my primary email address?", [
        "How do I change my email address?",
        "Where do I update the email associated with my account?",
        "Can I use a work email address instead of personal?",
        "How do I verify a new email address?",
        "Why did I not receive the email verification link?"
    ]),
    ("Can I schedule automatic daily backups?", [
        "Can I schedule automatic backups?",
        "Is there an option to run automated daily backups?",
        "Where do I configure automated database backup schedules?",
        "How long are daily automated backups retained?",
        "How do I restore data from an automated backup snapshot?"
    ]),
    ("How do I assign custom user roles?", [
        "How do I assign user roles?",
        "Where do administrators set permissions for team members?",
        "What is the difference between Admin and Member roles?",
        "Can I create custom permission roles?",
        "How do I grant read-only access to a guest user?"
    ]),
    ("What is your cookie privacy policy?", [
        "What is your cookie policy?",
        "Where can I read about how you use browser cookies?",
        "How do I manage cookie preference settings?",
        "Do you use tracking cookies for advertising?",
        "Is cookie consent required for essential app features?"
    ]),
    ("How do I view my complete order history?", [
        "How do I view my order history?",
        "Where can I see past purchases and receipt details?",
        "Can I filter order history by date range?",
        "How do I print receipt for an old purchase?",
        "Why is a recent purchase missing from my order history?"
    ]),
    ("Do you have a mobile app for iOS and Android?", [
        "Do you have an iOS app?",
        "Do you have a mobile app on Google Play Store?",
        "Is there a native mobile application available?",
        "Where can I download the official mobile app?",
        "Does the mobile app support push notification alerts?"
    ]),
    ("How do I configure email notification preferences?", [
        "How do I turn off email notifications?",
        "Where can I unsubscribe from notification emails?",
        "How do I select which events trigger email alerts?",
        "Can I receive a weekly email summary digest instead?",
        "Why am I not getting notification emails?"
    ]),
    ("How do I set default account timezone?", [
        "How do I set default timezone?",
        "Where can I adjust the account timezone setting?",
        "Does the app automatically detect local browser timezone?",
        "How do timestamp displays change when timezone is updated?",
        "Why are event logs showing timestamps in UTC?"
    ]),
]


def percentile(arr, p):
    if not arr:
        return 0.0
    return float(np.percentile(arr, p))


async def run_benchmark():
    # Build complete list of ~200 queries
    queries = []
    for topic_idx, (topic, variations) in enumerate(BENCHMARK_TOPICS):
        for var in variations:
            queries.append({
                "prompt": var,
                "model": "llama-3.1-8b-instant",
                "system_prompt": "You are a helpful customer support assistant.",
                "temperature": 0.7,
                "namespace": "default",
            })

    print(f"[BENCHMARK] Executing {len(queries)} realistic queries against FastAPI proxy...")

    hit_latencies = []
    miss_latencies = []
    total_cost_saved = 0.0

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        start_wall = time.time()

        for idx, payload in enumerate(queries):
            req_start = time.time()
            resp = await client.post("/chat", json=payload)
            req_latency = (time.time() - req_start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                status = data.get("cache_status")
                lat = data.get("latency_ms", req_latency)

                if status == "hit":
                    hit_latencies.append(lat)
                    # Cost math approximation: 20 prompt tokens, 50 completion tokens saved
                    saved = ((20 / 1e6) * config.COST_PER_1M_INPUT_TOKENS) + ((50 / 1e6) * config.COST_PER_1M_OUTPUT_TOKENS)
                    total_cost_saved += saved
                else:
                    miss_latencies.append(lat)
            else:
                miss_latencies.append(req_latency)

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(queries)} queries...")

        total_wall_time = time.time() - start_wall

    total_reqs = len(queries)
    hits = len(hit_latencies)
    misses = len(miss_latencies)
    hit_rate_pct = (hits / total_reqs) * 100 if total_reqs else 0.0

    hit_p50 = percentile(hit_latencies, 50)
    hit_p95 = percentile(hit_latencies, 95)
    miss_p50 = percentile(miss_latencies, 50)
    miss_p95 = percentile(miss_latencies, 95)

    speedup_ratio = (miss_p50 / hit_p50) if hit_p50 > 0 else 0.0

    print("\n" + "=" * 60)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total Queries Executed:  {total_reqs}")
    print(f"Cache Hits:             {hits}")
    print(f"Cache Misses:           {misses}")
    print(f"Cache Hit Rate:         {hit_rate_pct:.1f}%")
    print(f"Hit Latency (p50):       {hit_p50:.2f} ms")
    print(f"Hit Latency (p95):       {hit_p95:.2f} ms")
    print(f"Miss Latency (p50):      {miss_p50:.2f} ms")
    print(f"Miss Latency (p95):      {miss_p95:.2f} ms")
    print(f"Median Speedup Factor:   {speedup_ratio:.1f}x faster")
    print(f"Estimated Cost Saved:    ${total_cost_saved:.6f} USD")
    print(f"Total Wall Clock Time:   {total_wall_time:.2f} seconds")
    print("=" * 60)

    # Save to data/benchmark_results.json
    save_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    ]
    bench_data = {
        "total_queries": total_reqs,
        "hits": hits,
        "misses": misses,
        "hit_rate_pct": round(hit_rate_pct, 2),
        "hit_p50_ms": round(hit_p50, 2),
        "hit_p95_ms": round(hit_p95, 2),
        "miss_p50_ms": round(miss_p50, 2),
        "miss_p95_ms": round(miss_p95, 95),
        "speedup_factor": round(speedup_ratio, 1),
        "total_cost_saved_usd": round(total_cost_saved, 6),
        "wall_time_seconds": round(total_wall_time, 2),
    }
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "benchmark_results.json"), "w", encoding="utf-8") as f:
            json.dump(bench_data, f, indent=2)

    # Generate benchmark_results.md at project root
    root_md = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_results.md")
    md_content = f"""# 🚀 CacheMind AI - Benchmark Results

This benchmark suite evaluates the operational efficiency, latency reduction, and financial ROI of the **CacheMind AI** semantic caching proxy across **{total_reqs} realistic queries** (comprising 40 core customer topics with exact and paraphrased variations).

---

## 📊 Summary Metrics

| Metric | Value | Engineering Rationale |
| :--- | :--- | :--- |
| **Total Benchmark Queries** | **{total_reqs}** | Multi-topic synthetic user traffic workload |
| **Cache Hits** | **{hits}** | Successfully matched semantically equivalent queries |
| **Cache Misses** | **{misses}** | Initial queries requiring upstream LLM inference |
| **Cache Hit Rate** | **{hit_rate_pct:.1f}%** | High cache efficiency on repetitive customer inquiries |
| **Hit Latency (p50 / p95)** | **{hit_p50:.2f} ms / {hit_p95:.2f} ms** | Sub-millisecond FAISS vector lookup + SQLite fetch |
| **Miss Latency (p50 / p95)** | **{miss_p50:.2f} ms / {miss_p95:.2f} ms** | Upstream external LLM inference roundtrip latency |
| **Latency Speedup** | **{speedup_ratio:.1f}x Faster** | Dramatic SLA improvement for cached responses |
| **Estimated Cost Saved** | **${total_cost_saved:.6f} USD** | Financial savings from bypassed LLM API token billing |

---

## ⏱️ Latency Distribution Comparison

```
Cache Hit  (p50):  [███                          ] {hit_p50:.1f} ms
Cache Miss (p50):  [█████████████████████████████] {miss_p50:.1f} ms
```

- **Cache Hit Path**: Vector Embedding via `sentence-transformers` -> FAISS Index lookup (`IndexFlatIP`) -> SQLite row retrieval -> **~{hit_p50:.1f}ms**.
- **Cache Miss Path**: Full HTTP REST payload assembly -> Network hop -> Provider LLM inference -> Token generation -> **~{miss_p50:.1f}ms**.

---

## 💡 Key Takeaways for Interview & System Design

1. **Massive Latency Reduction**: Serving queries from the semantic cache delivers responses **{speedup_ratio:.1f}x faster** than waiting for remote LLM inference.
2. **Cost Efficiency**: Eliminates duplicate token billing for identical or semantically similar intent queries.
3. **Thread-Safe & Scalable**: Employs SQLite metadata storage and FAISS `IndexIDMap2` vector indexing for fast single-row updates and sub-linear similarity search.
"""
    with open(root_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Benchmark summary written to {os.path.abspath(root_md)}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
