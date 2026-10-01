# JV-Link ingestion worker (planned)

JRA-VAN Data Lab data is only available through JV-Link, a Windows COM component.
This directory will hold a small worker that runs on Windows, pulls data through
JV-Link, and writes parquet files in the runner-table format defined in
`src/aikeiba/schema.py`. Everything downstream runs on Linux/Docker and never
talks to JV-Link directly.

Requirements to settle before implementing (see `ROADMAP.md`, Phase 2):

- JRA-VAN Data Lab subscription and its terms for storing and processing data.
- 32-bit vs 64-bit Python for the COM component, and the COM bridge (`pywin32`).
- Which record types are needed first (race results, runners, odds, weights).
- Where the parquet files go (a DVC remote; never this Git repository).
- When each value becomes available, so `available_at` can be filled in
  (e.g. horse weight about an hour before post time, odds up to the close).

TARGET frontier can export CSV and is an acceptable manual source while this
worker does not exist. Check a site's terms of service before scraping it.
