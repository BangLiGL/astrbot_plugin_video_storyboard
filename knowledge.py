"""Durable video evidence and group-scoped retrieval; no model-weight training."""

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


def terms(text):
    text = str(text).lower()
    result = set(re.findall(r"[a-z0-9]{3,}", text))
    for word in re.findall(r"[\u4e00-\u9fff]+", text):
        result.update(word[i : i + 2] for i in range(len(word) - 1))
    return result


class Knowledge:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS videos (
                    key TEXT PRIMARY KEY, record TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    scope TEXT NOT NULL, key TEXT NOT NULL, record TEXT NOT NULL, seen REAL NOT NULL,
                    PRIMARY KEY(scope,key)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def scope(origin):
        return hashlib.sha256(origin.encode()).hexdigest()

    def cached(self, key, hours=168):
        with self.connect() as db:
            row = db.execute(
                "SELECT record FROM videos WHERE key=? AND created>?",
                (key, time.time() - hours * 3600),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, key, record):
        # Media paths and credentials are never part of persistent chat knowledge.
        clean = {
            k: record[k]
            for k in (
                "bvid",
                "page",
                "title",
                "part_title",
                "uploader",
                "url",
                "duration",
                "understanding",
                "transcript_source",
                "warnings",
                "visual",
            )
            if k in record
        }
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO videos VALUES(?,?,?)",
                (key, json.dumps(clean, ensure_ascii=False), time.time()),
            )
        return clean

    def cached_subtitle(self, keys, hours=168):
        """Reuse only genuine subtitle results; never extend their original TTL."""
        with self.connect() as db:
            for key in keys:
                row = db.execute(
                    "SELECT record FROM videos WHERE key=? AND created>?",
                    (key, time.time() - hours * 3600),
                ).fetchone()
                if row:
                    record = json.loads(row[0])
                    if record.get("transcript_source") == "bilibili_subtitle":
                        return key, record
        return None

    def remember(self, origin, key, record):
        scope = self.scope(origin)
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO memories VALUES(?,?,?,?)",
                (scope, key, json.dumps(record, ensure_ascii=False), time.time()),
            )
            # Keep a bounded factual history per group, independent of media eviction.
            db.execute(
                "DELETE FROM memories WHERE scope=? AND key NOT IN (SELECT key FROM memories WHERE scope=? ORDER BY seen DESC LIMIT 100)",
                (scope, scope),
            )

    def retrieve(self, origin, text, limit=2):
        with self.connect() as db:
            rows = db.execute(
                "SELECT record,seen FROM memories WHERE scope=? ORDER BY seen DESC LIMIT 100",
                (self.scope(origin),),
            ).fetchall()
        followup = bool(
            re.search(
                r"刚才|刚刚|那个视频|这个视频|那段|这段|这个up|那个up|他为什么|她为什么", text, re.I
            )
        )
        query = terms(text)
        ranked = []
        for index, (raw, seen) in enumerate(rows):
            record = json.loads(raw)
            haystack = terms(record.get("title", "") + " " + record.get("understanding", ""))
            score = len(query & haystack)
            if score >= 2 or (followup and index < 2 and time.time() - seen < 1800):
                ranked.append((score + (1 if followup and index == 0 else 0), seen, record))
        return [
            item[2]
            for item in sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)[:limit]
        ]
