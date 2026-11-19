import os
import json
import tempfile
import unittest
import psycopg2

from batch_job import find_contiguous_watermark
from monitor import read_new_ground_truth, sample_active_events


class TestWatermark(unittest.TestCase):
    def test_empty_and_missing_dir(self):
        self.assertEqual(find_contiguous_watermark("/tmp/nonexistent_lambda_dir_xyz"), (0, []))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(find_contiguous_watermark(d), (0, []))

    def test_contiguous_gap_overlap_and_ignored_files(self):
        with tempfile.TemporaryDirectory() as d:
            names = [
                "part_00000001_00000010_1000.parquet",
                "part_00000005_00000020_1001.parquet",  # overlaps
                "part_00000021_00000030_1002.parquet",  # contiguous
                "part_00000032_00000040_1003.parquet",  # gap at 31
                "ignore_me.tmp",
            ]
            for name in names:
                open(os.path.join(d, name), "w").close()

            h, files = find_contiguous_watermark(d)
            self.assertEqual(h, 30)
            self.assertEqual(len(files), 3)


class TestMonitorHelpers(unittest.TestCase):
    def test_partial_json_line_is_not_lost_or_crashed(self):
        with tempfile.TemporaryDirectory() as d:
            gt_path = os.path.join(d, "gt.jsonl")
            ev1 = {"event_id": "e1", "seq": 1, "produced_at": "2026-01-01T00:00:00+00:00", "amount_cents": 100}
            ev2 = {"event_id": "e2", "seq": 2, "produced_at": "2026-01-01T00:00:01+00:00", "amount_cents": 200}
            line2 = json.dumps(ev2)

            with open(gt_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(ev1) + "\n")
                f.write(line2[:15])  # truncated mid-line

            gt_events = {}
            offset, added = read_new_ground_truth(gt_path, 0, gt_events)
            self.assertEqual([e["event_id"] for e in added], ["e1"])

            with open(gt_path, "a", encoding="utf-8") as f:
                f.write(line2[15:] + "\n")

            offset, added2 = read_new_ground_truth(gt_path, offset, gt_events)
            self.assertEqual([e["event_id"] for e in added2], ["e2"])
            self.assertEqual(len(gt_events), 2)

    def test_sampling_skips_warmup_and_spreads_across_active_window(self):
        sampled = set()
        pending = {}
        warmup_evs = [
            {"event_id": f"w{i}", "seq": i, "produced_at": "2026-01-01T00:00:00+00:00"}
            for i in range(1, 50)
        ]
        next_t = sample_active_events(warmup_evs, elapsed=19.9, warmup=20, next_sample_elapsed=20.0,
                                      sample_step_sec=1.25, sampled_ids=sampled, pending_samples=pending)
        self.assertEqual(len(sampled), 0)
        self.assertEqual(next_t, 20.0)

        active_evs = [
            {"event_id": f"a{i}", "seq": 50 + i, "produced_at": "2026-01-01T00:00:20+00:00"}
            for i in range(10)
        ]
        next_t = sample_active_events(active_evs, elapsed=20.1, warmup=20, next_sample_elapsed=next_t,
                                      sample_step_sec=1.25, sampled_ids=sampled, pending_samples=pending)
        self.assertEqual(len(sampled), 1)
        self.assertAlmostEqual(next_t, 21.25)


class TestServingViews(unittest.TestCase):
    def test_views_deduplicate_at_watermark(self):
        conn_str = "dbname=lambda_serving user=lambda_user password=lambda_password host=localhost port=5432"
        try:
            conn = psycopg2.connect(conn_str, connect_timeout=2)
        except psycopg2.OperationalError:
            self.skipTest("PostgreSQL container not running")

        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM batch_views; DELETE FROM speed_events;")
                cur.execute("UPDATE watermark_meta SET h_seq = 2 WHERE id = 1;")
                cur.execute("INSERT INTO batch_views (category, cnt, sum_cents, h_seq) VALUES ('ZONE_1', 2, 300, 2);")
                cur.execute("""
                    INSERT INTO speed_events (event_id, seq, category, amount_cents, produced_at) VALUES
                    ('e1', 1, 'ZONE_1', 100, NOW()),
                    ('e2', 2, 'ZONE_1', 200, NOW()),
                    ('e3', 3, 'ZONE_1', 400, NOW()),
                    ('e4', 4, 'ZONE_2', 500, NOW());
                """)
                cur.execute("SELECT system_total_count, system_total_sum_cents, current_batch_h FROM v_q2_total;")
                total_cnt, total_sum, h_seq = cur.fetchone()
                self.assertEqual((total_cnt, total_sum, h_seq), (4, 1200, 2))
        finally:
            conn.rollback()
            conn.close()


if __name__ == "__main__":
    unittest.main()
