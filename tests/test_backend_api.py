"""
Comprehensive Unit & Integration Test Suite for Backend & API
============================================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI
"""

import os
import sys
import unittest
import uuid
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.getcwd())
from backend.database import DatabaseManager
from backend.inference_engine import OralScanEngine, CHECKPOINTS_CONFIG, compute_file_sha256
from backend.pdf_generator import ScreeningPDFGenerator
from backend.app import (
    health_check, model_status, register, login, logout,
    get_screening_record, get_user_history, delete_screening,
    UserRegisterRequest, UserLoginRequest
)


class TestBackendDatabase(unittest.TestCase):
    def setUp(self):
        # Generate a unique test database per test method to prevent Windows file lock issues
        self.test_db_path = Path(f"data/test_oralscan_{uuid.uuid4().hex[:8]}.db")
        self.db = DatabaseManager(db_path=self.test_db_path)

    def tearDown(self):
        # Clean up database file after garbage collection
        import gc
        gc.collect()
        if self.test_db_path.exists():
            try:
                self.test_db_path.unlink()
            except PermissionError:
                pass

    def test_user_registration_and_authentication(self):
        # Register user
        user = self.db.register_user("testdoc", "testdoc@example.com", "SecretPass123!")
        self.assertEqual(user["username"], "testdoc")
        self.assertEqual(user["email"], "testdoc@example.com")
        self.assertIsInstance(user["id"], int)

        # Duplicate username should fail
        with self.assertRaises(ValueError):
            self.db.register_user("testdoc", "other@example.com", "Password123!")

        # Duplicate email should fail
        with self.assertRaises(ValueError):
            self.db.register_user("otherdoc", "testdoc@example.com", "Password123!")

        # Successful authentication
        auth_user = self.db.authenticate_user("testdoc", "SecretPass123!")
        self.assertIsNotNone(auth_user)
        self.assertEqual(auth_user["id"], user["id"])

        # Authenticate by email
        auth_email = self.db.authenticate_user("testdoc@example.com", "SecretPass123!")
        self.assertIsNotNone(auth_email)

        # Failed authentication
        bad_auth = self.db.authenticate_user("testdoc", "WrongPassword")
        self.assertIsNone(bad_auth)

    def test_session_management(self):
        user = self.db.register_user("sessionuser", "session@example.com", "Password123!")
        token = self.db.create_session(user["id"])
        self.assertEqual(len(token), 64)  # 32 bytes in hex

        active_user = self.db.get_user_by_session(token)
        self.assertIsNotNone(active_user)
        self.assertEqual(active_user["username"], "sessionuser")

        # Delete session
        self.db.delete_session(token)
        revoked_user = self.db.get_user_by_session(token)
        self.assertIsNone(revoked_user)

    def test_screening_persistence_and_user_isolation(self):
        user1 = self.db.register_user("user1", "u1@example.com", "Pass12345!")
        user2 = self.db.register_user("user2", "u2@example.com", "Pass12345!")

        sid = str(uuid.uuid4())
        self.db.save_screening(
            screening_id=sid,
            user_id=user1["id"],
            original_filename="lesion_01.jpg",
            stored_image_path="uploads/test.jpg",
            pred_label=1,
            pred_name="CANCER",
            cautious_result="Suspicious pattern detected by the model.",
            cancer_score=85.4,
            confidence=85.4,
            vit_share_pct=64.2,
            gat_share_pct=35.8,
            ig_abs_error=0.15,
            ig_rel_error_pct=18.2,
            narrative="Model observed elevated visual attention.",
            has_perturbation=False
        )

        # User 1 should access their screening
        rec1 = self.db.get_screening(sid, user1["id"])
        self.assertIsNotNone(rec1)
        self.assertEqual(rec1["pred_name"], "CANCER")
        self.assertEqual(rec1["cancer_score"], 85.4)

        # User 2 MUST NOT access User 1's screening (User isolation check)
        rec2 = self.db.get_screening(sid, user2["id"])
        self.assertIsNone(rec2, "Security violation: User 2 accessed User 1's screening record!")

        # User stats
        stats1 = self.db.get_user_stats(user1["id"])
        self.assertEqual(stats1["total_screenings"], 1)
        self.assertEqual(stats1["suspicious_count"], 1)
        self.assertEqual(stats1["benign_count"], 0)


class TestInferenceEngineAndCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = OralScanEngine.get_instance()

    def test_checkpoint_hashes(self):
        for name, cfg in CHECKPOINTS_CONFIG.items():
            path = cfg["path"]
            expected = cfg["expected_sha256"]
            self.assertTrue(os.path.exists(path), f"Checkpoint missing: {path}")
            measured = compute_file_sha256(path)
            self.assertEqual(measured, expected, f"Hash mismatch on {name} checkpoint!")

    def test_model_status_metadata(self):
        status = self.engine.get_status()
        self.assertTrue(status["operational"])
        self.assertEqual(status["architecture"]["vit_parameters"], 1965890)
        self.assertEqual(status["architecture"]["gatv2_parameters"], 222722)
        self.assertEqual(status["architecture"]["fusion_parameters"], 49794)
        self.assertEqual(status["architecture"]["total_parameters"], 2238406)

    def test_screening_execution_on_validation_image(self):
        test_img = "Oral Cancer/Oral Cancer Dataset/CANCER/005.jpeg"
        self.assertTrue(os.path.exists(test_img), "Validation test image missing!")

        res = self.engine.run_screening(test_img, run_perturbation=False)
        self.assertIn("screening_id", res)
        self.assertIn(res["pred_label"], [0, 1])
        self.assertIn("cautious_result", res)
        self.assertIn("Suspicious pattern detected" if res["pred_label"] == 1 else "No suspicious pattern detected", res["cautious_result"])
        self.assertGreaterEqual(res["cancer_score"], 0.0)
        self.assertLessEqual(res["cancer_score"], 100.0)

        # Attribution shares sum to 100%
        fusion_attr = res["fusion_attribution"]
        total_share = fusion_attr["vit_share_pct"] + fusion_attr["gat_share_pct"]
        self.assertAlmostEqual(total_share, 100.0, places=1)

        # Visual overlays generated
        viz = res["visualizations"]
        self.assertTrue(os.path.exists(viz["rollout_path"]))
        self.assertTrue(os.path.exists(viz["graph_path"]))


class TestPDFReportGeneration(unittest.TestCase):
    def setUp(self):
        self.pdf_gen = ScreeningPDFGenerator(output_dir=Path("outputs/test_reports"))

    def tearDown(self):
        shutil_path = Path("outputs/test_reports")
        if shutil_path.exists():
            import shutil
            shutil.rmtree(shutil_path)

    def test_generate_pdf_report(self):
        sample_data = {
            "id": "test-screening-1234",
            "pred_label": 1,
            "pred_name": "CANCER",
            "cautious_result": "Suspicious pattern detected by the model.",
            "cancer_score": 78.5,
            "confidence": 78.5,
            "vit_share_pct": 65.2,
            "gat_share_pct": 34.8,
            "ig_abs_error": 0.12,
            "ig_rel_error_pct": 14.5,
            "narrative": "Visual attention focused along peripheral margins.\nGraph connections highlighted mucosal clusters.",
            "created_at": "2026-11-24 20:00:00 UTC"
        }
        test_img = "Oral Cancer/Oral Cancer Dataset/CANCER/005.jpeg"
        pdf_path = self.pdf_gen.generate_report(sample_data, raw_image_path=test_img)

        self.assertTrue(os.path.exists(pdf_path))
        filesize = os.path.getsize(pdf_path)
        self.assertGreater(filesize, 5000, "PDF filesize unusually small, check rendering integrity!")


class TestAPIEndpoints(unittest.TestCase):
    def test_health_and_status(self):
        h = health_check()
        self.assertEqual(h["status"], "healthy")

        s = model_status()
        self.assertTrue(s["operational"])
        self.assertEqual(s["architecture"]["total_parameters"], 2238406)

    def test_auth_workflow(self):
        uname = f"apiuser_{uuid.uuid4().hex[:6]}"
        email = f"{uname}@example.com"
        reg_req = UserRegisterRequest(username=uname, email=email, password="MySecurePassword1!")
        reg_res = register(reg_req)
        self.assertTrue(reg_res["success"])
        self.assertIn("token", reg_res)

        login_req = UserLoginRequest(username_or_email=uname, password="MySecurePassword1!")
        login_res = login(login_req)
        self.assertTrue(login_res["success"])
        self.assertIn("token", login_res)


if __name__ == "__main__":
    unittest.main()
