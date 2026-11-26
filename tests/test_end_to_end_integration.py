"""
End-to-End System Integration Test
===================================
Project: Oral Cancer Screening Using Vision Transformer and Graph Attention Network with Explainable AI

Validates the complete workflow:
User Register -> Login -> Image Upload -> ViT+GATv2 Screening -> XAI Generation -> PDF Report -> History -> Deletion
"""

import os
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, os.getcwd())
from backend.database import DatabaseManager
from backend.inference_engine import OralScanEngine
from backend.pdf_generator import ScreeningPDFGenerator
from backend.app import (
    register, login, screen_oral_image, get_screening_record,
    download_pdf_report, get_user_history, delete_screening,
    UserRegisterRequest, UserLoginRequest
)


class MockUploadFile:
    """Mock UploadFile for direct endpoint testing without network."""
    def __init__(self, filepath: str):
        self.filename = Path(filepath).name
        self.file = open(filepath, "rb")

    def __del__(self):
        try:
            self.file.close()
        except Exception:
            pass


class TestEndToEndSystemIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = OralScanEngine.get_instance()
        cls.db = DatabaseManager()
        cls.pdf_gen = ScreeningPDFGenerator()
        cls.test_image_cancer = "Oral Cancer/Oral Cancer Dataset/CANCER/005.jpeg"
        cls.test_image_benign = "Oral Cancer/Oral Cancer Dataset/NON CANCER/007.jpeg"

    def test_complete_user_screening_workflow(self):
        # 1. Register new clinician account
        uname = f"clinician_{uuid.uuid4().hex[:6]}"
        email = f"{uname}@hospital.org"
        pwd = "ClinicalPassword2026!"
        reg_res = register(UserRegisterRequest(username=uname, email=email, password=pwd))
        self.assertTrue(reg_res["success"])
        token = reg_res["token"]
        user = reg_res["user"]

        # 2. Authenticate login
        login_res = login(UserLoginRequest(username_or_email=uname, password=pwd))
        self.assertTrue(login_res["success"])

        # 3. Upload and screen an oral lesion image
        mock_file = MockUploadFile(self.test_image_cancer)
        import asyncio
        screen_res = asyncio.run(screen_oral_image(
            file=mock_file,
            run_perturbation=False,
            current_user=user
        ))

        self.assertIn("screening_id", screen_res)
        sid = screen_res["screening_id"]
        self.assertIn(screen_res["pred_label"], [0, 1])
        self.assertIn("Suspicious pattern detected" if screen_res["pred_label"] == 1 else "No suspicious pattern detected", screen_res["cautious_result"])
        self.assertGreaterEqual(screen_res["cancer_score"], 0.0)

        # Verify XAI visualizations
        viz = screen_res["visualizations"]
        self.assertTrue(os.path.exists(viz["rollout_path"]))
        self.assertTrue(os.path.exists(viz["graph_path"]))

        # Verify Fusion branch sensitivity
        fusion = screen_res["fusion_attribution"]
        self.assertGreater(fusion["vit_share_pct"], 0.0)
        self.assertGreater(fusion["gat_share_pct"], 0.0)
        self.assertAlmostEqual(fusion["vit_share_pct"] + fusion["gat_share_pct"], 100.0, places=1)

        # 4. Retrieve screening record through API
        record = get_screening_record(sid, current_user=user)
        self.assertEqual(record["id"], sid)
        self.assertEqual(record["user_id"], user["id"])

        # 5. Generate and download PDF report
        pdf_response = download_pdf_report(sid, current_user=user)
        pdf_file = pdf_response.path
        self.assertTrue(os.path.exists(pdf_file))
        self.assertGreater(os.path.getsize(pdf_file), 5000)

        # 6. Verify screening appears in history
        history = get_user_history(limit=10, current_user=user)
        self.assertTrue(any(r["id"] == sid for r in history))

        # 7. Delete screening record
        del_res = delete_screening(sid, current_user=user)
        self.assertTrue(del_res["success"])

        # 8. Confirm record is deleted
        history_after = get_user_history(limit=10, current_user=user)
        self.assertFalse(any(r["id"] == sid for r in history_after))


if __name__ == "__main__":
    unittest.main()
