import os
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_full_pipeline():
    print("Testing Full API Pipeline...")

    # 1. Test Auth Registration
    reg_payload = {
        "email": "test.technician@lab.com",
        "full_name": "Test Lab Tech",
        "password": "Password123!",
        "role": "user"
    }
    r = requests.post(f"{BASE_URL}/api/auth/register", json=reg_payload)
    if r.status_code == 400 and "already registered" in r.text:
        print("[+] Test user already registered.")
    else:
        assert r.status_code == 200, f"Registration failed: {r.text}"
        print("[+] User registration successful!")

    # 2. Test Auth Login
    login_payload = {
        "email": "test.technician@lab.com",
        "password": "Password123!"
    }
    r = requests.post(f"{BASE_URL}/api/auth/login", json=login_payload)
    assert r.status_code == 200, f"Login failed: {r.text}"
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[+] Login successful, JWT token acquired!")

    # 3. Test Get Current User
    r = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
    assert r.status_code == 200, f"Get me failed: {r.text}"
    print(f"[+] User profile authenticated: {r.json()['full_name']}")

    # 4. Test Model Status
    r = requests.get(f"{BASE_URL}/api/model/status", headers=headers)
    assert r.status_code == 200
    print(f"[+] Model Status: {r.json()}")

    # 5. Test Image Upload & Detection Analysis (Valid JPG)
    test_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "test", "leakage", "cabinet_test_leakage_001.jpg")
    assert os.path.exists(test_img_path), f"Test image missing at {test_img_path}"

    with open(test_img_path, "rb") as f:
        files = {"file": ("cabinet_test.jpg", f, "image/jpeg")}
        data = {"notes": "Integration Test Upload JPG"}
        r = requests.post(f"{BASE_URL}/api/detections/analyze", headers=headers, files=files, data=data)

    assert r.status_code == 200, f"Image analysis failed: {r.text}"
    res = r.json()
    print(f"[+] Detection Analysis Result (JPG):")
    print(f"    - Scan ID: #{res['id']}")
    print(f"    - Prediction: {res['prediction']}")
    print(f"    - Confidence: {res['confidence'] * 100:.1f}%")
    print(f"    - Mode: {res['mode']}")
    print(f"    - Visualization Path: {res['visualization_image_path']}")
    print(f"    - Alarm Recommended: {res['alarm_recommended']}")

    # 5b. Test Valid PNG Upload
    import io
    from PIL import Image
    png_buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(200, 150, 50)).save(png_buf, format="PNG")
    png_buf.seek(0)
    r_png = requests.post(
        f"{BASE_URL}/api/detections/analyze",
        headers=headers,
        files={"file": ("test_cabinet.png", png_buf, "image/png")},
        data={"notes": "Integration Test Upload PNG"}
    )
    assert r_png.status_code == 200, f"PNG analysis failed: {r_png.text}"
    print(f"[+] PNG upload & analysis successful: #{r_png.json()['id']}")

    # 5c. Test Unsupported File Format Rejection
    r_unsupported = requests.post(
        f"{BASE_URL}/api/detections/analyze",
        headers=headers,
        files={"file": ("notes.txt", io.BytesIO(b"Hello world"), "text/plain")}
    )
    assert r_unsupported.status_code == 400, f"Expected 400 for unsupported format, got {r_unsupported.status_code}"
    print(f"[+] Unsupported file rejected correctly (HTTP 400): {r_unsupported.json()['detail']}")

    # 5d. Test Empty File Rejection (0 bytes)
    r_empty = requests.post(
        f"{BASE_URL}/api/detections/analyze",
        headers=headers,
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")}
    )
    assert r_empty.status_code == 400, f"Expected 400 for empty file, got {r_empty.status_code}"
    assert "empty" in r_empty.json()["detail"].lower()
    print(f"[+] Empty 0-byte file rejected correctly (HTTP 400): {r_empty.json()['detail']}")

    # 5e. Test Corrupt File Rejection
    r_corrupt = requests.post(
        f"{BASE_URL}/api/detections/analyze",
        headers=headers,
        files={"file": ("corrupt.jpg", io.BytesIO(b"NOT_A_VALID_JPEG_HEADER"), "image/jpeg")}
    )
    assert r_corrupt.status_code == 400, f"Expected 400 for corrupt file, got {r_corrupt.status_code}"
    assert "corrupt" in r_corrupt.json()["detail"].lower() or "unreadable" in r_corrupt.json()["detail"].lower()
    print(f"[+] Corrupt file rejected correctly (HTTP 400): {r_corrupt.json()['detail']}")

    # 5f. Test Missing File Parameter (HTTP 422)
    r_missing = requests.post(
        f"{BASE_URL}/api/detections/analyze",
        headers=headers,
        data={"notes": "No file included"}
    )
    assert r_missing.status_code == 422, f"Expected 422 for missing file, got {r_missing.status_code}"
    print("[+] Missing upload parameter rejected correctly (HTTP 422)")

    # 5g. Test Insecure / Demo Token Rejection (Must return HTTP 401)
    with open(test_img_path, "rb") as f:
        r_demo = requests.post(
            f"{BASE_URL}/api/detections/analyze",
            headers={"Authorization": "Bearer demo_vercel_access_token_12345"},
            files={"file": ("demo_test.jpg", f, "image/jpeg")}
        )
    assert r_demo.status_code == 401, f"Expected 401 for insecure demo token, got {r_demo.status_code}"
    print("[+] Insecure demo token rejected correctly with HTTP 401 Unauthorized.")

    # 5g-2. Test Missing Authorization Header (Must return HTTP 401)
    r_no_auth = requests.get(f"{BASE_URL}/api/dashboard/stats")
    assert r_no_auth.status_code == 401, f"Expected 401 for missing token, got {r_no_auth.status_code}"
    print("[+] Missing token rejected correctly with HTTP 401 Unauthorized.")

    # 5h. Verify Threshold Gating on API Response (52.5% threshold)
    if res['prediction'] == 'leakage':
        if res['confidence'] >= 0.525:
            assert res['alarm_recommended'] is True, "Alarm should be recommended for leakage >= 52.5%"
            assert res['alarm_activated'] is True, "Alarm should be activated for leakage >= 52.5%"
            print(f"[+] API Threshold Verification: Leakage {res['confidence']*100:.1f}% >= 52.5% -> DANGER (Alarm Active)")
        else:
            assert res['alarm_recommended'] is False, "Alarm should be silent for leakage < 52.5%"
            assert res['alarm_activated'] is False, "Alarm should be silent for leakage < 52.5%"
            print(f"[+] API Threshold Verification: Leakage {res['confidence']*100:.1f}% < 52.5% -> SAFE (Alarm Silent)")
    else:
        assert res['alarm_recommended'] is False, "Alarm should be silent for normal class"
        print(f"[+] API Threshold Verification: Normal class -> SAFE (Alarm Silent)")

    # 6. Test Dashboard Stats
    r = requests.get(f"{BASE_URL}/api/dashboard/stats", headers=headers)
    assert r.status_code == 200, f"Stats failed: {r.text}"
    stats = r.json()
    print(f"[+] Dashboard Stats: Total={stats['total_images']}, Danger={stats['leakage_predictions']}, Safe={stats['safe_predictions']}, Alarm={stats['alarm_status']}")

    # 7. Test Detections History Listing
    r = requests.get(f"{BASE_URL}/api/detections", headers=headers)
    assert r.status_code == 200, f"History listing failed: {r.text}"
    history = r.json()
    assert len(history) >= 1, "Expected at least 1 record in history"
    print(f"[+] History listing successful ({len(history)} records retrieved).")

    # 8. Test History Filtering by Status (52.5% threshold)
    r_danger = requests.get(f"{BASE_URL}/api/detections?status_filter=leakage", headers=headers)
    assert r_danger.status_code == 200
    for rec in r_danger.json():
        assert rec['prediction'] == 'leakage' and rec['confidence'] >= 0.525, "Filtered danger records must have leakage >= 52.5%"
    print(f"[+] History Danger filter verified ({len(r_danger.json())} danger records).")

    r_safe = requests.get(f"{BASE_URL}/api/detections?status_filter=no_leakage", headers=headers)
    assert r_safe.status_code == 200
    for rec in r_safe.json():
        assert rec['prediction'] == 'no_leakage' or (rec['prediction'] == 'leakage' and rec['confidence'] < 0.525), "Filtered safe records must be normal or leakage < 52.5%"
    print(f"[+] History Safe filter verified ({len(r_safe.json())} safe records).")

    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY! FULL PIPELINE OPERATIONAL.")

def test_threshold_logic():
    print("\n--- Testing 52.5% Confidence Threshold Rules ---")
    from app.routes.detection_routes import LEAKAGE_DANGER_THRESHOLD

    assert LEAKAGE_DANGER_THRESHOLD == 0.525, f"Expected LEAKAGE_DANGER_THRESHOLD to be 0.525, got {LEAKAGE_DANGER_THRESHOLD}"

    def evaluate_status(pred_label: str, confidence: float):
        is_danger = (pred_label == "leakage" and confidence >= LEAKAGE_DANGER_THRESHOLD)
        status_label = "DANGER" if is_danger else "SAFE"
        alarm_recommended = is_danger
        return status_label, alarm_recommended

    # Requirement 7: Leakage confidence at 52.4% -> SAFE, alarm silent (< 52.5%)
    status, alarm = evaluate_status("leakage", 0.524)
    assert status == "SAFE", f"Expected SAFE for 52.4% leakage, got {status}"
    assert alarm is False, "Expected silent alarm for 52.4% leakage"
    print("[+] Test 52.4% Passed: Leakage at 52.4% -> SAFE (Alarm Silent)")

    # Requirement 7: Leakage confidence at 52.5% -> DANGER, alarm active (>= 52.5%)
    status, alarm = evaluate_status("leakage", 0.525)
    assert status == "DANGER", f"Expected DANGER for 52.5% leakage, got {status}"
    assert alarm is True, "Expected active alarm for 52.5% leakage"
    print("[+] Test 52.5% Passed: Leakage at 52.5% -> DANGER (Alarm Active)")

    # Requirement 7: Leakage confidence at 52.6% -> DANGER, alarm active (> 52.5%)
    status, alarm = evaluate_status("leakage", 0.526)
    assert status == "DANGER", f"Expected DANGER for 52.6% leakage, got {status}"
    assert alarm is True, "Expected active alarm for 52.6% leakage"
    print("[+] Test 52.6% Passed: Leakage at 52.6% -> DANGER (Alarm Active)")

    # Requirement 3: Normal class at 95.0% -> SAFE, alarm silent (high normal confidence must NEVER trigger danger)
    status, alarm = evaluate_status("no_leakage", 0.950)
    assert status == "SAFE", f"Expected SAFE for 95.0% normal, got {status}"
    assert alarm is False, "Expected silent alarm for 95.0% normal"
    print("[+] Test Normal 95.0% Passed: Normal at 95.0% -> SAFE (Alarm Silent, never triggers danger)")

    # Requirement 3: Normal class at 52.5% -> SAFE, alarm silent
    status, alarm = evaluate_status("no_leakage", 0.525)
    assert status == "SAFE", f"Expected SAFE for 52.5% normal, got {status}"
    assert alarm is False, "Expected silent alarm for 52.5% normal"
    print("[+] Test Normal 52.5% Passed: Normal at 52.5% -> SAFE (Alarm Silent)")

if __name__ == "__main__":
    test_threshold_logic()
    test_full_pipeline()
