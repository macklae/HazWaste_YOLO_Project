import statistics
import time
from io import BytesIO

import requests
from PIL import Image

URL = "http://127.0.0.1:8000/api/detect/combined"

image = Image.new("RGB", (640, 640), "white")
buffer = BytesIO()
image.save(buffer, format="PNG")
payload = buffer.getvalue()


def measure_request():
    start = time.perf_counter()

    response = requests.post(
        URL,
        files={"file": ("test.png", payload, "image/png")},
        timeout=30,
    )

    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


print("Warming up API...")
for _ in range(3):
    measure_request()

print("Measuring 20 requests...")
times = [measure_request() for _ in range(20)]

print(f"Average: {statistics.mean(times):.1f} ms")
print(f"Median: {statistics.median(times):.1f} ms")
print(f"Minimum: {min(times):.1f} ms")
print(f"Maximum: {max(times):.1f} ms")
print(f"P95: {sorted(times)[18]:.1f} ms")