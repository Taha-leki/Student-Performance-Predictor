
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

url = (
    "https://archive.ics.uci.edu/static/public/320/"
    "student+performance.zip"
)

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

print("Downloading the official UCI dataset...")

try:
    with urlopen(url, timeout=30) as response:
        outer_data = response.read()

    with ZipFile(BytesIO(outer_data)) as outer_zip:
        print("Opening the downloaded archive...")

        nested_name = next(
            name for name in outer_zip.namelist()
            if Path(name).name.lower() == "student.zip"
        )

        nested_data = outer_zip.read(nested_name)

    with ZipFile(BytesIO(nested_data)) as inner_zip:
        for filename in ("student-mat.csv", "student-por.csv"):
            matches = [
                name for name in inner_zip.namelist()
                if Path(name.replace("\\", "/")).name.lower()
                == filename
            ]

            if matches:
                output_path = data_dir / filename
                output_path.write_bytes(inner_zip.read(matches[0]))
                print(f"Saved: {output_path}")

    print("\nDataset download finished.")

except Exception as error:
    print(f"\nDownload failed: {error}")
