
from pathlib import Path
from collections import Counter
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as transform_coords


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

YEARS = (2015, 2020, 2025)

LANDSAT = {
    year: RAW / "landsat" / f"Landsat_{year}_KMC_9Band.tif"
    for year in YEARS
}

REFERENCE = {
    year: RAW / "lulc_reference" / f"LULC_Reference_Samples_{year}.kmz"
    for year in YEARS
}

EXPECTED_CRS = "EPSG:32644"
EXPECTED_SIZE = (531, 331)
EXPECTED_RESOLUTION = (30.0, 30.0)
EXPECTED_BANDS = 9
EXPECTED_SAMPLES = 375

BAND_NAMES = [
    "Blue", "Green", "Red", "NIR",
    "SWIR1", "SWIR2", "NDVI", "MNDWI", "NDBI"
]


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def find_child(element, name):
    return next(
        (child for child in element if local_name(child.tag) == name),
        None
    )


def child_text(element, name):
    child = find_child(element, name)
    return (child.text or "").strip() if child is not None else ""


def parse_reference_kmz(path):
    with ZipFile(path) as archive:
        kml_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".kml")
        ]

        if not kml_files:
            raise ValueError(f"No KML document found: {path.name}")

        root = ET.fromstring(archive.read(kml_files[0]))

    points = []
    non_point_count = 0

    def walk(element, folders):
        nonlocal non_point_count

        tag = local_name(element.tag)

        if tag in ("Document", "Folder"):
            folder_name = child_text(element, "name")
            if tag == "Folder" and folder_name:
                folders = folders + [folder_name]

        if tag == "Placemark":
            point = next(
                (
                    item for item in element.iter()
                    if local_name(item.tag) == "Point"
                ),
                None
            )

            if point is None:
                non_point_count += 1
                return

            coord_text = child_text(point, "coordinates")

            record = {
                "name": child_text(element, "name"),
                "folder": " / ".join(folders),
                "description": child_text(element, "description"),
                "attributes": {},
                "longitude": None,
                "latitude": None,
            }

            for item in element.iter():
                tag_name = local_name(item.tag)

                if tag_name == "Data":
                    key = item.attrib.get("name", "")
                    value = child_text(item, "value")
                    if key:
                        record["attributes"][key] = value

                elif tag_name == "SimpleData":
                    key = item.attrib.get("name", "")
                    if key:
                        record["attributes"][key] = (
                            item.text or ""
                        ).strip()

            try:
                parts = coord_text.strip().split()[0].split(",")
                record["longitude"] = float(parts[0])
                record["latitude"] = float(parts[1])
            except (IndexError, ValueError):
                pass

            points.append(record)
            return

        for child in element:
            walk(child, folders)

    walk(root, [])

    return points, non_point_count


def validate_landsat():
    print("\n========== LANDSAT VALIDATION ==========")

    reference_profile = None
    all_ok = True

    for year, path in LANDSAT.items():
        print(f"\n--- {year} ---")

        if not path.is_file():
            print("FAIL: Missing file:", path)
            all_ok = False
            continue

        with rasterio.open(path) as src:
            print("File:", path.name)
            print("CRS:", src.crs)
            print("Resolution:", src.res)
            print("Dimensions:", src.width, "x", src.height)
            print("Bands:", src.count)
            print("Data types:", src.dtypes)
            print("NoData:", src.nodata)

            checks = {
                "CRS": str(src.crs) == EXPECTED_CRS,
                "Dimensions": (src.width, src.height) == EXPECTED_SIZE,
                "Resolution": bool(
                    np.allclose(src.res, EXPECTED_RESOLUTION)
                ),
                "Band count": src.count == EXPECTED_BANDS,
                "Float32": all(
                    dtype == "float32" for dtype in src.dtypes
                ),
                "NoData": src.nodata == -9999,
            }

            data = src.read(masked=True)
            invalid = np.ma.getmaskarray(data) | ~np.isfinite(
                data.data
            )
            valid_pixels = ~np.any(invalid, axis=0)

            valid_count = int(valid_pixels.sum())
            print("Fully valid pixels:", valid_count)
            checks["Usable pixels"] = valid_count > 0

            profile = {
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "transform": src.transform,
                "bands": src.count,
                "nodata": src.nodata,
            }

            if reference_profile is None:
                reference_profile = profile
            else:
                aligned = (
                    profile["crs"] == reference_profile["crs"]
                    and profile["width"] == reference_profile["width"]
                    and profile["height"] == reference_profile["height"]
                    and profile["bands"] == reference_profile["bands"]
                    and profile["nodata"] == reference_profile["nodata"]
                    and bool(np.allclose(
                        tuple(profile["transform"]),
                        tuple(reference_profile["transform"]),
                        rtol=0,
                        atol=1e-7
                    ))
                )
                checks["Grid alignment"] = aligned

            for name, passed in checks.items():
                print(f"{name}: {'PASS' if passed else 'FAIL'}")

            all_ok = all_ok and all(checks.values())

    return all_ok


def validate_references():
    print("\n========== LULC REFERENCE VALIDATION ==========")

    all_ok = True

    for year, path in REFERENCE.items():
        print(f"\n--- {year} ---")

        if not path.is_file():
            print("FAIL: Missing file:", path)
            all_ok = False
            continue

        try:
            points, non_points = parse_reference_kmz(path)
        except Exception as exc:
            print("FAIL: KMZ reading error:", exc)
            all_ok = False
            continue

        print("File:", path.name)
        print("Point placemarks:", len(points))
        print("Non-point placemarks:", non_points)

        count_ok = len(points) == EXPECTED_SAMPLES
        print("Expected sample count:",
              "PASS" if count_ok else "FAIL")
        all_ok = all_ok and count_ok

        folder_counts = Counter(p["folder"] for p in points)
        print("\nReference folder groups:")
        for folder, count in folder_counts.most_common(12):
            print(f"  {folder or '(root)'}: {count}")

        print("\nFirst 5 reference records:")
        for point in points[:5]:
            print({
                "name": point["name"],
                "folder": point["folder"],
                "attributes": point["attributes"],
                "lon": point["longitude"],
                "lat": point["latitude"],
            })

        valid_coordinates = [
            p for p in points
            if p["longitude"] is not None
            and p["latitude"] is not None
            and np.isfinite(p["longitude"])
            and np.isfinite(p["latitude"])
            and -180 <= p["longitude"] <= 180
            and -90 <= p["latitude"] <= 90
        ]

        coordinate_ok = len(valid_coordinates) == len(points)

        print("Valid coordinates:", len(valid_coordinates))
        print("Coordinate check:",
              "PASS" if coordinate_ok else "FAIL")

        all_ok = all_ok and coordinate_ok

        with rasterio.open(LANDSAT[year]) as src:
            inside = 0
            usable = 0

            if valid_coordinates:
                longitudes = [
                    p["longitude"] for p in valid_coordinates
                ]
                latitudes = [
                    p["latitude"] for p in valid_coordinates
                ]

                xs, ys = transform_coords(
                    "EPSG:4326",
                    src.crs,
                    longitudes,
                    latitudes
                )

                for point, x, y in zip(valid_coordinates, xs, ys):
                    r, c = rowcol(src.transform, x, y)

                    if 0 <= r < src.height and 0 <= c < src.width:
                        inside += 1

                        sample = next(
                            src.sample([(x, y)], masked=True)
                        )

                        mask = np.ma.getmaskarray(sample)
                        values = np.asarray(sample.data)

                        if not np.any(mask) and np.all(
                            np.isfinite(values)
                        ):
                            usable += 1
                        else:
                            print("\nINVALID REFERENCE SAMPLE")
                            print("Sample ID:", point["name"])
                            print("Class folder:", point["folder"])
                            print("Longitude:", point["longitude"])
                            print("Latitude:", point["latitude"])
                            print("Raster pixel:", r, c)
                            print("Band values:", values.tolist())
                            print("Masked bands:", np.flatnonzero(mask).tolist())
                            print("Non-finite bands:",
                                np.flatnonzero(~np.isfinite(values)).tolist())

            print("Points inside raster:", inside)
            print("Points on valid raster data:", usable)

            spatial_ok = (
                inside == len(points)
                and usable == len(points)
            )

            print("Raster sample coverage:",
                  "PASS" if spatial_ok else "FAIL")

            all_ok = all_ok and spatial_ok

    return all_ok


def main():
    print("=" * 60)
    print("PART 3 - FUTURE WETLAND BOUNDARY PROJECTION")
    print("SCRIPT 01 - INPUT DATA VALIDATION")
    print("=" * 60)

    landsat_ok = validate_landsat()

    if not landsat_ok:
        print("\nInput raster checks require investigation.")
        print("Reference validation will not run yet.")
        return

    reference_ok = validate_references()

    print("\n" + "=" * 60)

    if landsat_ok and reference_ok:
        print("PART 3 INITIAL INPUT VALIDATION: PASS")
        print("Next: verify class labels and study-area boundary.")
    else:
        print("PART 3 INITIAL INPUT VALIDATION: REVIEW REQUIRED")

    print("=" * 60)


if __name__ == "__main__":
    main()
