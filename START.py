import numpy as np
import pandas as pd
from skyfield.api import load, wgs84

from ingestion import build_sky_graph
from vectorizer import text_to_graph
from gnn_engine import find_best_anchor
from alignment_solver import get_best_alignment
from observatory import get_sky_forecast
from renderer import launch_telescope

SEED = 0   # makes anchor selection reproducible


def main():
    np.random.seed(SEED)

    print("========================================")
    print("   CELESTIAL ARCHAEOLOGIST INITIATED    ")
    print("========================================")

    print("\n[1] Loading Astronomical Ephemeris & Catalogs...")
    eph, ts = load('de421.bsp'), load.timescale()
    sky_nodes, sky_edges, sky_ids = build_sky_graph("hygdata_v42.csv.gz")
    df_raw = pd.read_csv("hygdata_v42.csv.gz")

    print("\n[2] Calibrating Observatory Location...")
    lat = float(input("Enter Latitude (default 34.18): ") or 34.1854)
    lon = float(input("Enter Longitude (default -83.92): ") or -83.9244)
    observer = eph['earth'] + wgs84.latlon(lat, lon)

    target_word = input("\nEnter the word to find in the stars: ").strip() or "STAR"
    print(f"\n[3] Generating Graph Topology for '{target_word}'...")
    word_nodes, word_edges = text_to_graph(target_word)

    print("\n[4] Searching star-dense regions for the best-fitting anchor...")
    best_idx, anchor_coords, anchor_mse = find_best_anchor(
        word_nodes, word_edges, sky_nodes, seed=SEED)
    print(f"    -> Best anchor region found! (Star ID: {sky_ids[best_idx]}, "
          f"fit MSE: {anchor_mse:.4f})")

    print("\n[5] Fitting word with shape-preserving similarity transform...")
    smooth_nodes, snapped_nodes, mse = get_best_alignment(
        word_nodes, word_edges, sky_nodes, anchor_coords)
    print(f"    -> Alignment Complete. Mean Squared Error: {mse:.4f}")

    print("\n[6] Calculating Live Earth Visibility...")
    anchor_row = df_raw[df_raw['id'] == sky_ids[best_idx]].iloc[0]
    alt, az, compass_dir = get_sky_forecast(anchor_row, observer, ts)

    launch_telescope(smooth_nodes, snapped_nodes, word_edges, sky_nodes, sky_ids,
                     df_raw, anchor_coords, compass_dir, az)


if __name__ == "__main__":
    main()
