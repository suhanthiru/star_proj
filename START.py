import pandas as pd
import torch
import torch.nn.functional as F
from skyfield.api import load, wgs84

from ingestion import build_sky_graph
from vectorizer import text_to_graph
from gnn_engine import ConstellationGNN, create_pyg_data
from alignment_solver import get_best_alignment
from observatory import get_sky_forecast
from renderer import launch_telescope

def main():
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

    print("\n[4] Running Graph Neural Network Pattern Recognition...")
    sky_data = create_pyg_data(sky_nodes, sky_edges)
    word_data = create_pyg_data(word_nodes, word_edges)
    
    model = ConstellationGNN(in_channels=3, hidden_channels=16, out_channels=8)
    model.eval()

    with torch.no_grad():
        sky_emb = F.normalize(model(sky_data.x, sky_data.edge_index), p=2, dim=1)
        word_emb = F.normalize(model(word_data.x, word_data.edge_index), p=2, dim=1)

    similarity_matrix = torch.mm(word_emb, sky_emb.t())
    best_star_idx = torch.argmax(similarity_matrix[0]).item()
    anchor_coords = sky_nodes[best_star_idx, :2]
    print(f"    -> Best anchor region found! (Star ID: {sky_ids[best_star_idx]})")

    print("\n[5] Executing Elastic Alignment...")
    smooth_nodes, snapped_nodes, mse = get_best_alignment(word_nodes, word_edges, sky_nodes, anchor_coords)
    print(f"    -> Alignment Complete. Mean Squared Error: {mse:.4f}")

    print("\n[6] Calculating Live Earth Visibility...")
    anchor_row = df_raw[df_raw['id'] == sky_ids[best_star_idx]].iloc[0]
    alt, az, compass_dir = get_sky_forecast(anchor_row, observer, ts)

    launch_telescope(smooth_nodes, snapped_nodes, word_edges, sky_nodes, sky_ids, df_raw, anchor_coords, compass_dir, az)

if __name__ == "__main__":
    main()