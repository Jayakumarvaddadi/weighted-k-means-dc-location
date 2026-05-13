import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium

# =========================
# LOAD DATA
# =========================

input_file = "saavu2.xlsx"

df = pd.read_excel(input_file)

# =========================
# REQUIRED COLUMNS
# =========================

LAT_COL = "lat"
LON_COL = "long"
WEIGHT_COL = "sales"
STORE_COL = "store"

# Remove missing values
df = df.dropna(subset=[LAT_COL, LON_COL, WEIGHT_COL]).copy()

# =========================
# PREPARE DATA
# =========================

X = df[[LAT_COL, LON_COL]].values
weights = df[WEIGHT_COL].values

# =========================
# WEIGHTED K-MEANS
# =========================

K = 4

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

kmeans.fit(X, sample_weight=weights)

# Assign clusters
df["cluster"] = kmeans.labels_

# =========================
# DC LOCATIONS
# =========================

dc_locations = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=["dc_lat", "dc_long"]
)

dc_locations["dc_id"] = [
    f"DC_{i+1}" for i in range(K)
]

# =========================
# ASSIGN DC TO STORES
# =========================

cluster_to_dc = {
    i: f"DC_{i+1}" for i in range(K)
}

df["assigned_dc"] = df["cluster"].map(cluster_to_dc)

# =========================
# SAVE OUTPUT FILES
# =========================

df.to_excel(
    "clustered_output.xlsx",
    index=False
)

dc_locations.to_excel(
    "dc_locations.xlsx",
    index=False
)

# =========================
# CREATE INTERACTIVE MAP
# =========================

# Map center
center_lat = df[LAT_COL].mean()
center_lon = df[LON_COL].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5
)

# Cluster colors
colors = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "cadetblue",
    "darkgreen"
]

# =========================
# PLOT STORES
# =========================

for idx, row in df.iterrows():

    cluster_id = int(row["cluster"])
    color = colors[cluster_id % len(colors)]

    folium.CircleMarker(
        location=[row[LAT_COL], row[LON_COL]],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=(
            f"Store: {row[STORE_COL]}<br>"
            f"Sales: {row[WEIGHT_COL]}<br>"
            f"Assigned DC: {row['assigned_dc']}"
        )
    ).add_to(m)

# =========================
# PLOT DC LOCATIONS
# =========================

for idx, row in dc_locations.iterrows():

    color = colors[idx % len(colors)]

    folium.Marker(
        location=[row["dc_lat"], row["dc_long"]],
        popup=f"{row['dc_id']}",
        icon=folium.Icon(
            color=color,
            icon="warehouse",
            prefix="fa"
        )
    ).add_to(m)

# =========================
# SAVE MAP
# =========================

m.save("dc_network_map.html")

# =========================
# PRINT SUMMARY
# =========================

print("\nWeighted K-Means Completed Successfully")
print(f"\nNumber of DCs Created: {K}")

print("\nStores per Cluster:")
print(df["assigned_dc"].value_counts())

print("\nFiles Generated:")
print("1. clustered_output.xlsx")
print("2. dc_locations.xlsx")
print("3. dc_network_map.html")
