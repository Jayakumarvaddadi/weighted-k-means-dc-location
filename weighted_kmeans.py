import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from math import radians, sin, cos, sqrt, atan2

# =====================================================
# LOAD INPUT FILE
# =====================================================

input_file = "saavu2.xlsx"

df = pd.read_excel(input_file)

# =====================================================
# REQUIRED COLUMNS
# =====================================================

LAT_COL = "lat"
LON_COL = "long"
WEIGHT_COL = "sales"
STORE_COL = "store"
DEMAND_COL = "demand_cft"

# =====================================================
# PARAMETERS
# =====================================================

K = 4

# =====================================================
# REMOVE MISSING VALUES
# =====================================================

df = df.dropna(
    subset=[
        LAT_COL,
        LON_COL,
        WEIGHT_COL,
        DEMAND_COL
    ]
).copy()

# =====================================================
# HAVERSINE FUNCTION
# =====================================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c

# =====================================================
# PREPARE DATA
# =====================================================

X = df[[LAT_COL, LON_COL]].values
weights = df[WEIGHT_COL].values

# =====================================================
# WEIGHTED K-MEANS
# =====================================================

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

kmeans.fit(
    X,
    sample_weight=weights
)

# =====================================================
# CLUSTER ASSIGNMENT
# =====================================================

df["cluster"] = kmeans.labels_

# =====================================================
# DC LOCATIONS
# =====================================================

dc_locations = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=["dc_lat", "dc_long"]
)

dc_locations["dc_id"] = [
    f"DC_{i+1}" for i in range(K)
]

# =====================================================
# ASSIGN DC
# =====================================================

cluster_to_dc = {
    i: f"DC_{i+1}"
    for i in range(K)
}

df["assigned_dc"] = df["cluster"].map(
    cluster_to_dc
)

# =====================================================
# STORE TO DC DISTANCE
# =====================================================

store_dc_distances = []

for idx, row in df.iterrows():

    cluster_id = row["cluster"]

    dc_lat = (
        kmeans.cluster_centers_[cluster_id][0]
    )

    dc_lon = (
        kmeans.cluster_centers_[cluster_id][1]
    )

    distance = haversine(
        row[LAT_COL],
        row[LON_COL],
        dc_lat,
        dc_lon
    )

    store_dc_distances.append(distance)

df["distance_to_dc_km"] = (
    store_dc_distances
)

# =====================================================
# SAVE OUTPUT FILES
# =====================================================

df.to_excel(
    "clustered_output.xlsx",
    index=False
)

dc_locations.to_excel(
    "dc_locations.xlsx",
    index=False
)

df[
    [
        STORE_COL,
        "assigned_dc",
        "distance_to_dc_km"
    ]
].to_excel(
    "store_dc_distances.xlsx",
    index=False
)

# =====================================================
# CREATE INTERACTIVE MAP
# =====================================================

center_lat = df[LAT_COL].mean()
center_lon = df[LON_COL].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5
)

colors = [
    "red",
    "blue",
    "green",
    "purple"
]

# =====================================================
# PLOT STORES
# =====================================================

for idx, row in df.iterrows():

    cluster_id = int(row["cluster"])

    color = colors[
        cluster_id % len(colors)
    ]

    folium.CircleMarker(
        location=[
            row[LAT_COL],
            row[LON_COL]
        ],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=(
            f"Store: {row[STORE_COL]}<br>"
            f"Sales: {row[WEIGHT_COL]}<br>"
            f"Demand: {row[DEMAND_COL]}<br>"
            f"Assigned DC: {row['assigned_dc']}<br>"
            f"Distance to DC: "
            f"{row['distance_to_dc_km']:.2f} km"
        )
    ).add_to(m)

# =====================================================
# PLOT DCS
# =====================================================

for idx, row in dc_locations.iterrows():

    color = colors[
        idx % len(colors)
    ]

    folium.Marker(
        location=[
            row["dc_lat"],
            row["dc_long"]
        ],
        popup=row["dc_id"],
        icon=folium.Icon(
            color=color,
            icon="home"
        )
    ).add_to(m)

# =====================================================
# SAVE MAP
# =====================================================

m.save("index.html")

# =====================================================
# SUMMARY
# =====================================================

print("\n================================")
print("WEIGHTED K-MEANS COMPLETED")
print("================================")

print(f"\nNumber of DCs = {K}")

print("\nStores per DC:")

print(
    df["assigned_dc"].value_counts()
)

print(
    f"\nMaximum Store Distance = "
    f"{df['distance_to_dc_km'].max():.2f} km"
)

print(
    f"\nAverage Store Distance = "
    f"{df['distance_to_dc_km'].mean():.2f} km"
)

print("\nGenerated Files:")

print("1. clustered_output.xlsx")
print("2. dc_locations.xlsx")
print("3. store_dc_distances.xlsx")
print("4. index.html")
