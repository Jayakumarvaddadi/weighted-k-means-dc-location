import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =====================================================
# 1. CORE UTILITIES
# =====================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # Earth radius in km
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# =====================================================
# 2. LOAD & CLEAN DATA
# =====================================================
# Update filenames if necessary (e.g., "saavu2.xlsx")
df = pd.read_excel("saavu2.xlsx")
df.columns = df.columns.str.lower().str.strip()

# Cleaning
df = df[(df["lat"] != 0) & (df["long"] != 0)].dropna(subset=["lat", "long", "sales"])
df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(1).clip(lower=1)

# =====================================================
# 3. PHASE 1: DC NETWORK DESIGN (FIXED K=4)
# =====================================================
K_CLUSTERS = 4
X = df[["lat", "long"]].values
weights = df["sales"].values

print(f"Executing Weighted KMeans with K={K_CLUSTERS}...")
kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(X, sample_weight=weights)

df["cluster"] = kmeans.labels_
centroids = kmeans.cluster_centers_

# SNAP TO NEAREST REAL STORE
real_centroids = []
for center in centroids:
    distances = np.sqrt((X[:, 0] - center[0]) ** 2 + (X[:, 1] - center[1]) ** 2)
    real_centroids.append(X[np.argmin(distances)])
snapped_centroids = np.array(real_centroids)

# ASSIGN DC COORDINATES back to stores
df["dc_lat"] = df["cluster"].apply(lambda x: snapped_centroids[x][0])
df["dc_long"] = df["cluster"].apply(lambda x: snapped_centroids[x][1])

# Calculate distances
df["distance_km"] = df.apply(
    lambda row: haversine(row["lat"], row["long"], row["dc_lat"], row["dc_long"]), axis=1
)

# =====================================================
# 4. EXPORT OUTPUTS
# =====================================================
df.to_excel("clustered_output.xlsx", index=False)
pd.DataFrame(snapped_centroids, columns=["lat", "long"]).to_excel("dc_locations.xlsx", index=False)

pd.DataFrame({
    "Fixed_K": [K_CLUSTERS],
    "Max_Distance_km": [df["distance_km"].max()],
    "Avg_Distance_km": [df["distance_km"].mean()]
}).to_excel("model_summary.xlsx", index=False)

# =====================================================
# 5. VISUALIZATION (MAP)
# =====================================================
# Create map centered on average coordinates
m = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start=5)

# Standard map colors for clusters
color_list = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

# Plot Stores
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=3,
        color=color_list[int(row["cluster"]) % len(color_list)],
        fill=True,
        fill_opacity=0.7,
        popup=f"Store: {row.get('store', 'N/A')}<br>Cluster: {row['cluster']}"
    ).add_to(m)

# Plot DCs as Stars
for i, dc in enumerate(snapped_centroids):
    folium.Marker(
        location=[dc[0], dc[1]],
        icon=folium.Icon(color="black", icon="star"),
        popup=f"DC {i} (Primary Hub)"
    ).add_to(m)

m.save("dc_network_map.html")
print(f"Process complete. K={K_CLUSTERS}. Map saved as dc_network_map.html")
