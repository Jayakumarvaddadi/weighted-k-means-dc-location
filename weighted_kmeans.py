import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =====================================================
# HAVERSINE FUNCTION
# =====================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = (sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# =====================================================
# LOAD & CLEAN DATA
# =====================================================
df = pd.read_excel("saavu2.xlsx")
df.columns = df.columns.str.lower().str.strip()

# Cleanup
df = df[(df["lat"] != 0) & (df["long"] != 0)].dropna(subset=["lat", "long", "sales"])
df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(1).clip(lower=1)

# =====================================================
# PREPARE CLUSTERING (K-MEANS)
# =====================================================
X = df[["lat", "long"]].values
weights = df["sales"].values
MAX_STORE_DISTANCE = 700

best_k = None
best_df = None
best_centroids = None

# =====================================================
# AUTOMATIC K SELECTION
# =====================================================
for k in range(1, 21):
    print(f"Trying K = {k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X, sample_weight=weights)
    
    temp_df = df.copy()
    temp_df["cluster"] = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # SNAP TO NEAREST REAL STORE
    real_centroids = []
    for center in centroids:
        # Distance to all stores to find the closest real one
        distances = np.sqrt((X[:, 0] - center[0]) ** 2 + (X[:, 1] - center[1]) ** 2)
        real_centroids.append(X[np.argmin(distances)])
    
    snapped_centroids = np.array(real_centroids)

    # ASSIGN DC COORDINATES
    temp_df["dc_lat"] = temp_df["cluster"].apply(lambda x: snapped_centroids[x][0])
    temp_df["dc_long"] = temp_df["cluster"].apply(lambda x: snapped_centroids[x][1])

    # VALIDATE DISTANCE
    temp_df["distance_km"] = temp_df.apply(
        lambda row: haversine(row["lat"], row["long"], row["dc_lat"], row["dc_long"]), axis=1
    )

    max_dist = temp_df["distance_km"].max()
    print(f"Max Distance found: {max_dist:.2f} km")

    if max_dist <= MAX_STORE_DISTANCE:
        best_k = k
        best_df = temp_df
        best_centroids = snapped_centroids
        print(f"--> Feasible Network Found with K = {k}")
        break

# =====================================================
# EXPORT OUTPUTS
# =====================================================
# 1. Full Clustered List
best_df.to_excel("clustered_output.xlsx", index=False)

# 2. DC Specific Locations
pd.DataFrame(best_centroids, columns=["lat", "long"]).to_excel("dc_locations.xlsx", index=False)

# 3. Simple Summary
pd.DataFrame({
    "Optimal_K": [best_k],
    "Max_Distance_km": [best_df["distance_km"].max()],
    "Avg_Distance_km": [best_df["distance_km"].mean()]
}).to_excel("model_summary.xlsx", index=False)

# =====================================================
# VISUALIZATION (MAP)
# =====================================================
m = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start=5)
colors = [mcolors.to_hex(plt.cm.tab20(i/max(best_k, 1))) for i in range(best_k)]

# Plot Stores
for _, row in best_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=3,
        color=colors[int(row["cluster"])],
        fill=True,
        popup=f"Store: {row.get('store', 'N/A')}"
    ).add_to(m)

# Plot DCs
for i, dc in enumerate(best_centroids):
    folium.Marker(
        location=[dc[0], dc[1]],
        icon=folium.Icon(color="red", icon="star"),
        popup=f"DC {i}"
    ).add_to(m)

m.save("dc_network_map.html")
print("\nPhase 1 Completed. Files generated: clustered_output.xlsx, dc_locations.xlsx, model_summary.xlsx, dc_network_map.html")
