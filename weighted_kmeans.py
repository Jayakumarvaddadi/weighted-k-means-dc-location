import os

print("FILES IN DIRECTORY:")
print(os.listdir())
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):

    R = 6371  # Earth radius in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c
# Load data
df = pd.read_excel('saavu2.xlsx')
df.columns = df.columns.str.lower()

# Clean data
df = df[(df['lat'] != 0) & (df['long'] != 0)]
df = df.dropna(subset=['lat', 'long', 'sales'])

df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
df['sales'] = df['sales'].fillna(1)
df['sales'] = df['sales'].clip(lower=1)

# Prepare data
X = df[['lat', 'long']].values
weights = df['sales'].values

# Weighted K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X, sample_weight=weights)

df['cluster'] = kmeans.labels_
centroids = kmeans.cluster_centers_
# Assign DC coordinates to each store
df['dc_lat'] = df['cluster'].apply(lambda x: centroids[x][0])
df['dc_long'] = df['cluster'].apply(lambda x: centroids[x][1])

# Calculate haversine distance
df['distance_km'] = df.apply(
    lambda row: haversine(
        row['lat'],
        row['long'],
        row['dc_lat'],
        row['dc_long']
    ),
    axis=1
)

# Optional logistics KPI
df['weighted_distance'] = df['sales'] * df['distance_km']

# Remove old outputs
import os

for file in [
    "clustered_output.xlsx",
    "dc_locations.xlsx",
    "store_dc_distances.xlsx",
    "index.html"
]:
    if os.path.exists(file):
        os.remove(file)
# Save output
df.to_excel('clustered_output.xlsx', index=False)
# Save store-to-DC haversine distances
distance_output = df[[
    'lat',
    'long',
    'sales',
    'cluster',
    'dc_lat',
    'dc_long',
    'distance_km',
    'weighted_distance'
]]

distance_output.to_excel(
    'store_dc_distances.xlsx',
    index=False
)

centroids_df = pd.DataFrame(centroids, columns=['lat', 'long'])
import os

dc_file = "dc_locations.xlsx"

# Delete old file if exists
if os.path.exists(dc_file):
    os.remove(dc_file)

# Save new file
centroids_df.to_excel(dc_file, index=False)

# Create map
map_india = folium.Map(location=[22.5, 78.9], zoom_start=5)

k = df['cluster'].nunique()
colormap = cm.get_cmap('viridis', k)

for _, row in df.iterrows():

    color = colors.to_hex(colormap(row['cluster']))

    folium.CircleMarker(
        location=[row['lat'], row['long']],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,

        popup=(
            f"Cluster: {row['cluster']}<br>"
            f"Distance: {row['distance_km']:.2f} km<br>"
            f"Sales: {row['sales']}"
        )

    ).add_to(map_india)

for _, row in centroids_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['long']],
        icon=folium.Icon(color='red', icon='star')
    ).add_to(map_india)

map_india.save("index.html")

print("Done! Files generated.")
