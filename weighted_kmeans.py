import os

print("FILES IN DIRECTORY:")
print(os.listdir())
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors

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

# Remove old outputs
import os

for file in ["clustered_output.xlsx", "dc_locations.xlsx", "map.html"]:
    if os.path.exists(file):
        os.remove(file)
# Save output
df.to_excel('clustered_output.xlsx', index=False)

centroids_df = pd.DataFrame(centroids, columns=['lat', 'long'])
centroids_df.to_excel('dc_locations.xlsx', index=False)

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
        fill_color=color
    ).add_to(map_india)

for _, row in centroids_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['long']],
        icon=folium.Icon(color='red', icon='star')
    ).add_to(map_india)

map_india.save("map.html")

print("Done! Files generated.")
