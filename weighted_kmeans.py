from itertools import combinations
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

# =====================================
# HAVERSINE DISTANCE FUNCTION
# =====================================

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

# =====================================
# NEAREST NEIGHBOR ROUTING
# =====================================

def nearest_neighbor_route(dc_lat, dc_long, stores_df):

    unvisited = stores_df.copy()

    route_sequence = []

    current_lat = dc_lat
    current_long = dc_long

    total_distance = 0

    while len(unvisited) > 0:

        unvisited['temp_distance'] = unvisited.apply(
            lambda row: haversine(
                current_lat,
                current_long,
                row['lat'],
                row['long']
            ),
            axis=1
        )

        nearest_store = unvisited.loc[
            unvisited['temp_distance'].idxmin()
        ]

        route_sequence.append(nearest_store)

        total_distance += nearest_store['temp_distance']

        current_lat = nearest_store['lat']
        current_long = nearest_store['long']

        unvisited = unvisited.drop(nearest_store.name)

    # Return to DC
    total_distance += haversine(
        current_lat,
        current_long,
        dc_lat,
        dc_long
    )

    return route_sequence, total_distance

# =====================================
# LOAD INPUT FILES
# =====================================

df = pd.read_excel('saavu2.xlsx')

truck_df = pd.read_excel('truck_master.xlsx')

df.columns = df.columns.str.lower()

truck_df.columns = truck_df.columns.str.lower()

# =====================================
# CLEAN DATA
# =====================================

df = df[(df['lat'] != 0) & (df['long'] != 0)]

df = df.dropna(
    subset=['lat', 'long', 'sales', 'demand_cft']
)

df['sales'] = pd.to_numeric(
    df['sales'],
    errors='coerce'
)

df['sales'] = df['sales'].fillna(1)

df['sales'] = df['sales'].clip(lower=1)

df['demand_cft'] = pd.to_numeric(
    df['demand_cft'],
    errors='coerce'
)

df['demand_cft'] = df['demand_cft'].fillna(1)

# =====================================
# PREPARE CLUSTERING DATA
# =====================================

X = df[['lat', 'long']].values

weights = df['sales'].values

# =====================================
# AUTOMATIC K SELECTION
# =====================================

max_allowed_distance = 700

best_k = None
best_df = None
best_centroids = None

for k in range(2, 15):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X, sample_weight=weights)

    temp_df = df.copy()

    temp_df['cluster'] = kmeans.labels_

    centroids = kmeans.cluster_centers_

    # Snap centroids to nearest real store

    real_centroids = []

    for center in centroids:

        distances = np.sqrt(
            (X[:, 0] - center[0]) ** 2 +
            (X[:, 1] - center[1]) ** 2
        )

        nearest_idx = np.argmin(distances)

        real_centroids.append(X[nearest_idx])

    centroids = np.array(real_centroids)

    # Assign DC coordinates

    temp_df['dc_lat'] = temp_df['cluster'].apply(
        lambda x: centroids[x][0]
    )

    temp_df['dc_long'] = temp_df['cluster'].apply(
        lambda x: centroids[x][1]
    )

    # Haversine distance

    temp_df['distance_km'] = temp_df.apply(
        lambda row: haversine(
            row['lat'],
            row['long'],
            row['dc_lat'],
            row['dc_long']
        ),
        axis=1
    )

    temp_df['weighted_distance'] = (
        temp_df['sales'] * temp_df['distance_km']
    )

    max_distance = temp_df['distance_km'].max()

    print(f"K={k}, Max Distance={max_distance:.2f} km")

    if max_distance <= max_allowed_distance:

        best_k = k
        best_df = temp_df
        best_centroids = centroids

        break

# =====================================
# FINAL CLUSTER OUTPUTS
# =====================================

df = best_df

centroids = best_centroids

print(f"\nOptimal K Found: {best_k}")

# =====================================
# DELETE OLD FILES
# =====================================

for file in [

    "clustered_output.xlsx",

    "dc_locations.xlsx",

    "store_dc_distances.xlsx",

    "model_summary.xlsx",

    "secondary_logistics_routes.xlsx",

    "secondary_logistics_summary.xlsx",

    "store_monthly_logistics_cost.xlsx",

    "index.html"

]:

    if os.path.exists(file):

        os.remove(file)

# =====================================
# SAVE CLUSTER OUTPUT
# =====================================

df.to_excel(
    'clustered_output.xlsx',
    index=False
)

# =====================================
# SAVE STORE-DC DISTANCES
# =====================================

distance_output = df[[

    'store',

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

# =====================================
# SAVE MODEL SUMMARY
# =====================================

summary_df = pd.DataFrame({

    'optimal_k': [best_k],

    'max_distance_km': [
        df['distance_km'].max()
    ],

    'avg_distance_km': [
        df['distance_km'].mean()
    ]

})

summary_df.to_excel(
    'model_summary.xlsx',
    index=False
)

# =====================================
# SAVE DC LOCATIONS
# =====================================

centroids_df = pd.DataFrame(
    centroids,
    columns=['lat', 'long']
)

centroids_df.to_excel(
    'dc_locations.xlsx',
    index=False
)

# =====================================
# CREATE MAP
# =====================================

map_india = folium.Map(
    location=[22.5, 78.9],
    zoom_start=5
)

k = df['cluster'].nunique()

colormap = cm.get_cmap('viridis', k)

for _, row in df.iterrows():

    color = colors.to_hex(
        colormap(row['cluster'])
    )

    folium.CircleMarker(

        location=[row['lat'], row['long']],

        radius=4,

        color=color,

        fill=True,

        fill_color=color,

        popup=(

            f"Store: {row['store']}<br>"
            f"Cluster: {row['cluster']}<br>"
            f"Distance: {row['distance_km']:.2f} km<br>"
            f"Sales: {row['sales']}"

        )

    ).add_to(map_india)

# DC markers

for _, row in centroids_df.iterrows():

    folium.Marker(

        location=[row['lat'], row['long']],

        icon=folium.Icon(
            color='red',
            icon='star'
        )

    ).add_to(map_india)

map_india.save("index.html")

# =====================================
# SECONDARY LOGISTICS
# =====================================

routes_output = []

store_level_cost_output = []

truck_df = truck_df.sort_values(
    'capacity_cft'
)

# =====================================
# ROUTE BUILDING
# =====================================

for cluster_id in sorted(df['cluster'].unique()):

    cluster_stores = df[
        df['cluster'] == cluster_id
    ].copy()

    dc_lat = cluster_stores.iloc[0]['dc_lat']

    dc_long = cluster_stores.iloc[0]['dc_long']

    cluster_stores = cluster_stores.sort_values(
        'distance_km'
    )

    current_route = []

    current_load = 0

    for _, store in cluster_stores.iterrows():

        demand = store['demand_cft']

        current_route.append(store)

        current_load += demand

        selected_truck = None

        # Select smallest feasible truck

        for _, truck_option in truck_df.iterrows():

            if current_load <= truck_option['capacity_cft']:

                selected_truck = truck_option

                break

        # Use largest truck if needed

        if selected_truck is None:

            selected_truck = truck_df.iloc[-1]

        truck_capacity = selected_truck[
            'capacity_cft'
        ]

        # Close route at 90% utilization

        if current_load >= 0.9 * truck_capacity:

            fixed_cost = selected_truck[
                'fixed_cost'
            ]

            variable_cost = selected_truck[
                'variable_cost_per_km'
            ]

            route_df = pd.DataFrame(
                current_route
            )

            route_sequence, route_distance = nearest_neighbor_route(
                dc_lat,
                dc_long,
                route_df
            )

            # Monthly cost
            monthly_route_cost = (

                fixed_cost +

                (
                    variable_cost *
                    route_distance *
                    10
                )
            )

            # Store-wise allocation

            for s in route_sequence:

                store_share = (
                    s['demand_cft'] / current_load
                )

                allocated_cost = (
                    monthly_route_cost *
                    store_share
                )

                store_level_cost_output.append({

                    'cluster': cluster_id,

                    'store': s['store'],

                    'truck_type': selected_truck[
                        'truck_type'
                    ],

                    'demand_cft': s['demand_cft'],

                    'store_share_percent':
                        store_share * 100,

                    'allocated_monthly_logistics_cost':
                        allocated_cost,

                    'route_distance_km':
                        route_distance
                })

            routes_output.append({

                'cluster': cluster_id,

                'truck_type': selected_truck[
                    'truck_type'
                ],

                'stores_served': len(
                    current_route
                ),

                'store_list': ' -> '.join(
                    [
                        str(s['store'])
                        for s in route_sequence
                    ]
                ),

                'total_load_cft': current_load,

                'route_distance_km':
                    route_distance,

                'monthly_route_cost':
                    monthly_route_cost
            })

            current_route = []

            current_load = 0

    # Remaining route

    if len(current_route) > 0:

        selected_truck = None

        for _, truck_option in truck_df.iterrows():

            if current_load <= truck_option[
                'capacity_cft'
            ]:

                selected_truck = truck_option

                break

        if selected_truck is None:

            selected_truck = truck_df.iloc[-1]

        fixed_cost = selected_truck[
            'fixed_cost'
        ]

        variable_cost = selected_truck[
            'variable_cost_per_km'
        ]

        route_df = pd.DataFrame(
            current_route
        )

        route_sequence, route_distance = nearest_neighbor_route(
            dc_lat,
            dc_long,
            route_df
        )

        monthly_route_cost = (

            fixed_cost +

            (
                variable_cost *
                route_distance *
                10
            )
        )

        # Store-wise allocation

        for s in route_sequence:

            store_share = (
                s['demand_cft'] / current_load
            )

            allocated_cost = (
                monthly_route_cost *
                store_share
            )

            store_level_cost_output.append({

                'cluster': cluster_id,

                'store': s['store'],

                'truck_type': selected_truck[
                    'truck_type'
                ],

                'demand_cft': s['demand_cft'],

                'store_share_percent':
                    store_share * 100,

                'allocated_monthly_logistics_cost':
                    allocated_cost,

                'route_distance_km':
                    route_distance
            })

        routes_output.append({

            'cluster': cluster_id,

            'truck_type': selected_truck[
                'truck_type'
            ],

            'stores_served': len(
                current_route
            ),

            'store_list': ' -> '.join(
                [
                    str(s['store'])
                    for s in route_sequence
                ]
            ),

            'total_load_cft': current_load,

            'route_distance_km':
                route_distance,

            'monthly_route_cost':
                monthly_route_cost
        })

# =====================================
# SAVE ROUTE OUTPUT
# =====================================

routes_df = pd.DataFrame(routes_output)

routes_df.to_excel(
    'secondary_logistics_routes.xlsx',
    index=False
)

# =====================================
# SAVE STORE-WISE COST
# =====================================

store_cost_df = pd.DataFrame(
    store_level_cost_output
)

store_cost_df.to_excel(
    'store_monthly_logistics_cost.xlsx',
    index=False
)

# =====================================
# SAVE SECONDARY SUMMARY
# =====================================

summary_secondary = pd.DataFrame({

    'total_routes': [
        len(routes_df)
    ],

    'total_secondary_cost': [
        routes_df[
            'monthly_route_cost'
        ].sum()
    ],

    'average_route_cost': [
        routes_df[
            'monthly_route_cost'
        ].mean()
    ],

    'total_route_distance': [
        routes_df[
            'route_distance_km'
        ].sum()
    ]
})

summary_secondary.to_excel(
    'secondary_logistics_summary.xlsx',
    index=False
)

print("Done! Files generated successfully.")
