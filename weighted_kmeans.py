import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from math import radians, sin, cos, sqrt, atan2

# =====================================================
# PHASE 1
# DC LOCATION OPTIMIZATION
# =====================================================

# =====================================================
# LOAD INPUT FILES
# =====================================================

input_file = "saavu2.xlsx"
truck_file = "truck_master.xlsx"

df = pd.read_excel(input_file)
truck_df = pd.read_excel(truck_file)

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

K = 6

TARGET_DC = "DC_1"

MAX_ROUTE_DISTANCE = 700

MONTHLY_MULTIPLIER = 10

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
# ASSIGN CLUSTERS
# =====================================================

df["cluster"] = kmeans.labels_

# =====================================================
# GET DC LOCATIONS
# FORCE TO REAL STORE LOCATIONS
# =====================================================

raw_dc_locations = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=["dc_lat", "dc_long"]
)

real_dc_locations = []

for idx, dc in raw_dc_locations.iterrows():

    min_distance = 999999

    nearest_store = None

    for _, store in df.iterrows():

        distance = haversine(
            dc["dc_lat"],
            dc["dc_long"],
            store[LAT_COL],
            store[LON_COL]
        )

        if distance < min_distance:

            min_distance = distance

            nearest_store = store

    real_dc_locations.append({

        "dc_lat":
            nearest_store[LAT_COL],

        "dc_long":
            nearest_store[LON_COL]
    })

dc_locations = pd.DataFrame(
    real_dc_locations
)

dc_locations["dc_id"] = [
    f"DC_{i+1}"
    for i in range(K)
]

# =====================================================
# ASSIGN DCS
# =====================================================

cluster_to_dc = {
    i: f"DC_{i+1}"
    for i in range(K)
}

df["assigned_dc"] = df["cluster"].map(
    cluster_to_dc
)

# =====================================================
# CALCULATE STORE TO DC DISTANCES
# =====================================================

store_dc_distances = []

for idx, row in df.iterrows():

    cluster_id = row["cluster"]

    dc_lat = dc_locations.iloc[
        cluster_id
    ]["dc_lat"]

    dc_lon = dc_locations.iloc[
        cluster_id
    ]["dc_long"]

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
# SAVE PHASE 1 OUTPUTS
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
    "purple",
    "orange",
    "darkred"
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
            f"Distance: "
            f"{row['distance_to_dc_km']:.2f} km"
        )
    ).add_to(m)

# =====================================================
# PLOT DC LOCATIONS
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
# PHASE 2
# DC1 MILK RUN OPTIMIZATION
# =====================================================

dc1_df = df[
    df["assigned_dc"] == TARGET_DC
].copy()

# =====================================================
# GET DC1 LOCATION
# =====================================================

selected_dc = dc_locations[
    dc_locations["dc_id"] == TARGET_DC
].iloc[0]

DC_LAT = selected_dc["dc_lat"]

DC_LON = selected_dc["dc_long"]

# =====================================================
# DISTANCE FROM DC1
# =====================================================

dc1_df["distance_from_dc"] = dc1_df.apply(

    lambda row: haversine(
        DC_LAT,
        DC_LON,
        row[LAT_COL],
        row[LON_COL]
    ),

    axis=1
)

# =====================================================
# SORT STORES
# =====================================================

dc1_df = dc1_df.sort_values(
    by="distance_from_dc",
    ascending=False
).reset_index(drop=True)

# =====================================================
# SORT TRUCKS
# =====================================================

truck_df = truck_df.sort_values(
    by="capacity_cft"
).reset_index(drop=True)

# =====================================================
# BUILD ROUTES
# =====================================================

used_stores = set()

routes = []

route_id = 1

for idx, row in dc1_df.iterrows():

    if row[STORE_COL] in used_stores:
        continue

    route_stores = []

    route_load = 0

    max_distance = row["distance_from_dc"]

    # =================================================
    # START ROUTE
    # =================================================

    route_stores.append(row)

    route_load += row[DEMAND_COL]

    used_stores.add(
        row[STORE_COL]
    )

    # =================================================
    # ADD MORE STORES
    # =================================================

    for jdx, next_row in dc1_df.iterrows():

        if next_row[STORE_COL] in used_stores:
            continue

        next_distance = next_row[
            "distance_from_dc"
        ]

        if next_distance > MAX_ROUTE_DISTANCE:
            continue

        tentative_load = (
            route_load
            + next_row[DEMAND_COL]
        )

        feasible_trucks = truck_df[
            truck_df["capacity_cft"]
            >= tentative_load
        ]

        if len(feasible_trucks) == 0:
            continue

        route_stores.append(next_row)

        route_load = tentative_load

        max_distance = max(
            max_distance,
            next_distance
        )

        used_stores.add(
            next_row[STORE_COL]
        )

    # =================================================
    # SELECT OPTIMAL TRUCK
    # =================================================

    feasible_trucks = truck_df[
        truck_df["capacity_cft"]
        >= route_load
    ].copy()

    selected_truck = feasible_trucks.iloc[0]

    truck_capacity = (
        selected_truck["capacity_cft"]
    )

    fixed_cost = (
        selected_truck["fixed_cost"]
    )

    variable_cost = (
        selected_truck[
            "variable_cost_per_km"
        ]
    )

    # =================================================
    # ROUTE DISTANCE
    # =================================================

    route_distance = max_distance * 2

    # =================================================
    # UTILIZATION
    # =================================================

    utilization = (
        route_load / truck_capacity
    ) * 100

    # =================================================
    # MONTHLY COST
    # =================================================

    monthly_cost = (
        fixed_cost
        + variable_cost * route_distance
    ) * MONTHLY_MULTIPLIER

    # =================================================
    # STORE SEQUENCE
    # =================================================

    store_sequence = " -> ".join(

        [
            s[STORE_COL]
            for s in route_stores
        ]
    )

    routes.append({

        "route_id":
            f"R{route_id}",

        "dc":
            TARGET_DC,

        "stores_served":
            store_sequence,

        "number_of_stores":
            len(route_stores),

        "total_demand_cft":
            round(route_load, 2),

        "truck_selected":
            selected_truck["truck_type"],

        "truck_capacity_cft":
            truck_capacity,

        "truck_utilization_percent":
            round(utilization, 2),

        "route_distance_km":
            round(route_distance, 2),

        "monthly_cost":
            round(monthly_cost, 2)
    })

    route_id += 1

# =====================================================
# SAVE PHASE 2 OUTPUT
# =====================================================

routes_df = pd.DataFrame(routes)

routes_df.to_excel(
    "dc1_milk_run_routes.xlsx",
    index=False
)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n================================")
print("PHASE 1 + PHASE 2 COMPLETED")
print("================================")

print(f"\nNumber of DCs = {K}")

print(
    f"\nDC1 Total Routes = "
    f"{len(routes_df)}"
)

print(
    f"\nDC1 Total Monthly Cost = "
    f"{routes_df['monthly_cost'].sum():,.2f}"
)

print(
    f"\nAverage Utilization = "
    f"{routes_df['truck_utilization_percent'].mean():.2f}%"
)

print("\nGenerated Files:")

print("1. clustered_output.xlsx")
print("2. dc_locations.xlsx")
print("3. store_dc_distances.xlsx")
print("4. dc1_milk_run_routes.xlsx")
print("5. index.html")
