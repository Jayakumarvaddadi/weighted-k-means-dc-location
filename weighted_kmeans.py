import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

from sklearn.cluster import KMeans

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =====================================================
# PRINT FILES
# =====================================================

print("FILES IN DIRECTORY:")
print(os.listdir())

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

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# =====================================================
# LOAD FILES
# =====================================================

df = pd.read_excel("saavu2.xlsx")

truck_df = pd.read_excel("truck_master.xlsx")

# =====================================================
# LOWERCASE COLUMNS
# =====================================================

df.columns = df.columns.str.lower()

truck_df.columns = truck_df.columns.str.lower()

# =====================================================
# CLEAN DATA
# =====================================================

df = df[
    (df["lat"] != 0)
    &
    (df["long"] != 0)
]

df = df.dropna(
    subset=[
        "lat",
        "long",
        "sales",
        "demand_cft"
    ]
)

df["sales"] = pd.to_numeric(
    df["sales"],
    errors="coerce"
).fillna(1)

df["sales"] = df["sales"].clip(lower=1)

df["demand_cft"] = pd.to_numeric(
    df["demand_cft"],
    errors="coerce"
).fillna(1)

# =====================================================
# DELETE OLD FILES
# =====================================================

old_files = [

    "clustered_output.xlsx",

    "dc_locations.xlsx",

    "store_dc_distances.xlsx",

    "model_summary.xlsx",

    "secondary_logistics_routes.xlsx",

    "secondary_logistics_summary.xlsx",

    "store_monthly_logistics_cost.xlsx",

    "optimized_routes_map.html"
]

for file in old_files:

    if os.path.exists(file):

        os.remove(file)

# =====================================================
# PREPARE KMEANS
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

for k in range(2, 20):

    print(f"\nTrying K = {k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(
        X,
        sample_weight=weights
    )

    temp_df = df.copy()

    temp_df["cluster"] = kmeans.labels_

    centroids = kmeans.cluster_centers_

    # =================================================
    # SNAP TO NEAREST REAL STORE
    # =================================================

    real_centroids = []

    for center in centroids:

        distances = np.sqrt(

            (X[:, 0] - center[0]) ** 2 +

            (X[:, 1] - center[1]) ** 2

        )

        nearest_idx = np.argmin(distances)

        real_centroids.append(
            X[nearest_idx]
        )

    centroids = np.array(real_centroids)

    # =================================================
    # ASSIGN DC COORDINATES
    # =================================================

    temp_df["dc_lat"] = temp_df["cluster"].apply(
        lambda x: centroids[x][0]
    )

    temp_df["dc_long"] = temp_df["cluster"].apply(
        lambda x: centroids[x][1]
    )

    # =================================================
    # DISTANCES
    # =================================================

    temp_df["distance_km"] = temp_df.apply(

        lambda row: haversine(

            row["lat"],
            row["long"],

            row["dc_lat"],
            row["dc_long"]

        ),

        axis=1
    )

    max_distance = temp_df["distance_km"].max()

    print(f"Max Distance = {max_distance:.2f} km")

    if max_distance <= MAX_STORE_DISTANCE:

        best_k = k
        best_df = temp_df
        best_centroids = centroids

        print(f"Feasible K Found = {k}")

        break

# =====================================================
# FINAL DATA
# =====================================================

df = best_df.copy()

centroids = best_centroids.copy()

# =====================================================
# RENUMBER CLUSTERS
# =====================================================

unique_clusters = sorted(
    df["cluster"].unique()
)

cluster_mapping = {

    old: new

    for new, old in enumerate(unique_clusters)
}

df["cluster"] = df["cluster"].map(
    cluster_mapping
)

centroids = np.array([

    centroids[old]

    for old in unique_clusters
])

# =====================================================
# SAVE CLUSTER OUTPUT
# =====================================================

df.to_excel(
    "clustered_output.xlsx",
    index=False
)

# =====================================================
# STORE DISTANCE OUTPUT
# =====================================================

distance_output = df[[

    "store",

    "cluster",

    "lat",

    "long",

    "dc_lat",

    "dc_long",

    "distance_km",

    "sales",

    "demand_cft"
]]

distance_output.to_excel(
    "store_dc_distances.xlsx",
    index=False
)

# =====================================================
# DC OUTPUT
# =====================================================

centroids_df = pd.DataFrame(

    centroids,

    columns=["lat", "long"]
)

centroids_df.to_excel(
    "dc_locations.xlsx",
    index=False
)

# =====================================================
# MODEL SUMMARY
# =====================================================

summary_df = pd.DataFrame({

    "optimal_k": [best_k],

    "max_store_distance_km": [
        df["distance_km"].max()
    ],

    "avg_store_distance_km": [
        df["distance_km"].mean()
    ]
})

summary_df.to_excel(
    "model_summary.xlsx",
    index=False
)

# =====================================================
# MAP INITIALIZATION
# =====================================================

map_india = folium.Map(
    location=[22.5, 78.9],
    zoom_start=5
)

# =====================================================
# COLORS
# =====================================================

k = df["cluster"].nunique()

colormap = plt.colormaps["viridis"]

cluster_colors = {}

for cluster_id in sorted(df["cluster"].unique()):

    cluster_colors[cluster_id] = mcolors.to_hex(

        colormap(
            cluster_id / max(k - 1, 1)
        )
    )

# =====================================================
# PLOT STORES
# =====================================================

for _, row in df.iterrows():

    color = cluster_colors[row["cluster"]]

    folium.CircleMarker(

        location=[
            row["lat"],
            row["long"]
        ],

        radius=4,

        color=color,

        fill=True,

        fill_color=color,

        fill_opacity=0.8,

        popup=(

            f"Store: {row['store']}<br>"

            f"Cluster: {row['cluster']}<br>"

            f"Distance: {row['distance_km']:.2f} km"

        )

    ).add_to(map_india)

# =====================================================
# PLOT DCS
# =====================================================

for cluster_id, row in centroids_df.iterrows():

    folium.Marker(

        location=[
            row["lat"],
            row["long"]
        ],

        popup=f"DC {cluster_id}",

        icon=folium.Icon(
            color="red",
            icon="star"
        )

    ).add_to(map_india)

# =====================================================
# SECONDARY LOGISTICS
# =====================================================

routes_output = []

store_level_cost_output = []

MAX_ROUTE_DISTANCE = 1400

# =====================================================
# CLUSTER LOOP
# =====================================================

for cluster_id in sorted(df["cluster"].unique()):

    print(f"\nOptimizing Cluster {cluster_id}")

    cluster_df = df[
        df["cluster"] == cluster_id
    ].copy().reset_index(drop=True)

    dc_lat = cluster_df.iloc[0]["dc_lat"]

    dc_long = cluster_df.iloc[0]["dc_long"]

    # =================================================
    # LOCATIONS
    # =================================================

    locations = [

        (dc_lat, dc_long)
    ]

    for _, row in cluster_df.iterrows():

        locations.append(

            (
                row["lat"],
                row["long"]
            )
        )

    # =================================================
    # DISTANCE MATRIX
    # =================================================

    distance_matrix = []

    for from_node in locations:

        row_distance = []

        for to_node in locations:

            dist = haversine(

                from_node[0],
                from_node[1],

                to_node[0],
                to_node[1]
            )

            row_distance.append(
                int(dist)
            )

        distance_matrix.append(
            row_distance
        )

    # =================================================
    # DEMANDS
    # =================================================

    demands = [0]

    for _, row in cluster_df.iterrows():

        demands.append(

            int(row["demand_cft"])
        )

    # =================================================
    # VEHICLES
    # =================================================

    num_vehicles = len(cluster_df)

    largest_capacity = int(
        truck_df["capacity_cft"].max()
    )

    vehicle_capacities = [

        largest_capacity
    ] * num_vehicles

    # =================================================
    # ROUTING MODEL
    # =================================================

    manager = pywrapcp.RoutingIndexManager(

        len(distance_matrix),

        num_vehicles,

        0
    )

    routing = pywrapcp.RoutingModel(
        manager
    )

    # =================================================
    # DISTANCE CALLBACK
    # =================================================

    def distance_callback(
        from_index,
        to_index
    ):

        from_node = manager.IndexToNode(
            from_index
        )

        to_node = manager.IndexToNode(
            to_index
        )

        return distance_matrix[
            from_node
        ][
            to_node
        ]

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    # =================================================
    # DEMAND CALLBACK
    # =================================================

    def demand_callback(from_index):

        from_node = manager.IndexToNode(
            from_index
        )

        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )

    # =================================================
    # CAPACITY CONSTRAINT
    # =================================================

    routing.AddDimensionWithVehicleCapacity(

        demand_callback_index,

        0,

        vehicle_capacities,

        True,

        "Capacity"
    )

    # =================================================
    # ROUTE DISTANCE CONSTRAINT
    # =================================================

    routing.AddDimension(

        transit_callback_index,

        0,

        MAX_ROUTE_DISTANCE,

        True,

        "Distance"
    )

    distance_dimension = routing.GetDimensionOrDie(
        "Distance"
    )

    # =================================================
    # COST MINIMIZATION
    # =================================================

    distance_dimension.SetGlobalSpanCostCoefficient(
        1000
    )

    # =================================================
    # SEARCH PARAMETERS
    # =================================================

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (

        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (

        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.seconds = 180

    # =================================================
    # SOLVE
    # =================================================

    solution = routing.SolveWithParameters(
        search_parameters
    )

    if solution is None:

        print(f"No feasible solution for cluster {cluster_id}")

        continue

    # =================================================
    # ROUTES
    # =================================================

    for vehicle_id in range(num_vehicles):

        index = routing.Start(vehicle_id)

        route_distance = 0

        route_load = 0

        route_stores = []

        route_coordinates = [

            [dc_lat, dc_long]
        ]

        while not routing.IsEnd(index):

            node_index = manager.IndexToNode(index)

            if node_index != 0:

                store_row = cluster_df.iloc[
                    node_index - 1
                ]

                route_stores.append(
                    store_row["store"]
                )

                route_load += store_row[
                    "demand_cft"
                ]

                route_coordinates.append([

                    store_row["lat"],
                    store_row["long"]

                ])

            previous_index = index

            index = solution.Value(
                routing.NextVar(index)
            )

            route_distance += routing.GetArcCostForVehicle(

                previous_index,

                index,

                vehicle_id
            )

        # =================================================
        # SKIP EMPTY ROUTES
        # =================================================

        if len(route_stores) == 0:

            continue

        route_coordinates.append([

            dc_lat,
            dc_long
        ])

        # =================================================
        # SELECT BEST TRUCK
        # =================================================

        feasible_trucks = truck_df[

            truck_df["capacity_cft"] >= route_load

        ].copy()

        if len(feasible_trucks) == 0:

            selected_truck = truck_df.sort_values(
                "capacity_cft"
            ).iloc[-1]

        else:

            feasible_trucks["utilization"] = (

                route_load

                /

                feasible_trucks["capacity_cft"]

            )

            feasible_trucks["estimated_cost"] = (

                feasible_trucks["fixed_cost"]

                +

                (
                    feasible_trucks["variable_cost_per_km"]

                    *

                    route_distance

                    *

                    10
                )
            )

            feasible_trucks["score"] = (

                feasible_trucks["estimated_cost"]

                /

                feasible_trucks["utilization"]

            )

            selected_truck = feasible_trucks.sort_values(
                "score"
            ).iloc[0]

        # =================================================
        # COST CALCULATION
        # =================================================

        fixed_cost = selected_truck[
            "fixed_cost"
        ]

        variable_cost = selected_truck[
            "variable_cost_per_km"
        ]

        monthly_route_cost = (

            fixed_cost

            +

            (
                variable_cost

                *

                route_distance

                *

                10
            )
        )

        utilization = (

            route_load

            /

            selected_truck["capacity_cft"]

        ) * 100

        # =================================================
        # ROUTE OUTPUT
        # =================================================

        routes_output.append({

            "cluster": cluster_id,

            "truck_type":
                selected_truck["truck_type"],

            "stores_served":
                len(route_stores),

            "store_list":
                " -> ".join(route_stores),

            "route_load_cft":
                route_load,

            "truck_capacity_cft":
                selected_truck["capacity_cft"],

            "truck_utilization_percent":
                utilization,

            "route_distance_km":
                route_distance,

            "monthly_route_cost":
                monthly_route_cost
        })

        # =================================================
        # STORE COST ALLOCATION
        # =================================================

        for store_name in route_stores:

            store_data = cluster_df[
                cluster_df["store"] == store_name
            ].iloc[0]

            share = (

                store_data["demand_cft"]

                /

                route_load

            )

            allocated_cost = (

                monthly_route_cost

                *

                share

            )

            store_level_cost_output.append({

                "cluster": cluster_id,

                "store": store_name,

                "truck_type":
                    selected_truck["truck_type"],

                "demand_cft":
                    store_data["demand_cft"],

                "allocated_monthly_logistics_cost":
                    allocated_cost,

                "route_distance_km":
                    route_distance
            })

        # =================================================
        # ROUTE LINES ON MAP
        # =================================================

        folium.PolyLine(

            locations=route_coordinates,

            color=cluster_colors[cluster_id],

            weight=3,

            opacity=0.8,

            popup=(

                f"Cluster: {cluster_id}<br>"

                f"Truck: {selected_truck['truck_type']}<br>"

                f"Route Distance: {route_distance:.2f} km"

            )

        ).add_to(map_india)

# =====================================================
# SAVE ROUTE OUTPUT
# =====================================================

routes_df = pd.DataFrame(
    routes_output
)

routes_df.to_excel(
    "secondary_logistics_routes.xlsx",
    index=False
)

# =====================================================
# SAVE STORE COST OUTPUT
# =====================================================

store_cost_df = pd.DataFrame(
    store_level_cost_output
)

store_cost_df.to_excel(
    "store_monthly_logistics_cost.xlsx",
    index=False
)

# =====================================================
# SECONDARY SUMMARY
# =====================================================

summary_secondary = pd.DataFrame({

    "total_routes": [
        len(routes_df)
    ],

    "total_secondary_cost": [

        routes_df[
            "monthly_route_cost"
        ].sum()
    ],

    "average_route_cost": [

        routes_df[
            "monthly_route_cost"
        ].mean()
    ],

    "total_route_distance_km": [

        routes_df[
            "route_distance_km"
        ].sum()
    ]
})

summary_secondary.to_excel(
    "secondary_logistics_summary.xlsx",
    index=False
)

# =====================================================
# SAVE FINAL MAP
# =====================================================

map_india.save(
    "optimized_routes_map.html"
)

print("\nOptimization Completed Successfully")
