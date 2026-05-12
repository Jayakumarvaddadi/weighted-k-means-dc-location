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
# LOAD INPUT FILES
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
# DELETE OLD OUTPUTS
# =====================================================

old_files = [

    "clustered_output.xlsx",

    "dc_locations.xlsx",

    "store_dc_distances.xlsx",

    "model_summary.xlsx",

    "secondary_logistics_routes.xlsx",

    "secondary_logistics_summary.xlsx",

    "store_monthly_logistics_cost.xlsx",

    "fleet_summary.xlsx",

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
    # DISTANCE CALCULATION
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
# SAVE PRIMARY OUTPUTS
# =====================================================

df.to_excel(
    "clustered_output.xlsx",
    index=False
)

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

centroids_df = pd.DataFrame(
    centroids,
    columns=["lat", "long"]
)

centroids_df.to_excel(
    "dc_locations.xlsx",
    index=False
)

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
# INITIALIZE MAP
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

MAX_ROUTE_DISTANCE = 1700

# =====================================================
# CLUSTER-WISE ORTOOLS
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

    locations = [(dc_lat, dc_long)]

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
   # HETEROGENEOUS FLEET
# =================================================

vehicle_capacities = []
vehicle_fixed_costs = []
vehicle_variable_costs = []
vehicle_names = []

# Create multiple copies of each truck type

for _, truck in truck_df.iterrows():

    for i in range(15):

        vehicle_capacities.append(
            int(truck["capacity_cft"])
        )

        vehicle_fixed_costs.append(
            float(truck["fixed_cost"])
        )

        vehicle_variable_costs.append(
            float(truck["variable_cost_per_km"])
        )

        vehicle_names.append(
            truck["truck_type"]
        )

num_vehicles = len(vehicle_capacities)
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

    # =================================================
# VEHICLE-SPECIFIC COSTS
# =================================================

cost_callback_indices = []

for vehicle_id in range(num_vehicles):

    variable_cost = vehicle_variable_costs[
        vehicle_id
    ]

    def vehicle_cost_callback(
        from_index,
        to_index,
        vc=variable_cost
    ):

        from_node = manager.IndexToNode(
            from_index
        )

        to_node = manager.IndexToNode(
            to_index
        )

        distance = distance_matrix[
            from_node
        ][
            to_node
        ]

        return int(distance * vc)

    callback_index = routing.RegisterTransitCallback(
        vehicle_cost_callback
    )

    cost_callback_indices.append(
        callback_index
    )

    routing.SetArcCostEvaluatorOfVehicle(
        callback_index,
        vehicle_id
    )
    # =================================================
# FIXED VEHICLE COSTS
# =================================================

   for vehicle_id in range(num_vehicles):

     routing.SetFixedCostOfVehicle(

        int(vehicle_fixed_costs[vehicle_id]),

        vehicle_id
    )
    # =================================================
    # PENALIZE EXTRA TRUCKS
    # =================================================

    for vehicle_id in range(num_vehicles):

        routing.SetFixedCostOfVehicle(
            50000,
            vehicle_id
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
    # DISTANCE CONSTRAINT
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

    search_parameters.time_limit.seconds = 300

    # =================================================
    # SOLVE
    # =================================================

    solution = routing.SolveWithParameters(
        search_parameters
    )

    # =================================================
    # FALLBACK
    # =================================================

    if solution is None:

        print(f"Fallback activated for Cluster {cluster_id}")

        for _, store_row in cluster_df.iterrows():

            route_distance = int(
                store_row["distance_km"] * 2
            )

            route_load = store_row["demand_cft"]

            feasible_trucks = truck_df[
                truck_df["capacity_cft"] >= route_load
            ].copy()

            if len(feasible_trucks) == 0:

                selected_truck = truck_df.sort_values(
                    "capacity_cft"
                ).iloc[-1]

            else:

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

                selected_truck = feasible_trucks.sort_values(
                    "estimated_cost"
                ).iloc[0]

            utilization = (

                route_load
                /
                selected_truck["capacity_cft"]

            ) * 100

            monthly_route_cost = (

                selected_truck["fixed_cost"]

                +

                (
                    selected_truck["variable_cost_per_km"]

                    *
                    route_distance
                    *
                    10
                )
            )

            routes_output.append({

                "cluster": cluster_id,

                "truck_type":
                    selected_truck["truck_type"],

                "stores_served": 1,

                "store_list":
                    store_row["store"],

                "route_load_cft":
                    route_load,

                "truck_capacity_cft":
                    selected_truck["capacity_cft"],

                "truck_utilization_percent":
                    round(utilization, 2),

                "route_distance_km":
                    route_distance,

                "monthly_route_cost":
                    monthly_route_cost
            })

            store_level_cost_output.append({

                "cluster": cluster_id,

                "store": store_row["store"],

                "truck_type":
                    selected_truck["truck_type"],

                "allocated_monthly_logistics_cost":
                    monthly_route_cost
            })

        continue

    # =================================================
    # EXTRACT ROUTES
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

        if len(route_stores) == 0:

            continue

        route_coordinates.append([
            dc_lat,
            dc_long
        ])

        # =================================================
        # BEST TRUCK SELECTION
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

        utilization = (

            route_load
            /
            selected_truck["capacity_cft"]

        ) * 100

        monthly_route_cost = (

            selected_truck["fixed_cost"]

            +

            (
                selected_truck["variable_cost_per_km"]

                *
                route_distance
                *
                10
            )
        )

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
                round(utilization, 2),

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

                "allocated_monthly_logistics_cost":
                    allocated_cost
            })

        # =================================================
        # MAP ROUTES
        # =================================================

        folium.PolyLine(

            locations=route_coordinates,

            color=cluster_colors[cluster_id],

            weight=3,

            opacity=0.8

        ).add_to(map_india)

# =====================================================
# SAVE ROUTE OUTPUTS
# =====================================================

routes_df = pd.DataFrame(
    routes_output
)

routes_df.to_excel(
    "secondary_logistics_routes.xlsx",
    index=False
)

store_cost_df = pd.DataFrame(
    store_level_cost_output
)

store_cost_df.to_excel(
    "store_monthly_logistics_cost.xlsx",
    index=False
)

# =====================================================
# FLEET SUMMARY
# =====================================================

fleet_summary = routes_df.groupby(
    "truck_type"
).agg({

    "truck_type": "count",

    "monthly_route_cost": "sum",

    "route_distance_km": "sum",

    "route_load_cft": "sum",

    "truck_utilization_percent": "mean"
})

fleet_summary.columns = [

    "number_of_trucks_used",

    "total_monthly_cost",

    "total_route_distance_km",

    "total_load_cft",

    "average_utilization_percent"
]

fleet_summary = fleet_summary.reset_index()

fleet_summary.to_excel(
    "fleet_summary.xlsx",
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
    ]
})

summary_secondary.to_excel(
    "secondary_logistics_summary.xlsx",
    index=False
)

# =====================================================
# SAVE MAP
# =====================================================

map_india.save(
    "optimized_routes_map.html"
)

print("\nOptimization Completed Successfully")
