import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from math import radians, sin, cos, sqrt, atan2

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

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

MAX_ROUTE_DISTANCE = 1400
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
# PHASE 1
# WEIGHTED K-MEANS
# =====================================================

X = df[[LAT_COL, LON_COL]].values
weights = df[WEIGHT_COL].values

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

kmeans.fit(
    X,
    sample_weight=weights
)

df["cluster"] = kmeans.labels_

# =====================================================
# FORCE DCS TO REAL STORE LOCATIONS
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
# STORE TO DC DISTANCE
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
# CREATE MAP
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
            f"Assigned DC: {row['assigned_dc']}"
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

m.save("index.html")

# =====================================================
# PHASE 2
# TRUE MILK RUN OPTIMIZATION
# USING OR-TOOLS
# =====================================================

dc1_df = df[
    df["assigned_dc"] == TARGET_DC
].copy()

# =====================================================
# GET DC LOCATION
# =====================================================

selected_dc = dc_locations[
    dc_locations["dc_id"] == TARGET_DC
].iloc[0]

DC_LAT = selected_dc["dc_lat"]
DC_LON = selected_dc["dc_long"]

# =====================================================
# CREATE NODES
# =====================================================

locations = []
demands = []
store_names = []

# DC NODE
locations.append(
    (DC_LAT, DC_LON)
)

demands.append(0)

store_names.append(TARGET_DC)

# STORE NODES
for _, row in dc1_df.iterrows():

    locations.append(
        (
            row[LAT_COL],
            row[LON_COL]
        )
    )

    demands.append(
        int(row[DEMAND_COL])
    )

    store_names.append(
        row[STORE_COL]
    )

# =====================================================
# DISTANCE MATRIX
# STORE TO STORE DISTANCES
# =====================================================

num_nodes = len(locations)

distance_matrix = np.zeros(
    (num_nodes, num_nodes)
)

for i in range(num_nodes):

    for j in range(num_nodes):

        distance_matrix[i][j] = int(

            haversine(

                locations[i][0],
                locations[i][1],

                locations[j][0],
                locations[j][1]
            )
        )

# =====================================================
# VEHICLE CONFIGURATION
# =====================================================

largest_capacity = truck_df[
    "capacity_cft"
].max()

vehicle_count = len(dc1_df)

vehicle_capacities = [
    int(largest_capacity)
] * vehicle_count

# =====================================================
# DATA MODEL
# =====================================================

data = {}

data["distance_matrix"] = (
    distance_matrix.tolist()
)

data["demands"] = demands

data["vehicle_capacities"] = (
    vehicle_capacities
)

data["num_vehicles"] = vehicle_count

data["depot"] = 0

# =====================================================
# ROUTING MANAGER
# =====================================================

manager = pywrapcp.RoutingIndexManager(
    len(data["distance_matrix"]),
    data["num_vehicles"],
    data["depot"]
)

routing = pywrapcp.RoutingModel(
    manager
)

# =====================================================
# DISTANCE CALLBACK
# =====================================================

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

    return data["distance_matrix"][
        from_node
    ][to_node]

transit_callback_index = (
    routing.RegisterTransitCallback(
        distance_callback
    )
)

routing.SetArcCostEvaluatorOfAllVehicles(
    transit_callback_index
)

# =====================================================
# DEMAND CALLBACK
# =====================================================

def demand_callback(from_index):

    from_node = manager.IndexToNode(
        from_index
    )

    return data["demands"][from_node]

demand_callback_index = (
    routing.RegisterUnaryTransitCallback(
        demand_callback
    )
)

# =====================================================
# CAPACITY CONSTRAINT
# =====================================================

routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,
    data["vehicle_capacities"],
    True,
    "Capacity"
)

# =====================================================
# DISTANCE CONSTRAINT
# =====================================================

routing.AddDimension(
    transit_callback_index,
    0,
    MAX_ROUTE_DISTANCE,
    True,
    "Distance"
)

# =====================================================
# MINIMIZE NUMBER OF TRUCKS
# =====================================================

for vehicle_id in range(vehicle_count):

    routing.SetFixedCostOfVehicle(
        100000,
        vehicle_id
    )

# =====================================================
# SEARCH PARAMETERS
# =====================================================

search_parameters = (
    pywrapcp.DefaultRoutingSearchParameters()
)

search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)

search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)

search_parameters.time_limit.seconds = 60

# =====================================================
# SOLVE
# =====================================================

solution = routing.SolveWithParameters(
    search_parameters
)

# =====================================================
# EXTRACT ROUTES
# =====================================================

routes = []

route_id = 1

if solution:

    for vehicle_id in range(vehicle_count):

        index = routing.Start(vehicle_id)

        route_distance = 0
        route_load = 0

        route_nodes = []

        while not routing.IsEnd(index):

            node = manager.IndexToNode(index)

            if node != 0:

                route_nodes.append(
                    store_names[node]
                )

                route_load += (
                    data["demands"][node]
                )

            previous_index = index

            index = solution.Value(
                routing.NextVar(index)
            )

            route_distance += (
                routing.GetArcCostForVehicle(
                    previous_index,
                    index,
                    vehicle_id
                )
            )

        if len(route_nodes) == 0:
            continue

        # =================================================
        # SELECT OPTIMAL TRUCK
        # =================================================

        feasible_trucks = truck_df[
            truck_df["capacity_cft"]
            >= route_load
        ].copy()

        selected_truck = (
            feasible_trucks.iloc[0]
        )

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

        utilization = (
            route_load / truck_capacity
        ) * 100

        monthly_cost = (
            fixed_cost
            + variable_cost * route_distance
        ) * MONTHLY_MULTIPLIER

        routes.append({

            "route_id":
                f"R{route_id}",

            "dc":
                TARGET_DC,

            "stores_served":
                " -> ".join(route_nodes),

            "number_of_stores":
                len(route_nodes),

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
# SAVE ROUTES
# =====================================================

routes_df = pd.DataFrame(routes)

routes_df.to_excel(
    "dc1_milk_run_routes.xlsx",
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n================================")
print("FULL LOGISTICS OPTIMIZATION DONE")
print("================================")

print(
    f"\nTotal DC1 Routes = "
    f"{len(routes_df)}"
)

print(
    f"\nTotal Monthly Cost = "
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
