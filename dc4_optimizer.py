# =====================================================
# DC4 MILK RUN OPTIMIZATION
# =====================================================

import pandas as pd
from math import radians, sin, cos, sqrt, atan2

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# =====================================================
# LOAD FILES
# =====================================================

input_file = "clustered_output.xlsx"
truck_file = "truck_master.xlsx"
dc_file = "dc_locations.xlsx"

df = pd.read_excel(input_file)
truck_df = pd.read_excel(truck_file)
dc_locations = pd.read_excel(dc_file)

# =====================================================
# REQUIRED COLUMNS
# =====================================================

LAT_COL = "lat"
LON_COL = "long"
STORE_COL = "store"
DEMAND_COL = "demand_cft"

# =====================================================
# PARAMETERS
# =====================================================

TARGET_DC = "DC_4"

MAX_ROUTE_DISTANCE = 1400

MONTHLY_MULTIPLIER = 10

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
# FILTER DC4 STORES
# =====================================================

dc4_df = df[
    df["assigned_dc"] == TARGET_DC
].copy()

print("\nDC4 STORES FOUND:", len(dc4_df))

# =====================================================
# GET DC LOCATION
# =====================================================

selected_dc = dc_locations[
    dc_locations["dc_id"] == TARGET_DC
].iloc[0]

DC_LAT = float(selected_dc["dc_lat"])
DC_LON = float(selected_dc["dc_long"])

# =====================================================
# CREATE NODES
# =====================================================

locations = []
demands = []
store_names = []

locations.append((DC_LAT, DC_LON))
demands.append(0)
store_names.append(TARGET_DC)

for _, row in dc4_df.iterrows():

    locations.append(
        (
            float(row[LAT_COL]),
            float(row[LON_COL])
        )
    )

    demands.append(
        int(round(row[DEMAND_COL]))
    )

    store_names.append(
        str(row[STORE_COL])
    )

# =====================================================
# DISTANCE MATRIX
# =====================================================

num_nodes = len(locations)

distance_matrix = []

for i in range(num_nodes):

    row_distances = []

    for j in range(num_nodes):

        dist = int(round(

            haversine(

                float(locations[i][0]),
                float(locations[i][1]),

                float(locations[j][0]),
                float(locations[j][1])
            )
        ))

        row_distances.append(dist)

    distance_matrix.append(row_distances)

# =====================================================
# VEHICLE CONFIGURATION
# =====================================================

largest_capacity = int(
    truck_df["capacity_cft"].max()
)

vehicle_count = len(dc4_df)

vehicle_capacities = [
    largest_capacity
] * vehicle_count

# =====================================================
# DATA MODEL
# =====================================================

data = {}

data["distance_matrix"] = distance_matrix
data["demands"] = demands
data["vehicle_capacities"] = vehicle_capacities
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

routing = pywrapcp.RoutingModel(manager)

# =====================================================
# DISTANCE CALLBACK
# =====================================================

def distance_callback(from_index, to_index):

    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)

    return int(
        data["distance_matrix"][
            from_node
        ][to_node]
    )

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

    from_node = manager.IndexToNode(from_index)

    return int(
        data["demands"][from_node]
    )

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
    vehicle_capacities,
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
# MINIMIZE VEHICLES
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
# ROUTE DISTANCE FUNCTION
# =====================================================

def calculate_actual_route_distance(route_nodes):

    if len(route_nodes) == 0:
        return 0

    total_distance = 0

    first_store = route_nodes[0]

    first_row = dc4_df[
        dc4_df[STORE_COL] == first_store
    ].iloc[0]

    total_distance += haversine(
        DC_LAT,
        DC_LON,
        float(first_row[LAT_COL]),
        float(first_row[LON_COL])
    )

    for i in range(len(route_nodes) - 1):

        current_store = route_nodes[i]
        next_store = route_nodes[i + 1]

        current_row = dc4_df[
            dc4_df[STORE_COL] == current_store
        ].iloc[0]

        next_row = dc4_df[
            dc4_df[STORE_COL] == next_store
        ].iloc[0]

        total_distance += haversine(
            float(current_row[LAT_COL]),
            float(current_row[LON_COL]),
            float(next_row[LAT_COL]),
            float(next_row[LON_COL])
        )

    last_store = route_nodes[-1]

    last_row = dc4_df[
        dc4_df[STORE_COL] == last_store
    ].iloc[0]

    total_distance += haversine(
        float(last_row[LAT_COL]),
        float(last_row[LON_COL]),
        DC_LAT,
        DC_LON
    )

    return total_distance

# =====================================================
# EXTRACT ROUTES
# =====================================================

routes = []

route_id = 1

if solution:

    for vehicle_id in range(vehicle_count):

        index = routing.Start(vehicle_id)

        route_load = 0

        route_nodes = []

        while not routing.IsEnd(index):

            node = manager.IndexToNode(index)

            if node != 0:

                route_nodes.append(
                    store_names[node]
                )

                route_load += int(
                    data["demands"][node]
                )

            index = solution.Value(
                routing.NextVar(index)
            )

        if len(route_nodes) == 0:
            continue

        route_distance = (
            calculate_actual_route_distance(
                route_nodes
            )
        )

        feasible_trucks = truck_df[
            truck_df["capacity_cft"]
            >= route_load
        ].copy()

        feasible_trucks["monthly_cost"] = (

            feasible_trucks["fixed_cost"]

            +

            feasible_trucks[
                "variable_cost_per_km"
            ] * route_distance

        ) * MONTHLY_MULTIPLIER

        feasible_trucks = feasible_trucks.sort_values(
            by="monthly_cost"
        )

        selected_truck = feasible_trucks.iloc[0]

        truck_capacity = int(
            selected_truck["capacity_cft"]
        )

        utilization = (
            route_load / truck_capacity
        ) * 100

        monthly_cost = (
            selected_truck["fixed_cost"]
            +
            selected_truck["variable_cost_per_km"]
            * route_distance
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
# SAVE OUTPUT
# =====================================================

routes_df = pd.DataFrame(routes)

routes_df.to_excel(
    "dc4_milk_run_routes.xlsx",
    index=False
)

print("\nGenerated File:")
print("dc4_milk_run_routes.xlsx")
