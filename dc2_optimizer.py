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

TARGET_DC = "DC_2"

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
# FILTER DC2 STORES
# =====================================================

dc2_df = df[
    df["assigned_dc"] == TARGET_DC
].copy()

print("\nDC2 STORES FOUND:", len(dc2_df))

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

# DEPOT NODE

locations.append(
    (DC_LAT, DC_LON)
)

demands.append(0)

store_names.append(TARGET_DC)

# STORE NODES

for _, row in dc2_df.iterrows():

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
# HETEROGENEOUS FLEET
# =====================================================

vehicle_capacities = []
vehicle_fixed_costs = []
vehicle_variable_costs = []
vehicle_names = []

# CREATE MULTIPLE VEHICLES OF EACH TYPE

for _, row in truck_df.iterrows():

    for i in range(10):

        vehicle_capacities.append(
            int(round(row["capacity_cft"]))
        )

        vehicle_fixed_costs.append(
            int(round(row["fixed_cost"]))
        )

        vehicle_variable_costs.append(
            float(row["variable_cost_per_km"])
        )

        vehicle_names.append(
            str(row["truck_type"])
        )

vehicle_count = len(vehicle_capacities)

print("\nTOTAL VEHICLES:", vehicle_count)

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
# VEHICLE-SPECIFIC COST CALLBACKS
# =====================================================

transit_callback_indices = []

for vehicle_id in range(vehicle_count):

    variable_cost = vehicle_variable_costs[vehicle_id]

    def vehicle_distance_callback(
        from_index,
        to_index,
        variable_cost=variable_cost
    ):

        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        distance = data["distance_matrix"][
            from_node
        ][to_node]

        return int(
            distance * variable_cost
        )

    callback_index = (
        routing.RegisterTransitCallback(
            vehicle_distance_callback
        )
    )

    transit_callback_indices.append(
        callback_index
    )

    routing.SetArcCostEvaluatorOfVehicle(
        callback_index,
        vehicle_id
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
    transit_callback_indices[0],
    0,
    int(MAX_ROUTE_DISTANCE),
    True,
    "Distance"
)

# =====================================================
# FIXED COSTS
# =====================================================

for vehicle_id in range(vehicle_count):

    routing.SetFixedCostOfVehicle(
        int(vehicle_fixed_costs[vehicle_id]),
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

search_parameters.time_limit.seconds = 90

# =====================================================
# SOLVE
# =====================================================

solution = routing.SolveWithParameters(
    search_parameters
)

if solution:

    print("\nHFVRP Solution Found")

else:

    print("\nNo Solution Found")

# =====================================================
# EXTRACT ROUTES
# =====================================================

routes = []

route_id = 1

if solution:

    for vehicle_id in range(vehicle_count):

        index = routing.Start(vehicle_id)

        route_load = 0
        route_distance = 0

        route_nodes = []

        while not routing.IsEnd(index):

            node = manager.IndexToNode(index)

            previous_index = index

            index = solution.Value(
                routing.NextVar(index)
            )

            next_node = manager.IndexToNode(index)

            route_distance += distance_matrix[
                node
            ][next_node]

            if node != 0:

                route_nodes.append(
                    store_names[node]
                )

                route_load += demands[node]

        if len(route_nodes) == 0:
            continue

        truck_capacity = vehicle_capacities[
            vehicle_id
        ]

        utilization = (
            route_load / truck_capacity
        ) * 100

        monthly_cost = (

            vehicle_fixed_costs[vehicle_id]

            +

            route_distance
            * vehicle_variable_costs[vehicle_id]

        ) * MONTHLY_MULTIPLIER

        routes.append({

            "route_id":
                f"R{route_id}",

            "dc":
                TARGET_DC,

            "truck_selected":
                vehicle_names[vehicle_id],

            "truck_capacity_cft":
                truck_capacity,

            "stores_served":
                " -> ".join(route_nodes),

            "number_of_stores":
                len(route_nodes),

            "total_demand_cft":
                round(route_load, 2),

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

routes_df = routes_df.sort_values(
    by="monthly_cost"
)

routes_df.to_excel(
    "dc2_hfvrp_routes.xlsx",
    index=False
)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n================================")
print("DC2 HFVRP OPTIMIZATION")
print("================================")

print(
    "\nTotal Routes:",
    len(routes_df)
)

if len(routes_df) > 0:

    print(
        "\nTotal Monthly Cost:",
        round(
            routes_df["monthly_cost"].sum(),
            2
        )
    )

    print(
        "\nAverage Utilization:",
        round(
            routes_df[
                "truck_utilization_percent"
            ].mean(),
            2
        )
    )

print("\nGenerated File:")
print("dc2_hfvrp_routes.xlsx")
