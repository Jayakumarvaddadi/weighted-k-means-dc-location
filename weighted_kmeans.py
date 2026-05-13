# =====================================================
# REPLACE YOUR ENTIRE "EXTRACT ROUTES" SECTION
# WITH THIS IMPROVED VERSION
# =====================================================

# =====================================================
# HELPER FUNCTION
# GET ACTUAL ROUTE DISTANCE
# =====================================================

def calculate_actual_route_distance(route_nodes):

    if len(route_nodes) == 0:
        return 0

    total_distance = 0

    # ---------------------------------------------
    # DC -> FIRST STORE
    # ---------------------------------------------

    first_store = route_nodes[0]

    first_row = dc1_df[
        dc1_df[STORE_COL] == first_store
    ].iloc[0]

    total_distance += haversine(

        DC_LAT,
        DC_LON,

        float(first_row[LAT_COL]),
        float(first_row[LON_COL])
    )

    # ---------------------------------------------
    # STORE -> STORE
    # ---------------------------------------------

    for i in range(len(route_nodes) - 1):

        current_store = route_nodes[i]
        next_store = route_nodes[i + 1]

        current_row = dc1_df[
            dc1_df[STORE_COL] == current_store
        ].iloc[0]

        next_row = dc1_df[
            dc1_df[STORE_COL] == next_store
        ].iloc[0]

        total_distance += haversine(

            float(current_row[LAT_COL]),
            float(current_row[LON_COL]),

            float(next_row[LAT_COL]),
            float(next_row[LON_COL])
        )

    # ---------------------------------------------
    # LAST STORE -> DC
    # ---------------------------------------------

    last_store = route_nodes[-1]

    last_row = dc1_df[
        dc1_df[STORE_COL] == last_store
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

        # =========================================
        # GET ROUTE SEQUENCE
        # =========================================

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

        # =========================================
        # SKIP EMPTY ROUTES
        # =========================================

        if len(route_nodes) == 0:
            continue

        # =========================================
        # ACTUAL ROUTE DISTANCE
        # =========================================

        route_distance = (
            calculate_actual_route_distance(
                route_nodes
            )
        )

        # =========================================
        # HARD DISTANCE CHECK
        # =========================================

        if route_distance > MAX_ROUTE_DISTANCE:

            print(
                f"\nRoute {route_id} "
                f"violated distance constraint."
            )

            continue

        # =========================================
        # OPTIMAL TRUCK SELECTION
        # TRUE MINIMUM COST TRUCK
        # =========================================

        feasible_trucks = truck_df[
            truck_df["capacity_cft"]
            >= route_load
        ].copy()

        # -----------------------------------------
        # CALCULATE TRUE MONTHLY COST
        # FOR EACH FEASIBLE TRUCK
        # -----------------------------------------

        feasible_trucks[
            "monthly_cost"
        ] = (

            feasible_trucks["fixed_cost"]

            +

            feasible_trucks[
                "variable_cost_per_km"
            ] * route_distance

        ) * MONTHLY_MULTIPLIER

        # -----------------------------------------
        # SELECT LOWEST COST TRUCK
        # -----------------------------------------

        feasible_trucks = feasible_trucks.sort_values(
            by="monthly_cost"
        )

        selected_truck = (
            feasible_trucks.iloc[0]
        )

        # =========================================
        # TRUCK DETAILS
        # =========================================

        truck_capacity = int(round(
            selected_truck["capacity_cft"]
        ))

        fixed_cost = float(
            selected_truck["fixed_cost"]
        )

        variable_cost = float(
            selected_truck[
                "variable_cost_per_km"
            ]
        )

        monthly_cost = (

            fixed_cost

            +

            variable_cost * route_distance

        ) * MONTHLY_MULTIPLIER

        # =========================================
        # UTILIZATION
        # =========================================

        utilization = (
            route_load / truck_capacity
        ) * 100

        # =========================================
        # SAVE ROUTE
        # =========================================

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

            "fixed_cost":
                round(fixed_cost, 2),

            "variable_cost_per_km":
                round(variable_cost, 2),

            "monthly_cost":
                round(monthly_cost, 2)
        })

        route_id += 1

# =====================================================
# SAVE FINAL OUTPUT
# =====================================================

routes_df = pd.DataFrame(routes)

# =====================================================
# SORT BY MONTHLY COST
# =====================================================

routes_df = routes_df.sort_values(
    by="monthly_cost"
)

routes_df.to_excel(
    "dc1_milk_run_routes.xlsx",
    index=False
)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n================================")
print("ADVANCED MILK RUN OPTIMIZATION")
print("================================")

print(
    f"\nTotal Routes = "
    f"{len(routes_df)}"
)

if len(routes_df) > 0:

    print(
        f"\nTotal Monthly Cost = "
        f"{routes_df['monthly_cost'].sum():,.2f}"
    )

    print(
        f"\nAverage Utilization = "
        f"{routes_df['truck_utilization_percent'].mean():.2f}%"
    )

    print(
        f"\nAverage Route Distance = "
        f"{routes_df['route_distance_km'].mean():.2f} km"
    )

    print(
        f"\nMaximum Route Distance = "
        f"{routes_df['route_distance_km'].max():.2f} km"
    )

else:

    print("\nNo feasible routes found.")

print("\nGenerated File:")
print("dc1_milk_run_routes.xlsx")
