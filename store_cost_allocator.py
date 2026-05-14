import pandas as pd

# =====================================================
# LOAD MASTER STORE DATA
# =====================================================

stores_df = pd.read_excel(
    "clustered_output.xlsx"
)

# =====================================================
# ROUTE FILES
# =====================================================

route_files = [

    "dc1_milk_run_routes.xlsx",
    "dc2_milk_run_routes.xlsx",
    "dc3_milk_run_routes.xlsx",
    "dc4_milk_run_routes.xlsx",
    "dc5_milk_run_routes.xlsx",
    "dc6_milk_run_routes.xlsx"
]

# =====================================================
# FINAL OUTPUT STORAGE
# =====================================================

store_cost_rows = []

# =====================================================
# PROCESS EACH ROUTE FILE
# =====================================================

for file in route_files:

    print("\n================================")
    print(f"PROCESSING FILE: {file}")
    print("================================")

    # =================================================
    # READ FILE
    # =================================================

    try:

        routes_df = pd.read_excel(file)

        print(
            f"Routes Found: {len(routes_df)}"
        )

    except Exception as e:

        print(
            f"\nERROR READING FILE: {file}"
        )

        print(e)

        continue

    # =================================================
    # PROCESS EACH ROUTE
    # =================================================

    for _, route in routes_df.iterrows():

        try:

            route_id = str(
                route["route_id"]
            )

            dc = str(
                route["dc"]
            )

            total_route_demand = float(
                route["total_demand_cft"]
            )

            total_route_cost = float(
                route["monthly_cost"]
            )

            truck_capacity = float(
                route["truck_capacity_cft"]
            )

            print(
                f"\nProcessing Route: {route_id}"
            )

            # =========================================
            # STORE LIST
            # =========================================

            stores_string = str(
                route["stores_served"]
            )

            stores = [

                s.strip()

                for s in stores_string.split("->")

                if s.strip() != ""
            ]

            print(
                f"Stores in Route: {len(stores)}"
            )

            # =========================================
            # PROCESS EACH STORE
            # =========================================

            for store in stores:

                store_row = stores_df[

                    stores_df["store"]
                    == store

                ]

                # =====================================
                # STORE NOT FOUND
                # =====================================

                if len(store_row) == 0:

                    print(
                        f"Store Missing: {store}"
                    )

                    continue

                # =====================================
                # STORE DEMAND
                # =====================================

                store_demand = float(

                    store_row.iloc[0][
                        "demand_cft"
                    ]
                )

                # =====================================
                # COST ALLOCATION
                # =====================================

                contribution_percent = (

                    store_demand
                    /
                    truck_capacity

                )

                allocated_cost = (

                    contribution_percent
                    *
                    total_route_cost

                )

                # =====================================
                # SAVE OUTPUT
                # =====================================

                store_cost_rows.append({

                    "dc":
                        dc,

                    "route_id":
                        route_id,

                    "store":
                        store,

                    "store_demand_cft":
                        round(
                            store_demand,
                            2
                        ),

                    "truck_capacity_cft":
                        round(
                            truck_capacity,
                            2
                        ),

                    "route_total_demand_cft":
                        round(
                            total_route_demand,
                            2
                        ),

                    "store_contribution_percent":
                        round(
                            contribution_percent
                            * 100,
                            2
                        ),

                    "route_monthly_cost":
                        round(
                            total_route_cost,
                            2
                        ),

                    "allocated_monthly_logistics_cost":
                        round(
                            allocated_cost,
                            2
                        )
                })

        except Exception as e:

            print(
                f"\nERROR IN ROUTE:"
            )

            print(e)

# =====================================================
# CREATE OUTPUT DATAFRAME
# =====================================================

final_df = pd.DataFrame(
    store_cost_rows
)

# =====================================================
# SORT OUTPUT
# =====================================================

if len(final_df) > 0:

    final_df = final_df.sort_values(

        by=[

            "dc",
            "route_id"
        ]
    )

# =====================================================
# SAVE OUTPUT
# =====================================================

final_df.to_excel(

    "store_wise_monthly_costs.xlsx",

    index=False
)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n================================")
print("STORE COST ALLOCATION COMPLETE")
print("================================")

print(
    "\nTotal Stores Allocated:",
    len(final_df)
)

print(
    "\nGenerated File:"
)

print(
    "store_wise_monthly_costs.xlsx"
)
