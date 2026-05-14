import pandas as pd

# =====================================================
# LOAD ORIGINAL STORE DATA
# =====================================================

stores_df = pd.read_excel("clustered_output.xlsx")

# =====================================================
# ALL ROUTE FILES
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
# FINAL OUTPUT LIST
# =====================================================

store_cost_rows = []

# =====================================================
# PROCESS EACH ROUTE FILE
# =====================================================

for file in route_files:

    try:

        routes_df = pd.read_excel(file)

    except:

        print(f"\nSkipping missing file: {file}")
        continue

    # =================================================
    # PROCESS EACH ROUTE
    # =================================================

    for _, route in routes_df.iterrows():

        route_id = route["route_id"]

        dc = route["dc"]

        total_route_demand = float(
            route["total_demand_cft"]
        )

        total_route_cost = float(
            route["monthly_cost"]
        )

        stores_string = route["stores_served"]

        stores = [

            s.strip()

            for s in stores_string.split("->")

            if s.strip() != ""
        ]

        # =============================================
        # PROCESS EACH STORE
        # =============================================

        for store in stores:

            store_row = stores_df[
                stores_df["store"] == store
            ]

            if len(store_row) == 0:
                continue

            store_demand = float(
                store_row.iloc[0]["demand_cft"]
            )

            # =========================================
            # COST SHARE
            # =========================================

            contribution_percent = (

                store_demand
                /
                total_route_demand

            )

            allocated_cost = (

                contribution_percent
                *
                total_route_cost

            )

            # =========================================
            # SAVE OUTPUT
            # =========================================

            store_cost_rows.append({

                "store":
                    store,

                "dc":
                    dc,

                "route_id":
                    route_id,

                "store_demand_cft":
                    round(store_demand, 2),

                "route_total_demand_cft":
                    round(total_route_demand, 2),

                "store_contribution_percent":
                    round(
                        contribution_percent * 100,
                        2
                    ),

                "route_monthly_cost":
                    round(total_route_cost, 2),

                "allocated_monthly_cost":
                    round(allocated_cost, 2)
            })

# =====================================================
# CREATE OUTPUT DATAFRAME
# =====================================================

final_df = pd.DataFrame(store_cost_rows)

# =====================================================
# SAVE OUTPUT
# =====================================================

final_df.to_excel(

    "store_wise_monthly_costs.xlsx",

    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n=================================")
print("STORE COST ALLOCATION COMPLETED")
print("=================================")

print(
    "\nTotal Stores Processed:",
    len(final_df)
)

print(
    "\nGenerated File:"
)

print(
    "store_wise_monthly_costs.xlsx"
)
