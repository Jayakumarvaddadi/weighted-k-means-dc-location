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
