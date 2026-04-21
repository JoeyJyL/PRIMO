import gurobipy as gp
from gurobipy import GRB
import math

def haversine_miles(lat1, lon1, lat2, lon2):
    """Compute Haversine distance in miles between two (lat, lon) points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def solve_air_taxi_location():
    # --- 1. Data Input ---

    # Candidate station sites: id -> (name, lat, lon, fixed_cost_K$)
    stations = {
        'S1':  ('JFK Airport',         40.6413, -73.7781, 500),
        'S2':  ('LaGuardia Airport',   40.7769, -73.8740, 450),
        'S3':  ('Central Park South',  40.7648, -73.9730, 600),
        'S4':  ('World Trade Center',  40.7127, -74.0134, 550),
        'S5':  ('Times Square',        40.7580, -73.9855, 580),
        'S6':  ('Brooklyn Heights',    40.6960, -73.9936, 350),
        'S7':  ('Yankee Stadium',      40.8296, -73.9262, 300),
        'S8':  ('Newark Airport',      40.6895, -74.1745, 480),
        'S9':  ('Washington Square',   40.7308, -73.9973, 400),
        'S10': ('East Harlem',         40.7957, -73.9389, 320),
    }

    # Trip demand: id -> (pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    trips = {
        'T1':  (40.7654, -73.9760, 40.6420, -73.7790),
        'T2':  (40.7590, -73.9840, 40.7775, -73.8730),
        'T3':  (40.7130, -74.0120, 40.6410, -73.7785),
        'T4':  (40.7600, -73.9870, 40.7120, -74.0140),
        'T5':  (40.7650, -73.9720, 40.8300, -73.9270),
        'T6':  (40.6950, -73.9950, 40.6900, -74.1750),
        'T7':  (40.7300, -73.9980, 40.7770, -73.8750),
        'T8':  (40.7960, -73.9400, 40.6415, -73.7780),
        'T9':  (40.7580, -73.9850, 40.6960, -73.9940),
        'T10': (40.8290, -73.9260, 40.7650, -73.9740),
        'T11': (40.7650, -73.9735, 40.7310, -73.9970),
        'T12': (40.6420, -73.7785, 40.7590, -73.9860),
        'T13': (40.7770, -73.8745, 40.7130, -74.0130),
        'T14': (40.7125, -74.0140, 40.8295, -73.9265),
        'T15': (40.6895, -74.1740, 40.7650, -73.9725),
        'T16': (40.7955, -73.9385, 40.7585, -73.9845),
        'T17': (40.7310, -73.9975, 40.6950, -73.9940),
        'T18': (40.7650, -73.9750, 40.6895, -74.1748),
        'T19': (40.6415, -73.7788, 40.7960, -73.9395),
        'T20': (40.7580, -73.9855, 40.7305, -73.9978),
    }

    # Parameters
    R = 1.0     # Max on-road travel distance (miles)
    CS = 0.70   # Minimum demand fulfillment rate (70%)

    station_ids = list(stations.keys())
    trip_ids = list(trips.keys())
    min_covered = math.ceil(CS * len(trip_ids))  # At least 14 trips

    # --- 2. Pre-processing: Compute coverage matrices ---
    P = {}
    Q = {}
    for e in trip_ids:
        p_lat, p_lon, d_lat, d_lon = trips[e]
        P[e] = {}
        Q[e] = {}
        for s in station_ids:
            s_name, s_lat, s_lon, s_cost = stations[s]
            dist_pickup = haversine_miles(p_lat, p_lon, s_lat, s_lon)
            dist_dropoff = haversine_miles(d_lat, d_lon, s_lat, s_lon)
            P[e][s] = 1 if dist_pickup <= R else 0
            Q[e][s] = 1 if dist_dropoff <= R else 0

    # Print coverage matrix summary
    print("--- Coverage Matrix Summary ---")
    for e in trip_ids:
        p_stations = [s for s in station_ids if P[e][s] == 1]
        d_stations = [s for s in station_ids if Q[e][s] == 1]
        print(f"  {e}: pickup covered by {p_stations}, dropoff covered by {d_stations}")

    # --- 3. Optimization Model ---
    model = gp.Model("AirTaxi_Station_Location")

    # Decision variables
    y = model.addVars(station_ids, vtype=GRB.BINARY, name="open")
    z = model.addVars(trip_ids, vtype=GRB.BINARY, name="covered")

    # --- 4. Objective: Minimize total fixed cost ---
    model.setObjective(
        gp.quicksum(stations[s][3] * y[s] for s in station_ids),
        GRB.MINIMIZE
    )

    # --- 5. Constraints ---

    # C1: Pickup coverage
    for e in trip_ids:
        model.addConstr(
            gp.quicksum(P[e][s] * y[s] for s in station_ids) >= z[e],
            name=f"pickup_cov_{e}"
        )

    # C2: Dropoff coverage
    for e in trip_ids:
        model.addConstr(
            gp.quicksum(Q[e][s] * y[s] for s in station_ids) >= z[e],
            name=f"dropoff_cov_{e}"
        )

    # C3: Minimum demand fulfillment rate
    model.addConstr(
        gp.quicksum(z[e] for e in trip_ids) >= min_covered,
        name="min_coverage"
    )

    # C4: Mandatory station - S4 (World Trade Center) must be opened
    model.addConstr(
        y['S4'] == 1,
        name="mandatory_S4"
    )

    # --- 6. Solve ---
    model.optimize()

    # --- 7. Output ---
    if model.status == GRB.OPTIMAL:
        result = {
            "status": "optimal",
            "obj": round(model.ObjVal, 2),
        }

        print(f"\nOptimal Objective (Total Cost $K): {model.ObjVal:.0f}")
        print(f"Minimum trips to cover: {min_covered} / {len(trip_ids)}")

        open_stations = []
        total_cost = 0
        print("\n--- Opened Stations ---")
        for s in station_ids:
            if y[s].X > 0.5:
                name, lat, lon, cost = stations[s]
                open_stations.append(s)
                total_cost += cost
                print(f"  {s}: {name:<25} (${cost}K) at ({lat}, {lon})")

        covered_trips = [e for e in trip_ids if z[e].X > 0.5]
        uncovered_trips = [e for e in trip_ids if z[e].X < 0.5]
        print(f"\nCovered trips ({len(covered_trips)}/{len(trip_ids)}): {covered_trips}")
        print(f"Uncovered trips ({len(uncovered_trips)}): {uncovered_trips}")
        print(f"Coverage rate: {len(covered_trips)/len(trip_ids)*100:.1f}%")
        print(f"Stations opened: {len(open_stations)}")
        print(f"Total cost: ${total_cost}K")

        return result
    else:
        return {"status": "infeasible"}


if __name__ == "__main__":
    result = solve_air_taxi_location()
    print(f"\nResult: {result}")
