"""
Scenario test for the Urban Heat Resilience Agent.
Tests the agent on the primary challenge question:
'We are the City of Dallas. We have budget for 3 emergency cooling centers this summer. Where should they go and why?'
"""

import asyncio
from agent.tools import (
    recommend_candidate_facilities,
    get_historical_heat_frequency,
    perform_equity_audit,
    predict_heat_offset,
)

def run_test():
    print("=" * 70)
    print("SCENARIO TEST: City of Dallas 3 Cooling Centers Allocation")
    print("=" * 70)
    
    # 1. Historical Context Query
    print("\n[Step 1: Historical Heat Exposure Context]")
    hist = get_historical_heat_frequency(100)
    print(f"  7-Year Station Exceedances (>=100°F): {hist['total_station_days_exceeded']} days")
    print(f"  Annual Average Days >= 100°F: {hist['annual_average_station_days']} days/year")
    print(f"  All-time record high in station data: {hist['all_time_record_high_f']}°F")

    # 2. Candidate Siting Query via BQML & OSM
    print("\n[Step 2: Model-Predicted Candidate Siting (Top 3)]")
    candidates = recommend_candidate_facilities(facility_type="all", top_n=3)
    selected_osm_ids = []
    for idx, c in enumerate(candidates, 1):
        selected_osm_ids.append(c["osm_id"])
        print(f"  #{idx} {c['facility_name']} ({c['facility_type']})")
        print(f"      Coordinates: {c['coordinates']['lat']:.4f}, {c['coordinates']['lon']:.4f}")
        print(f"      Predicted Heat Offset: {c['predicted_temp_offset_f']:+.2f}°F")
        print(f"      Neighborhood Poverty: {c['poverty_pct']}% | Zero-Vehicle Households: {c['no_vehicle_pct']}%")
        print(f"      Distance to Nearest Park: {c['dist_to_park_m']}m")

    # 3. Microclimate Driver Explanation
    print("\n[Step 3: ML.EXPLAIN_PREDICT Attributions on Top Site]")
    top_site = candidates[0]
    explanation = predict_heat_offset(
        latitude=top_site["coordinates"]["lat"],
        longitude=top_site["coordinates"]["lon"],
        dist_to_park_m=top_site["dist_to_park_m"],
        poverty_rate=top_site["poverty_pct"] / 100.0,
        no_vehicle_rate=top_site["no_vehicle_pct"] / 100.0,
    )
    print(f"  Site: {top_site['facility_name']}")
    print(f"  Interpretation: {explanation['interpretation']}")
    print("  Top Drivers:")
    for d in explanation["top_drivers"]:
        print(f"    - {d['feature']}: {d['attribution']:+.3f}°F attribution")

    # 4. Equity Audit
    print("\n[Step 4: Equity Audit of Recommended Output]")
    audit = perform_equity_audit(selected_osm_ids)
    print(f"  Selected Sites Avg Poverty: {audit['selected_poverty_pct']}% vs Citywide Avg: {audit['citywide_poverty_pct']}%")
    print(f"  Selected Sites Avg Zero-Vehicle: {audit['selected_no_vehicle_pct']}% vs Citywide Avg: {audit['citywide_no_vehicle_pct']}%")
    print(f"  Equity Goal Satisfied: {'YES (Pass)' if audit['equity_reached'] else 'NO (Fail)'}")
    print("=" * 70)

if __name__ == "__main__":
    run_test()
