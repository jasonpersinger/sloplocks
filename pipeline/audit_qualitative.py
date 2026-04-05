import json
import csv
import os
from datetime import datetime
from collections import defaultdict

QUALITATIVE_LOG = "data/tracking/qualitative_log.jsonl"
RESULTS_LOG = "data/tracking/results_log.csv"

def audit_gemini():
    if not os.path.exists(QUALITATIVE_LOG):
        print("No qualitative log found.")
        return

    # 1. Load Gemini Thoughts
    gemini_thoughts = []
    with open(QUALITATIVE_LOG, "r") as f:
        for line in f:
            gemini_thoughts.append(json.loads(line))

    # 2. Load Results
    results = {}
    if os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Key: TeamA_vs_TeamB_Date
                key = f"{row['home_team']}_vs_{row['away_team']}_{row['match_date']}"
                results[key] = row

    # 3. Analyze
    stats = {
        "total": 0,
        "resolved": 0,
        "agreed_with_model": 0,
        "disagreed_with_model": 0,
        "correct_when_disagreed": 0,
        "incorrect_when_disagreed": 0,
        "avg_impact": 0.0
    }
    
    sport_stats = defaultdict(lambda: {"total": 0, "resolved": 0, "correct": 0})

    for thought in gemini_thoughts:
        stats["total"] += 1
        game_id = thought.get("game_id")
        sport = thought.get("sport", "unknown")
        sport_stats[sport]["total"] += 1
        
        result = results.get(game_id)
        if result and result.get("won"):
            stats["resolved"] += 1
            sport_stats[sport]["resolved"] += 1
            
            # Simplified Logic for MVP analysis:
            # Did Gemini's net edge match the actual winner?
            raw_res = thought.get("raw_response", "")
            try:
                parsed = json.loads(raw_res)
                edge = parsed.get("net_qualitative_edge", "none")
                actual_winner = result.get("won") # 'home' or 'away'
                
                if edge != "none":
                    if edge == actual_winner:
                        sport_stats[sport]["correct"] += 1
            except:
                pass

    print("--- Gemini Qualitative Audit ---")
    print(f"Total Thoughts: {stats['total']}")
    print(f"Resolved Games: {stats['resolved']}")
    print("\nPer Sport Accuracy (Directional Edge):")
    for sport, s in sport_stats.items():
        acc = (s["correct"] / s["resolved"] * 100) if s["resolved"] > 0 else 0
        print(f"  {sport.upper()}: {s['correct']}/{s['resolved']} ({acc:.1f}%)")
    
    print("\nNote: This is a directional audit. Full Brier Score impact analysis")
    print("will be available once more data accumulates in the results_log.")

if __name__ == "__main__":
    audit_gemini()
