"""
F1 2026 GP Prediction System Shared Config
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

# DRIVER → TEAM MAPPING  (2026 season)
DRIVER_TEAMS_2026 = {
    "ANT": "Mercedes",     "RUS": "Mercedes",
    "LEC": "Ferrari",      "HAM": "Ferrari",
    "NOR": "McLaren",      "PIA": "McLaren",
    "VER": "Red Bull",     "HAD": "Red Bull",
    "GAS": "Alpine",       "COL": "Alpine",
    "LAW": "Racing Bulls", "LIN": "Racing Bulls",
    "BOR": "Audi",         "HUL": "Audi",
    "OCO": "Haas",         "BEA": "Haas",
    "ALB": "Williams",     "SAI": "Williams",
    "BOT": "Cadillac",     "PER": "Cadillac",
    "ALO": "Aston Martin", "STR": "Aston Martin",
}
ROOKIES_2026 = {"LIN", "HAD"} # ROOKIE FLAG

# TEAM COLORS  (official 2026 livery)
TEAM_COLORS = {
    "Mercedes":    "#27F4D2",
    "Ferrari":     "#E8002D",
    "McLaren":     "#FF8000",
    "Red Bull":    "#3671C6",
    "Alpine":      "#FF87BC",
    "Racing Bulls":"#6692FF",
    "Audi":        "#2F9B48",
    "Haas":        "#B6BABD",
    "Williams":    "#64C4FF",
    "Cadillac":    "#FFFFFF",
    "Aston Martin":"#229971",
}

DNF = "DNF"
DNS = "DNS"

####### CONFIRMED 2026 RACE RESULTS ########
# FORMAT: { "DRIVER" : FINISHING POSITION}
# DNF = 20, DNS = 22

# ROUND 1 — ALBERT PARK CIRCUIT, MELBOURNE, AUSTRALIA (6 - 8 MARCH)
R01_AUS = {
    "RUS": 1, "ANT": 2, "LEC": 3, "HAM": 4,
    "NOR": 5, "VER": 6, "BEA": 7, "LIN": 8,
    "BOR": 9, "GAS": 10, "OCO": 11, "ALB": 12,
    "LAW": 13, "COL": 14, "SAI": 15, "PER": 16,
    "STR": DNF, "ALO": DNF, "BOT": DNF, "HAD": DNF, 
    "PIA": DNS, "HUL": DNS,
}

# ROUND 2 — SHANGHAI INTERNATIONAL CIRCUIT, SHANGHAI, CHINA (13 - 15 MARCH)
R02_CHN = {
    "ANT": 1, "RUS": 2, "HAM": 3, "LEC": 4,
    "BEA": 5, "GAS": 6, "LAW": 7, "HAD": 8,
    "SAI": 9, "COL": 10, "HUL": 11, "LIN": 12,
    "BOT": 13, "OCO": 14, "PER": 15, "VER": DNF, 
    "ALO": DNF, "STR": DNF, "PIA": DNS, "NOR": DNS,
    "BOR": DNS, "ALB": DNS,
}

# ROUND 3 — SUZUKA CIRCUIT, SUZUKA, JAPAN (27 - 29 MARCH)
R03_JPN = {
    "ANT": 1, "PIA": 2, "LEC": 3, "RUS": 4,
    "NOR": 5, "HAM": 6, "GAS": 7, "VER":8,
    "LAW": 9, "OCO": 10, "HUL": 11, "HAD": 12, 
    "BOR": 13, "LIN": 14, "SAI": 14, "COL": 16,
    "PER": 17, "ALO": 18, "BOT": 19, "ALB": 20,
    "STR": 20, "BEA": 20,
}

# ROUND 4 — MIAMI INTERNATIONAL AUTODROME, MIAMI, USA (1 - 3 MAY)
R04_MIA = {
    "ANT": 1, "NOR": 2, "PIA": 3, "RUS": 4,
    "VER": 5, "HAM": 6, "COL": 7, "LEC": 8, 
    "SAI": 9, "ALB": 10, "BEA": 11, "BOR": 12, 
    "OCO": 13, "LIN": 14, "ALO": 15, "PER": 16,
    "STR": 17, "BOT": 18, "HUL": 20, "LAW": 20, 
    "GAS": DNF, "HAD": DNF, 
}


# ROUND 5 - CIRCUIT  GILLES-VILLENEUVE, MONTREAL, CANADA (22 - 24 MAY)
R05_CAN = {
    "ANT": 1, "HAM": 2, "VER": 3, "LEC": 4,
    "HAD": 5, "COL": 6, "LAW": 7, "GAS": 8,
    "SAI": 9, "BEA": 10, "PIA": 11, "HUL": 12,
    "BOR": 13, "OCO": 14, "STR": 15, "BOT": 16,
    "PER": DNF, "NOR": DNF, "RUS": DNF, "ALO": DNF,
    "ALB": DNF, "LIN": DNS,
}

# SEASON SUMMARY
ALL_RACES = [
    ("R01", "Australia", R01_AUS),
    ("R02", "China", R02_CHN),
    ("R03", "Japan", R03_JPN),
    ("R04", "Miami", R04_MIA),
    ("R05", "Canada", R05_CAN),
]

def position_to_number(pos, dnf_value=98, dns_value=99):
    if pos == DNS:
        return float(dns_value)
    if pos == DNF:
        return float(dnf_value)
    return float(pos)

def compute_season_form(races_available, all_drivers, 
                        dnf_value=98, dns_value=99,
                        low_weight=0.8, high_weight=1.5):
    """Compute each driver's weighted average finishing position.

    Lower form score = better recent performance.

    Races are weighted on a linear ramp from 0.8 (oldest) to 1.5
    (most recent). If a driver has no entry in a race result they
    receive a neutral score of 11.0 for that race (midfield).
    """
    if not races_available:
        return {drv: 11.0 for drv in all_drivers}

    n = len(races_available)
    weights = np.linspace(low_weight, high_weight, n)  # Linear ramp: oldest=0.8, most recent=1.5

    form = {}
    for drv in all_drivers:
        positions = []
        race_weights = []
        for i, (_code, _name, results) in enumerate(races_available):
            raw = results.get(drv, 11.0)  # absent -> neutral midfield (11.0)
            positions.append(position_to_number(raw, dnf_value, dns_value))
            race_weights.append(weights[i])
        form[drv] = float(np.average(positions, weights=race_weights))

    return form




#### data features
def calculate_team_tiers(races_so_far, driver_teams):
    """
    1. loop over every completed race result in races_so_far
    2. for each driver's finishing position, look up how many points that position is worth
    3. add those points to the driver's team running total
    4. scale the result so best team = 1.0 and worst team = 5.0

    FIRST RACE: every team gets a neutral tier of 3.0, treating all teams as equal until real data exists

    POINT SYSTEM: 
    F1_POINTS = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6:  8, 7:  6, 8:  4, 9:  2, 10:  1,
    }

    P11 - 22: 0 points

    small penalties for DNF/DNS:
    The gap between each classified position near the bottom is 1 point (P9=2, P10=1, P11=0). We follow that same step downward:
        DNF = -1  (one step below drivers that finished but did not score)
        DNS = -2  (two steps for cars that did not make the start at all)
    small note: there are categories to a DNF incl. mechanical failures, and racing accident that are not true indicators of the car performance
                however, for now, we will take a more conservative approach and treat all DNFs the same. 

    Args:
        races_so_far : list of (code, name, results_dict) tuples
                       e.g., [("R01", "Australia", R01_AUS), ...]
        driver_teams : dict mapping driver abbre. --> team name
                       e.g., {"ANT": "Mercedes", "SAI" : "Williams}, ...}

    Returns:
        dict {team_name: tier_score} where tier_score is a float in [1.0, 5.0]}
    """
    all_teams = set(driver_teams.values())

    # FIRST RACE
    if not races_so_far:
        all_teams = set(driver_teams.values())
        return {team: 3.0 for team in all_teams}

    F1_POINTS = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
    }
    DNF_PENALTY = -1
    DNS_PENALTY = -2

    team_points = {team: 0 for team in all_teams}

    for _code, _name, results in races_so_far:
        for driver, pos in results.items():
            team = driver_teams.get(driver)
            if team is None: 
                continue

            if pos == DNS:
                value = DNS_PENALTY
            elif pos == DNF:
                value = DNF_PENALTY
            else:
                value = F1_POINTS.get(pos, 0)  # P11 and below get 0 points
            
            team_points[team] += value
    
    sorted_teams = sorted(
        team_points.items(), key=lambda x: x[1], reverse=True
    )

    max_pts   = sorted_teams[0][1]
    min_pts   = sorted_teams[-1][1]
    pts_range = max_pts - min_pts
 
    if pts_range == 0:
        return {team: 3.0 for team in all_teams}
 
    tiers = {}
    for team, pts in sorted_teams:
        tiers[team] = round(
            1.0 + 4.0 * (1 - (pts - min_pts) / pts_range), 3
        )
 
    return tiers

def calculate_quali_gap(races_completed, driver_teams, fastf1_module,
                        fallback_tier=3.0):
    """
    single lap pace from quali data across all 2026 races so far
    - calulate seconds behind pole each driver was,
    - group those gaps by team,
    - take median across both drivers and all races
    then scale:
    - smallest median gap : 1.0 (closest to pole)
    - largest median gap : 5.0 (furthest from pole)

    fallback_tier = 3.0 (neutral) for no data 
    """

    all_teams = set(driver_teams.values())

    if not races_completed:
        return {team: fallback_tier for team in all_teams}
    
    team_gaps = {team: [] for team in all_teams}

    for _code, circuit_name, _results in races_completed:
        try: 
            q_sess = fastf1_module.get_session(2026, circuit_name, "Q")
            q_sess.load(laps=True, telemetry=False,
                        weather=False, messages=False)
            
            best_laps = (
                q_sess.laps
                .pick_quicklaps()
                .groupby("Driver")["LapTime"]
                .min()
            )

            if best_laps.empty:
                print(f" [quali gap] No lap data for {circuit_name} - skip")
                continue
            
            pole_time_s = best_laps.min().total_seconds()

            for driver, lap_time in best_laps.items(): 
                team = driver_teams.get(driver)
                if team is None: 
                    continue
                team_gaps[team].append(lap_time.total_seconds() - pole_time_s)
            
            print(f" [quali_gap] {circuit_name}: pole {pole_time_s:.3f}s"
                  f"| {len(best_laps)} drivers loaded")


        except Exception as e:
            print(f" [quali_gap] {circuit_name} failed: {e} - skip")
            continue

    team_median_gap = {
        team: (float(np.median(gaps)) if gaps else None)
        for team, gaps in team_gaps.items()
    }

    teams_with_data = {t: g for t, g in team_median_gap.items()
                       if g is not None}
    teams_without = {t for t, g in team_median_gap.items()
                     if g is None}
    
    if not teams_with_data:
        print(f" [quali_gap] WARNING: no data loaded for any session. "
              f"All teams assigned fallback {fallback_tier}.")
        return {team: fallback_tier for team in all_teams}
    
    min_gap = min(teams_with_data.values())
    max_gap = max(teams_with_data.values())
    gap_range = max_gap - min_gap

    result = {}
    for team, median_gap in teams_with_data.items():
        if gap_range == 0:
            result[team] = 1.0
        else:
            result[team] = round(
                1.0 + 4.0 * (median_gap - min_gap) / gap_range, 3
            )
        
    for team in teams_without:
        result[team] = fallback_tier

    return result   


def tune_form_weights(historical_df, 
                      features, target, races_for_tuning,
                      all_drivers, low_candidates=None,
                      high_candidates=None, n_folds=5, 
                      random_state=42, verbose=True,):
    if low_candidates is None:
        low_candidates = [1.0]
    if high_candidates is None:
        high_candidates = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    
    # ── Validate inputs ───────────────────────────────────────────────
    if "SeasonForm" not in features:
        raise ValueError(
            "'SeasonForm' must be in features list — that is the "
            "column this function tunes the weights for."
        )
    if target not in historical_df.columns:
        raise ValueError(f"Target column '{target}' not in historical_df.")

    df = historical_df.dropna(subset=features + [target]).copy()
    n  = len(df)

    if n < n_folds:
        raise ValueError(
            f"historical_df has only {n} rows but n_folds={n_folds}. "
            f"Reduce n_folds or add more historical data."
        )

    # ── Build fold indices ────────────────────────────────────────────
    # Assign each row to a fold (0, 1, 2, ..., n_folds-1)
    rng        = np.random.default_rng(random_state)
    fold_ids   = np.tile(np.arange(n_folds), n // n_folds + 1)[:n]
    rng.shuffle(fold_ids)

    n_races = len(races_for_tuning)

    all_results = []
    total_valid = sum(
        1 for low in low_candidates
          for high in high_candidates
          if low < high
    )
    if verbose:
        print(f"Grid search: {len(low_candidates)} low × "
              f"{len(high_candidates)} high = "
              f"{total_valid} valid combinations | "
              f"{n_folds}-fold CV")
        print(f"{'Low':>6}  {'High':>6}  {'Mean MAE':>10}  {'Std MAE':>10}")
        print("─" * 40)

    # ── Main grid search loop ─────────────────────────────────────────
    for low in low_candidates:
        for high in high_candidates:

            # Skip invalid combinations where low >= high
            # (oldest race counting more than newest makes no sense)
            if low >= high:
                continue

            # Compute SeasonForm for every driver using these weights
            weights      = np.linspace(low, high, n_races) if n_races > 1 else np.array([1.0])
            form_scores  = {}
            for drv in all_drivers:
                positions     = []
                race_weights  = []
                for i, (_c, _n, results) in enumerate(races_for_tuning):
                    raw = results.get(drv, 15.0)
                    positions.append(position_to_number(raw))
                    race_weights.append(weights[i])
                form_scores[drv] = float(np.average(positions, weights=race_weights))

            # Overwrite SeasonForm column with these candidate weights
            df["SeasonForm"] = df["Driver"].map(form_scores).fillna(11.0)

            # ── Cross-validation ──────────────────────────────────────
            fold_maes = []
            for fold in range(n_folds):
                train_mask = fold_ids != fold
                test_mask  = fold_ids == fold

                X_train = df.loc[train_mask, features].values
                y_train = df.loc[train_mask, target].values
                X_test  = df.loc[test_mask,  features].values
                y_test  = df.loc[test_mask,  target].values

                scaler  = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s  = scaler.transform(X_test)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        min_samples_split=4,
                        random_state=random_state,
                    )
                    model.fit(X_train_s, y_train)

                preds = model.predict(X_test_s)
                fold_maes.append(mean_absolute_error(y_test, preds))

            mean_mae = float(np.mean(fold_maes))
            std_mae  = float(np.std(fold_maes))
            all_results.append((low, high, mean_mae, std_mae))

            if verbose:
                print(f"{low:>6.2f}  {high:>6.2f}  "
                      f"{mean_mae:>10.4f}  {std_mae:>10.4f}")

    # ── Sort and return ───────────────────────────────────────────────
    all_results.sort(key=lambda x: x[2])   # sort by mean MAE ascending
    best_low, best_high, best_mae, best_std = all_results[0]
    worst_mae  = all_results[-1][2]
    mae_spread = worst_mae - best_mae

    if verbose:
        print()
        print(f"✅ Best ratio: {best_high:.2f}x  "
              f"(most recent race counts {best_high:.2f}× the oldest)")
        print(f"   Cross-validated MAE: {best_mae:.4f} ± {best_std:.4f}")
        print()

        # Spread interpretation — this is how you justify the value
        print(f"📐 MAE spread across all ratios tested: {mae_spread:.4f} positions")
        if mae_spread < 0.1:
            print("   → Spread is tiny. The ratio barely affects predictions.")
            print("   → Justification: use ratio=1.0 (equal weights) as the")
            print("     simplest defensible choice. Form weighting does not")
            print("     meaningfully improve the model for this circuit.")
        elif mae_spread < 0.3:
            print("   → Moderate spread. The winning ratio is a reasonable choice")
            print("     but alternatives are not far behind.")
            print(f"   → Justification: ratio={best_high:.2f} reduced MAE by")
            print(f"     {mae_spread:.3f} positions vs the worst alternative.")
        else:
            print("   → Strong spread. The winning ratio is clearly better.")
            print(f"   → Justification: ratio={best_high:.2f} reduced MAE by")
            print(f"     {mae_spread:.3f} positions — a meaningful improvement")
            print("     that is well justified by the cross-validation data.")
        print()
        print("Full results (sorted best → worst):")
        print(f"  {'Ratio':>7}  {'Mean MAE':>10}  {'Std MAE':>10}")
        for row in all_results:
            marker = "  ← best" if row == all_results[0] else ""
            print(f"  {row[1]:>7.2f}  {row[2]:>10.4f}  {row[3]:>10.4f}{marker}")

    return {
        "best_low":    best_low,
        "best_high":   best_high,
        "best_mae":    best_mae,
        "mae_spread":  mae_spread,
        "all_results": all_results,
    }
