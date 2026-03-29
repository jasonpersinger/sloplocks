"""Dixon-Coles prediction model, Elo rating system, Adjusted Efficiency,
and Four Factors logistic regression for SLOP LOCKS."""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, poisson
from sklearn.linear_model import LinearRegression, LogisticRegression

from pipeline.config import (
    CONGESTION_PENALTY,
    ELO_HOME_ADVANTAGE,
    ELO_K_FACTOR,
    FORM_WEIGHT_MULTIPLIER,
    FORM_WINDOW,
    MAX_GOALS,
    MLB_PARK_FACTORS,
    TIME_DECAY_RATE,
)


# ---------------------------------------------------------------------------
# Dixon-Coles helpers
# ---------------------------------------------------------------------------

def _tau(x, y, lambda_h, lambda_a, rho):
    """Low-score correction factor (tau) for the Dixon-Coles model.

    Adjusts the independent Poisson probabilities for scorelines 0-0, 1-0,
    0-1 and 1-1 so that the model can capture the empirical correlation
    between low home and away goal counts.

    Parameters
    ----------
    x, y : int
        Home goals and away goals for the scoreline.
    lambda_h, lambda_a : float
        Expected goals for home and away teams.
    rho : float
        Correlation parameter (typically slightly negative).

    Returns
    -------
    float
        Multiplicative adjustment to the Poisson probability.
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_h * lambda_a * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_h * rho
    elif x == 1 and y == 0:
        return 1.0 + lambda_a * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def _compute_weights(matches):
    """Compute exponential time-decay weights for each match.

    More recent matches receive higher weight.  The last ``FORM_WINDOW``
    matches also receive an additional ``FORM_WEIGHT_MULTIPLIER`` boost.

    Parameters
    ----------
    matches : pd.DataFrame
        Must contain a ``date`` column (ISO-format string or datetime).

    Returns
    -------
    np.ndarray
        Weight vector aligned with the rows of *matches*.
    """
    dates = pd.to_datetime(matches["date"])
    most_recent = dates.max()
    days_ago = (most_recent - dates).dt.days.values.astype(float)

    weights = np.exp(-TIME_DECAY_RATE * days_ago)

    # Boost the most recent FORM_WINDOW matches
    n = len(matches)
    if n > 0:
        # Sort indices by date descending, take last FORM_WINDOW
        sorted_idx = np.argsort(-days_ago)  # smallest days_ago first
        boost_idx = sorted_idx[:min(FORM_WINDOW, n)]
        weights[boost_idx] *= FORM_WEIGHT_MULTIPLIER

    return weights


def _dc_log_likelihood(params, matches, teams, weights):
    """Negative log-likelihood for the Dixon-Coles model.

    Parameters are packed as::

        [attack_0 .. attack_{n-1}, defense_0 .. defense_{n-1}, home_adv, rho]

    where *n* = ``len(teams)``.

    Parameters
    ----------
    params : array-like
        Packed parameter vector.
    matches : pd.DataFrame
        Columns: home_team, away_team, home_goals, away_goals.
    teams : list[str]
        Ordered list of team names matching param indices.
    weights : np.ndarray
        Per-match weights from :func:`_compute_weights`.

    Returns
    -------
    float
        Negative log-likelihood (to be minimized).
    """
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    attack = params[:n_teams]
    defense = params[n_teams: 2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    log_lik = 0.0

    for i, row in matches.iterrows():
        hi = team_idx.get(row["home_team"])
        ai = team_idx.get(row["away_team"])
        if hi is None or ai is None:
            continue

        hg = int(row["home_goals"])
        ag = int(row["away_goals"])

        # Expected goals
        lambda_h = max(1e-6, math.exp(attack[hi] + defense[ai] + home_adv))
        lambda_a = max(1e-6, math.exp(attack[ai] + defense[hi]))

        # Poisson probabilities
        p_h = poisson.pmf(hg, lambda_h)
        p_a = poisson.pmf(ag, lambda_a)

        # Low-score correction
        tau_val = _tau(hg, ag, lambda_h, lambda_a, rho)

        lik = max(1e-20, p_h * p_a * tau_val)
        log_lik += weights[matches.index.get_loc(i)] * math.log(lik)

    return -log_lik


def fit_dixon_coles(matches, goals_col_home="home_goals",
                    goals_col_away="away_goals"):
    """Fit the Dixon-Coles model to historical match data.

    Uses ``scipy.optimize.minimize`` with the SLSQP method.  A linear
    constraint ensures that the sum of attack parameters is zero
    (identifiability).

    Parameters
    ----------
    matches : pd.DataFrame
        Must have columns: home_team, away_team, and the two goals columns.
    goals_col_home, goals_col_away : str
        Column names for home/away goals (allows reuse with xG data).

    Returns
    -------
    dict
        Keys: ``attack`` (team -> float), ``defense`` (team -> float),
        ``home_advantage`` (float), ``rho`` (float).
    """
    # Rename goals columns to canonical names for internal use
    df = matches.rename(columns={
        goals_col_home: "home_goals",
        goals_col_away: "away_goals",
    }).copy()

    # Only keep teams that actually appear in the data
    teams_in_data = sorted(
        set(df["home_team"].unique()) | set(df["away_team"].unique())
    )
    n = len(teams_in_data)

    weights = _compute_weights(df)

    # Initial values: attack/defense = 0, home_adv = 0.25, rho = -0.05
    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = 0.25      # home_advantage
    x0[2 * n + 1] = -0.05  # rho

    # Constraint: sum of attack params = 0
    constraints = [{
        "type": "eq",
        "fun": lambda p, n=n: np.sum(p[:n]),
    }]

    # Bounds: attacks/defenses unbounded-ish, home_adv > 0, rho in [-1, 1]
    bounds = (
        [(-3.0, 3.0)] * n +          # attack
        [(-3.0, 3.0)] * n +          # defense
        [(-0.5, 1.5)] +              # home_advantage
        [(-2.0, 2.0)]                # rho
    )

    result = minimize(
        _dc_log_likelihood,
        x0,
        args=(df, teams_in_data, weights),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    attack_params = result.x[:n]
    defense_params = result.x[n: 2 * n]
    home_advantage = result.x[2 * n]
    rho = result.x[2 * n + 1]

    attack_dict = {t: float(attack_params[i]) for i, t in enumerate(teams_in_data)}
    defense_dict = {t: float(defense_params[i]) for i, t in enumerate(teams_in_data)}

    return {
        "attack": attack_dict,
        "defense": defense_dict,
        "home_advantage": float(home_advantage),
        "rho": float(rho),
    }


def dixon_coles_predict(home_team, away_team, params,
                        congestion_home=False, congestion_away=False):
    """Produce a scoreline probability matrix from fitted Dixon-Coles params.

    Parameters
    ----------
    home_team, away_team : str
        Team names.
    params : dict
        Output of :func:`fit_dixon_coles`.
    congestion_home, congestion_away : bool
        If *True*, reduce the team's expected goals by
        ``CONGESTION_PENALTY``.

    Returns
    -------
    np.ndarray
        Shape ``(MAX_GOALS+1, MAX_GOALS+1)`` matrix where ``[i, j]`` is the
        probability of the home team scoring *i* and away team scoring *j*.
    """
    attack = params["attack"]
    defense = params["defense"]
    home_adv = params["home_advantage"]
    rho = params["rho"]

    lambda_h = math.exp(
        attack[home_team] + defense[away_team] + home_adv
    )
    lambda_a = math.exp(
        attack[away_team] + defense[home_team]
    )

    if congestion_home:
        lambda_h *= (1.0 - CONGESTION_PENALTY)
    if congestion_away:
        lambda_a *= (1.0 - CONGESTION_PENALTY)

    size = MAX_GOALS + 1
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            p_h = poisson.pmf(i, lambda_h)
            p_a = poisson.pmf(j, lambda_a)
            tau_val = _tau(i, j, lambda_h, lambda_a, rho)
            matrix[i, j] = p_h * p_a * tau_val

    # Normalise so the matrix sums to 1 (accounts for truncation at MAX_GOALS)
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def scoreline_to_probabilities(matrix):
    """Collapse a scoreline matrix into home / draw / away probabilities.

    Parameters
    ----------
    matrix : np.ndarray
        Shape ``(N, N)`` scoreline probability matrix.

    Returns
    -------
    dict
        Keys: ``home``, ``draw``, ``away`` — each a float probability.
    """
    size = matrix.shape[0]
    home = 0.0
    draw = 0.0
    away = 0.0

    for i in range(size):
        for j in range(size):
            p = matrix[i, j]
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p

    return {"home": home, "draw": draw, "away": away}


# ---------------------------------------------------------------------------
# Elo rating system
# ---------------------------------------------------------------------------

class EloRatings:
    """Simple Elo rating system with home advantage and goal-diff multiplier.

    Parameters
    ----------
    teams : list[str]
        Team names to initialise.
    initial_rating : float
        Starting Elo for every team (default 1500).
    k_factor : float or None
        K-factor for rating updates. Defaults to ``ELO_K_FACTOR`` from config.
    home_advantage : float or None
        Elo points added to home team. Defaults to ``ELO_HOME_ADVANTAGE``.
    """

    def __init__(self, teams, initial_rating=1500, k_factor=None, home_advantage=None):
        self.ratings = {t: float(initial_rating) for t in teams}
        self.k_factor = k_factor if k_factor is not None else ELO_K_FACTOR
        self.home_advantage = home_advantage if home_advantage is not None else ELO_HOME_ADVANTAGE

    def get_rating(self, team):
        """Return the current Elo rating for *team*."""
        return self.ratings.get(team, 1500.0)

    @staticmethod
    def expected_score(rating_a, rating_b):
        """Compute expected score for player A given both ratings."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, home, away, hg, ag):
        """Update ratings after a single match.

        Parameters
        ----------
        home, away : str
            Team names.
        hg, ag : int
            Goals scored by each team.
        """
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)

        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if hg > ag:
            s_home, s_away = 1.0, 0.0
        elif hg < ag:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # Expected scores — home team gets home_advantage boost
        e_home = self.expected_score(r_home + self.home_advantage, r_away)
        e_away = 1.0 - e_home

        # Goal-difference multiplier
        goal_diff = abs(hg - ag)
        if goal_diff <= 1:
            gd_mult = 1.0
        elif goal_diff == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11.0 + goal_diff) / 8.0

        k = self.k_factor * gd_mult

        self.ratings[home] = r_home + k * (s_home - e_home)
        self.ratings[away] = r_away + k * (s_away - e_away)

    def process_season(self, matches, recency_decay=0.003):
        """Process a DataFrame of matches in chronological order.

        Recent games receive higher K-factor via exponential decay so that
        current form matters more than early-season results.

        Parameters
        ----------
        matches : pd.DataFrame
            Must contain: home_team, away_team, home_goals, away_goals, date.
        recency_decay : float
            Decay rate for recency weighting.  0 = uniform weighting.
            Default 0.003 means a game from 6 months ago gets ~0.58× K.
        """
        df = matches.sort_values("date").reset_index(drop=True)

        # Compute per-game recency multiplier (1.0 for most recent, decays older)
        dates = pd.to_datetime(df["date"])
        most_recent = dates.max()
        days_ago = (most_recent - dates).dt.days.values.astype(float)
        recency = np.clip(np.exp(-recency_decay * days_ago), 0.5, 1.0)

        original_k = self.k_factor
        for idx, (_, row) in enumerate(df.iterrows()):
            self.k_factor = original_k * recency[idx]
            self.update(
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
            )
        self.k_factor = original_k


def elo_predict(elo, home_team, away_team, outcomes=None,
                home_rest_adj=0.0, away_rest_adj=0.0,
                home_advantage_override=None):
    """Convert Elo ratings into match-outcome probabilities.

    For 3-way outcomes (soccer), the draw probability is modelled as a
    function of rating closeness, calibrated to ~25% draws for typical gaps.
    For 2-way outcomes (basketball), a straight logistic is used.

    Parameters
    ----------
    elo : EloRatings
        Fitted Elo system.
    home_team, away_team : str
        Team names.
    outcomes : list[str] or None
        Valid outcomes, e.g. ``["home", "draw", "away"]`` or
        ``["home", "away"]``. Defaults to 3-way if not specified.
    home_rest_adj : float
        Elo-point adjustment for home team rest (negative = B2B penalty).
    away_rest_adj : float
        Elo-point adjustment for away team rest (negative = B2B penalty).
    home_advantage_override : float or None
        If provided, use this value instead of elo.home_advantage.

    Returns
    -------
    dict
        Probability for each outcome.
    """
    if outcomes is None:
        outcomes = ["home", "draw", "away"]

    home_adv = home_advantage_override if home_advantage_override is not None else elo.home_advantage
    r_home = elo.ratings.get(home_team, 1500.0) + home_adv + home_rest_adj
    r_away = elo.ratings.get(away_team, 1500.0) + away_rest_adj

    diff = r_home - r_away

    # Expected score for home (logistic)
    e_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    if "draw" not in outcomes:
        # 2-way: straight logistic
        return {"home": e_home, "away": 1.0 - e_home}

    # 3-way: carve out draw probability
    draw_base = 0.25
    draw_prob = draw_base * math.exp(-(diff ** 2) / (2 * 300 ** 2))
    draw_prob = max(0.10, min(0.35, draw_prob))

    remaining = 1.0 - draw_prob
    home_prob = remaining * e_home
    away_prob = remaining * (1.0 - e_home)

    return {"home": home_prob, "draw": draw_prob, "away": away_prob}


# ---------------------------------------------------------------------------
# Results-based feature model
# ---------------------------------------------------------------------------

class ResultsFeatureModel:
    """Two-way logistic model built from rolling team results features.

    The model is trained walk-forward style: each historical game is featurized
    from the teams' prior logs only, which avoids leaking future outcomes into
    training rows. It is intended for sports where we have reliable results but
    sparse box-score or roster data.
    """

    def __init__(self, games, feature_window=8, min_games=30):
        self.feature_window = feature_window
        self.min_games = min_games
        self.model = None
        self.team_logs = {}
        self.feature_names = [
            "season_win_pct_diff",
            "recent_win_pct_diff",
            "season_margin_diff",
            "recent_margin_diff",
            "venue_win_pct_diff",
            "rest_days_diff",
            "games_played_diff",
        ]
        self._fit(games)

    @staticmethod
    def _days_since_last(logs, cutoff_date=None):
        if not logs:
            return 7.0
        last_date = pd.to_datetime(logs[-1]["date"])
        current_date = pd.to_datetime(cutoff_date) if cutoff_date is not None else last_date
        days = (current_date - last_date).days
        return float(max(0, min(days, 7)))

    def _team_features(self, logs, venue=None, cutoff_date=None):
        if not logs:
            return {
                "season_win_pct": 0.5,
                "recent_win_pct": 0.5,
                "season_margin": 0.0,
                "recent_margin": 0.0,
                "venue_win_pct": 0.5,
                "rest_days": 7.0,
                "games_played": 0.0,
            }

        season_games = logs
        recent_games = logs[-self.feature_window :]
        venue_games = [g for g in logs if venue is None or g["venue"] == venue]

        def _win_pct(items):
            if not items:
                return 0.5
            return sum(g["result"] for g in items) / len(items)

        def _avg_margin(items):
            if not items:
                return 0.0
            return sum(g["margin"] for g in items) / len(items)

        return {
            "season_win_pct": _win_pct(season_games),
            "recent_win_pct": _win_pct(recent_games),
            "season_margin": _avg_margin(season_games),
            "recent_margin": _avg_margin(recent_games),
            "venue_win_pct": _win_pct(venue_games),
            "rest_days": self._days_since_last(logs, cutoff_date=cutoff_date),
            "games_played": float(len(season_games)),
        }

    def _feature_vector(self, home_logs, away_logs, neutral_site=False, cutoff_date=None):
        home_feats = self._team_features(
            home_logs, venue=None if neutral_site else "home", cutoff_date=cutoff_date
        )
        away_feats = self._team_features(
            away_logs, venue=None if neutral_site else "away", cutoff_date=cutoff_date
        )
        return np.array([
            home_feats["season_win_pct"] - away_feats["season_win_pct"],
            home_feats["recent_win_pct"] - away_feats["recent_win_pct"],
            home_feats["season_margin"] - away_feats["season_margin"],
            home_feats["recent_margin"] - away_feats["recent_margin"],
            home_feats["venue_win_pct"] - away_feats["venue_win_pct"],
            (home_feats["rest_days"] - away_feats["rest_days"]) / 7.0,
            (home_feats["games_played"] - away_feats["games_played"]) / 20.0,
        ])

    def _fit(self, games):
        if games is None or games.empty:
            return

        df = games.sort_values("date").reset_index(drop=True)
        team_logs = {
            team: []
            for team in sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        }

        X = []
        y = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            date_str = str(row["date"])[:10]
            if int(row["home_goals"]) == int(row["away_goals"]):
                continue

            home_logs = list(team_logs.get(home, []))
            away_logs = list(team_logs.get(away, []))

            X.append(self._feature_vector(home_logs, away_logs, cutoff_date=date_str))
            y.append(1 if int(row["home_goals"]) > int(row["away_goals"]) else 0)

            home_margin = int(row["home_goals"]) - int(row["away_goals"])
            away_margin = -home_margin
            team_logs.setdefault(home, []).append({
                "date": date_str,
                "result": 1.0 if home_margin > 0 else 0.0,
                "margin": float(home_margin),
                "venue": "home",
            })
            team_logs.setdefault(away, []).append({
                "date": date_str,
                "result": 1.0 if away_margin > 0 else 0.0,
                "margin": float(away_margin),
                "venue": "away",
            })

        self.team_logs = team_logs

        if len(X) < self.min_games or len(set(y)) < 2:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.array(X), np.array(y))

    def predict(self, home_team, away_team, neutral_site=False, game_date=None):
        if self.model is None:
            return {"home": 0.5, "away": 0.5}

        X = np.array([self._feature_vector(
            self.team_logs.get(home_team, []),
            self.team_logs.get(away_team, []),
            neutral_site=neutral_site,
            cutoff_date=game_date,
        )])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        home_idx = classes.index(1)
        away_idx = classes.index(0)
        return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}


def results_features_predict(model, home_team, away_team, neutral_site=False, game_date=None):
    """Predict two-way win probabilities from a results-feature model."""
    return model.predict(home_team, away_team, neutral_site=neutral_site, game_date=game_date)


# ---------------------------------------------------------------------------
# MLB starter matchup model
# ---------------------------------------------------------------------------

class PitcherMatchupModel:
    """Two-way logistic model built from historical starter-level performance."""

    def __init__(self, games, feature_window=8, min_games=20):
        self.feature_window = feature_window
        self.min_games = min_games
        self.model = None
        self.pitcher_logs = {}
        self.feature_names = [
            "season_ra9_diff",
            "recent_ra9_diff",
            "season_kbb_diff",
            "recent_kbb_diff",
            "recent_margin_diff",
            "recent_workload_diff",
            "days_rest_diff",
            "starts_diff",
        ]
        self._fit(games)

    def _pitcher_features(self, logs, game_date=None):
        if not logs:
            return {
                "season_ra9": 4.5,
                "recent_ra9": 4.5,
                "season_kbb": 0.0,
                "recent_kbb": 0.0,
                "recent_margin": 0.0,
                "recent_workload": 0.0,
                "days_rest": 5.0,
                "starts": 0.0,
            }

        recent = logs[-self.feature_window :]
        workload_slice = logs[-3:]

        def _ra9(items):
            innings = sum(item["innings"] for item in items)
            if innings <= 0:
                return 4.5
            return 9.0 * sum(item["earned_runs"] for item in items) / innings

        def _kbb(items):
            innings = sum(item["innings"] for item in items)
            if innings <= 0:
                return 0.0
            return (sum(item["strikeouts"] for item in items) - sum(item["walks"] for item in items)) / innings

        def _margin(items):
            return sum(item["margin"] for item in items) / len(items) if items else 0.0

        def _workload(items):
            return sum(item["innings"] for item in items) / len(items) if items else 0.0

        days_rest = 5.0
        if game_date is not None and logs:
            try:
                last_date = pd.to_datetime(logs[-1]["date"])
                days_rest = float(max(0, (pd.to_datetime(game_date) - last_date).days))
            except Exception:
                days_rest = 5.0

        return {
            "season_ra9": _ra9(logs),
            "recent_ra9": _ra9(recent),
            "season_kbb": _kbb(logs),
            "recent_kbb": _kbb(recent),
            "recent_margin": _margin(recent),
            "recent_workload": _workload(workload_slice),
            "days_rest": days_rest,
            "starts": float(len(logs)),
        }

    def _feature_vector(self, home_logs, away_logs, game_date=None):
        home = self._pitcher_features(home_logs, game_date=game_date)
        away = self._pitcher_features(away_logs, game_date=game_date)
        return np.array([
            away["season_ra9"] - home["season_ra9"],
            away["recent_ra9"] - home["recent_ra9"],
            home["season_kbb"] - away["season_kbb"],
            home["recent_kbb"] - away["recent_kbb"],
            home["recent_margin"] - away["recent_margin"],
            away["recent_workload"] - home["recent_workload"],
            (home["days_rest"] - away["days_rest"]) / 5.0,
            (home["starts"] - away["starts"]) / 10.0,
        ])

    def _fit(self, games):
        required = {
            "home_pitcher", "away_pitcher",
            "home_pitcher_ip", "away_pitcher_ip",
            "home_pitcher_earned_runs", "away_pitcher_earned_runs",
            "home_pitcher_walks", "away_pitcher_walks",
            "home_pitcher_strikeouts", "away_pitcher_strikeouts",
        }
        if games is None or games.empty or not required.issubset(set(games.columns)):
            return

        df = games.sort_values("date").reset_index(drop=True)
        pitcher_logs: dict[str, list[dict]] = {}
        X = []
        y = []

        for _, row in df.iterrows():
            home_pitcher = row.get("home_pitcher")
            away_pitcher = row.get("away_pitcher")
            if int(row["home_goals"]) == int(row["away_goals"]):
                continue
            if not home_pitcher or not away_pitcher or home_pitcher == "TBD" or away_pitcher == "TBD":
                continue

            home_logs = list(pitcher_logs.get(home_pitcher, []))
            away_logs = list(pitcher_logs.get(away_pitcher, []))
            X.append(self._feature_vector(home_logs, away_logs, game_date=row["date"]))
            y.append(1 if int(row["home_goals"]) > int(row["away_goals"]) else 0)

            home_margin = int(row["home_goals"]) - int(row["away_goals"])
            away_margin = -home_margin
            pitcher_logs.setdefault(home_pitcher, []).append({
                "date": row["date"],
                "innings": float(row.get("home_pitcher_ip", 0.0) or 0.0),
                "earned_runs": float(row.get("home_pitcher_earned_runs", 0.0) or 0.0),
                "walks": float(row.get("home_pitcher_walks", 0.0) or 0.0),
                "strikeouts": float(row.get("home_pitcher_strikeouts", 0.0) or 0.0),
                "margin": float(home_margin),
            })
            pitcher_logs.setdefault(away_pitcher, []).append({
                "date": row["date"],
                "innings": float(row.get("away_pitcher_ip", 0.0) or 0.0),
                "earned_runs": float(row.get("away_pitcher_earned_runs", 0.0) or 0.0),
                "walks": float(row.get("away_pitcher_walks", 0.0) or 0.0),
                "strikeouts": float(row.get("away_pitcher_strikeouts", 0.0) or 0.0),
                "margin": float(away_margin),
            })

        self.pitcher_logs = pitcher_logs

        if len(X) < self.min_games or len(set(y)) < 2:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.array(X), np.array(y))

    def predict(self, home_pitcher, away_pitcher, game_date=None):
        if self.model is None or not home_pitcher or not away_pitcher or home_pitcher == "TBD" or away_pitcher == "TBD":
            return {"home": 0.5, "away": 0.5}

        X = np.array([self._feature_vector(
            self.pitcher_logs.get(home_pitcher, []),
            self.pitcher_logs.get(away_pitcher, []),
            game_date=game_date,
        )])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        home_idx = classes.index(1)
        away_idx = classes.index(0)
        return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}


def pitcher_matchup_predict(model, home_pitcher, away_pitcher, game_date=None):
    """Predict two-way win probabilities from the starter matchup model."""
    return model.predict(home_pitcher, away_pitcher, game_date=game_date)


# ---------------------------------------------------------------------------
# MLB bullpen matchup model
# ---------------------------------------------------------------------------

class BullpenMatchupModel:
    """Two-way logistic model built from historical team bullpen performance."""

    def __init__(self, games, feature_window=12, recent_usage_window=5, min_games=20):
        self.feature_window = feature_window
        self.recent_usage_window = recent_usage_window
        self.min_games = min_games
        self.model = None
        self.team_logs = {}
        self.feature_names = [
            "season_ra9_diff",
            "recent_ra9_diff",
            "season_kbb_diff",
            "recent_kbb_diff",
            "recent_workload_diff",
            "recent_margin_support_diff",
        ]
        self._fit(games)

    def _bullpen_features(self, logs):
        if not logs:
            return {
                "season_ra9": 4.2,
                "recent_ra9": 4.2,
                "season_kbb": 0.0,
                "recent_kbb": 0.0,
                "recent_workload": 0.0,
                "recent_margin_support": 0.0,
            }

        recent = logs[-self.feature_window :]
        usage_slice = logs[-self.recent_usage_window :]

        def _ra9(items):
            innings = sum(item["innings"] for item in items)
            if innings <= 0:
                return 4.2
            return 9.0 * sum(item["earned_runs"] for item in items) / innings

        def _kbb(items):
            innings = sum(item["innings"] for item in items)
            if innings <= 0:
                return 0.0
            return (sum(item["strikeouts"] for item in items) - sum(item["walks"] for item in items)) / innings

        def _workload(items):
            return sum(item["innings"] for item in items) / max(1.0, float(len(items)))

        def _margin(items):
            return sum(item["margin"] for item in items) / len(items) if items else 0.0

        return {
            "season_ra9": _ra9(logs),
            "recent_ra9": _ra9(recent),
            "season_kbb": _kbb(logs),
            "recent_kbb": _kbb(recent),
            "recent_workload": _workload(usage_slice),
            "recent_margin_support": _margin(recent),
        }

    def _feature_vector(self, home_logs, away_logs):
        home = self._bullpen_features(home_logs)
        away = self._bullpen_features(away_logs)
        return np.array([
            away["season_ra9"] - home["season_ra9"],
            away["recent_ra9"] - home["recent_ra9"],
            home["season_kbb"] - away["season_kbb"],
            home["recent_kbb"] - away["recent_kbb"],
            away["recent_workload"] - home["recent_workload"],
            home["recent_margin_support"] - away["recent_margin_support"],
        ])

    def _fit(self, games):
        required = {
            "home_team", "away_team",
            "home_bullpen_ip", "away_bullpen_ip",
            "home_bullpen_earned_runs", "away_bullpen_earned_runs",
            "home_bullpen_walks", "away_bullpen_walks",
            "home_bullpen_strikeouts", "away_bullpen_strikeouts",
        }
        if games is None or games.empty or not required.issubset(set(games.columns)):
            return

        df = games.sort_values("date").reset_index(drop=True)
        team_logs: dict[str, list[dict]] = {}
        X = []
        y = []

        for _, row in df.iterrows():
            if int(row["home_goals"]) == int(row["away_goals"]):
                continue

            home_team = row["home_team"]
            away_team = row["away_team"]
            home_logs = list(team_logs.get(home_team, []))
            away_logs = list(team_logs.get(away_team, []))
            X.append(self._feature_vector(home_logs, away_logs))
            y.append(1 if int(row["home_goals"]) > int(row["away_goals"]) else 0)

            home_margin = int(row["home_goals"]) - int(row["away_goals"])
            away_margin = -home_margin
            team_logs.setdefault(home_team, []).append({
                "innings": float(row.get("home_bullpen_ip", 0.0) or 0.0),
                "earned_runs": float(row.get("home_bullpen_earned_runs", 0.0) or 0.0),
                "walks": float(row.get("home_bullpen_walks", 0.0) or 0.0),
                "strikeouts": float(row.get("home_bullpen_strikeouts", 0.0) or 0.0),
                "margin": float(home_margin),
            })
            team_logs.setdefault(away_team, []).append({
                "innings": float(row.get("away_bullpen_ip", 0.0) or 0.0),
                "earned_runs": float(row.get("away_bullpen_earned_runs", 0.0) or 0.0),
                "walks": float(row.get("away_bullpen_walks", 0.0) or 0.0),
                "strikeouts": float(row.get("away_bullpen_strikeouts", 0.0) or 0.0),
                "margin": float(away_margin),
            })

        self.team_logs = team_logs

        if len(X) < self.min_games or len(set(y)) < 2:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.array(X), np.array(y))

    def predict(self, home_team, away_team):
        if self.model is None:
            return {"home": 0.5, "away": 0.5}

        X = np.array([self._feature_vector(
            self.team_logs.get(home_team, []),
            self.team_logs.get(away_team, []),
        )])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        home_idx = classes.index(1)
        away_idx = classes.index(0)
        return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}


def bullpen_matchup_predict(model, home_team, away_team):
    """Predict two-way win probabilities from the bullpen matchup model."""
    return model.predict(home_team, away_team)


# ---------------------------------------------------------------------------
# MLB handedness matchup model
# ---------------------------------------------------------------------------

class HandednessMatchupModel:
    """Two-way logistic model using team offense splits versus pitcher hand."""

    def __init__(self, games, feature_window=18, min_games=20):
        self.feature_window = feature_window
        self.min_games = min_games
        self.model = None
        self.team_logs = {}
        self.feature_names = [
            "season_runs_split_diff",
            "recent_runs_split_diff",
            "season_margin_split_diff",
            "recent_margin_split_diff",
        ]
        self._fit(games)

    def _split_features(self, logs, pitcher_hand):
        if not logs:
            return {
                "season_runs": 4.4,
                "recent_runs": 4.4,
                "season_margin": 0.0,
                "recent_margin": 0.0,
            }

        split_logs = [item for item in logs if item.get("opponent_hand") == pitcher_hand]
        if not split_logs:
            split_logs = list(logs)
        recent = split_logs[-self.feature_window :]

        def _avg(items, key, default):
            return float(sum(item[key] for item in items) / len(items)) if items else default

        return {
            "season_runs": _avg(split_logs, "runs", 4.4),
            "recent_runs": _avg(recent, "runs", 4.4),
            "season_margin": _avg(split_logs, "margin", 0.0),
            "recent_margin": _avg(recent, "margin", 0.0),
        }

    def _feature_vector(self, home_logs, away_logs, home_pitcher_hand, away_pitcher_hand):
        home = self._split_features(home_logs, away_pitcher_hand)
        away = self._split_features(away_logs, home_pitcher_hand)
        return np.array([
            home["season_runs"] - away["season_runs"],
            home["recent_runs"] - away["recent_runs"],
            home["season_margin"] - away["season_margin"],
            home["recent_margin"] - away["recent_margin"],
        ])

    def _fit(self, games):
        required = {
            "home_team", "away_team", "home_goals", "away_goals",
            "home_pitcher_hand", "away_pitcher_hand",
        }
        if games is None or games.empty or not required.issubset(set(games.columns)):
            return

        df = games.sort_values("date").reset_index(drop=True)
        team_logs: dict[str, list[dict]] = {}
        X = []
        y = []

        for _, row in df.iterrows():
            if int(row["home_goals"]) == int(row["away_goals"]):
                continue

            home_hand = row.get("home_pitcher_hand") or "R"
            away_hand = row.get("away_pitcher_hand") or "R"
            home_team = row["home_team"]
            away_team = row["away_team"]

            X.append(self._feature_vector(
                team_logs.get(home_team, []),
                team_logs.get(away_team, []),
                home_hand,
                away_hand,
            ))
            y.append(1 if int(row["home_goals"]) > int(row["away_goals"]) else 0)

            home_margin = int(row["home_goals"]) - int(row["away_goals"])
            away_margin = -home_margin
            team_logs.setdefault(home_team, []).append({
                "runs": float(row["home_goals"]),
                "margin": float(home_margin),
                "opponent_hand": away_hand,
            })
            team_logs.setdefault(away_team, []).append({
                "runs": float(row["away_goals"]),
                "margin": float(away_margin),
                "opponent_hand": home_hand,
            })

        self.team_logs = team_logs

        if len(X) < self.min_games or len(set(y)) < 2:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.array(X), np.array(y))

    def predict(self, home_team, away_team, home_pitcher_hand=None, away_pitcher_hand=None):
        if self.model is None:
            return {"home": 0.5, "away": 0.5}

        X = np.array([self._feature_vector(
            self.team_logs.get(home_team, []),
            self.team_logs.get(away_team, []),
            home_pitcher_hand or "R",
            away_pitcher_hand or "R",
        )])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        home_idx = classes.index(1)
        away_idx = classes.index(0)
        return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}


def handedness_matchup_predict(model, home_team, away_team, home_pitcher_hand=None, away_pitcher_hand=None):
    """Predict two-way probabilities from the MLB handedness matchup model."""
    return model.predict(
        home_team,
        away_team,
        home_pitcher_hand=home_pitcher_hand,
        away_pitcher_hand=away_pitcher_hand,
    )


# ---------------------------------------------------------------------------
# MLB run-environment model
# ---------------------------------------------------------------------------

class RunEnvironmentModel:
    """Two-way logistic model using scoring form plus park context for MLB."""

    def __init__(self, games, feature_window=12, min_games=20, park_factors=None):
        self.feature_window = feature_window
        self.min_games = min_games
        self.park_factors = park_factors or MLB_PARK_FACTORS
        self.model = None
        self.team_logs = {}
        self.feature_names = [
            "season_runs_for_diff",
            "season_runs_against_diff",
            "recent_runs_for_diff",
            "recent_runs_against_diff",
            "recent_margin_diff",
            "home_park_factor",
        ]
        self._fit(games)

    def _team_features(self, logs):
        if not logs:
            return {
                "season_runs_for": 4.4,
                "season_runs_against": 4.4,
                "recent_runs_for": 4.4,
                "recent_runs_against": 4.4,
                "recent_margin": 0.0,
            }

        recent = logs[-self.feature_window :]

        def _avg(items, key, default):
            return float(sum(item[key] for item in items) / len(items)) if items else default

        return {
            "season_runs_for": _avg(logs, "runs_for", 4.4),
            "season_runs_against": _avg(logs, "runs_against", 4.4),
            "recent_runs_for": _avg(recent, "runs_for", 4.4),
            "recent_runs_against": _avg(recent, "runs_against", 4.4),
            "recent_margin": _avg(recent, "margin", 0.0),
        }

    def _feature_vector(self, home_logs, away_logs, home_team):
        home = self._team_features(home_logs)
        away = self._team_features(away_logs)
        park_factor = float(self.park_factors.get(home_team, 1.0))
        return np.array([
            home["season_runs_for"] - away["season_runs_for"],
            away["season_runs_against"] - home["season_runs_against"],
            home["recent_runs_for"] - away["recent_runs_for"],
            away["recent_runs_against"] - home["recent_runs_against"],
            home["recent_margin"] - away["recent_margin"],
            park_factor - 1.0,
        ])

    def _fit(self, games):
        required = {"home_team", "away_team", "home_goals", "away_goals"}
        if games is None or games.empty or not required.issubset(set(games.columns)):
            return

        df = games.sort_values("date").reset_index(drop=True)
        team_logs: dict[str, list[dict]] = {}
        X = []
        y = []

        for _, row in df.iterrows():
            if int(row["home_goals"]) == int(row["away_goals"]):
                continue

            home_team = row["home_team"]
            away_team = row["away_team"]
            home_logs = list(team_logs.get(home_team, []))
            away_logs = list(team_logs.get(away_team, []))
            X.append(self._feature_vector(home_logs, away_logs, home_team))
            y.append(1 if int(row["home_goals"]) > int(row["away_goals"]) else 0)

            home_margin = int(row["home_goals"]) - int(row["away_goals"])
            away_margin = -home_margin
            team_logs.setdefault(home_team, []).append({
                "runs_for": float(row["home_goals"]),
                "runs_against": float(row["away_goals"]),
                "margin": float(home_margin),
            })
            team_logs.setdefault(away_team, []).append({
                "runs_for": float(row["away_goals"]),
                "runs_against": float(row["home_goals"]),
                "margin": float(away_margin),
            })

        self.team_logs = team_logs

        if len(X) < self.min_games or len(set(y)) < 2:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.array(X), np.array(y))

    def predict(self, home_team, away_team):
        if self.model is None:
            return {"home": 0.5, "away": 0.5}

        X = np.array([self._feature_vector(
            self.team_logs.get(home_team, []),
            self.team_logs.get(away_team, []),
            home_team,
        )])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        home_idx = classes.index(1)
        away_idx = classes.index(0)
        return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}


def run_environment_predict(model, home_team, away_team):
    """Predict two-way win probabilities from the MLB run-environment model."""
    return model.predict(home_team, away_team)


# ---------------------------------------------------------------------------
# MLB totals model
# ---------------------------------------------------------------------------

class MlbTotalsModel:
    """Walk-forward regression model for MLB game totals."""

    def __init__(self, games, feature_window=12, min_games=20, park_factors=None, default_stddev=3.1):
        self.feature_window = feature_window
        self.min_games = min_games
        self.park_factors = park_factors or MLB_PARK_FACTORS
        self.default_stddev = default_stddev
        self.model = None
        self.team_logs = {}
        self.pitcher_logs = {}
        self.feature_names = [
            "home_recent_runs_for",
            "away_recent_runs_for",
            "home_recent_runs_allowed",
            "away_recent_runs_allowed",
            "home_starter_recent_ra9",
            "away_starter_recent_ra9",
            "home_bullpen_recent_ra9",
            "away_bullpen_recent_ra9",
            "park_factor",
        ]
        self.residual_std = default_stddev
        self._fit(games)

    def _recent_team_rates(self, team_logs):
        if not team_logs:
            return {"runs_for": 4.4, "runs_allowed": 4.4}
        recent = team_logs[-self.feature_window :]
        return {
            "runs_for": float(sum(item["runs_for"] for item in recent) / len(recent)),
            "runs_allowed": float(sum(item["runs_allowed"] for item in recent) / len(recent)),
        }

    def _recent_ra9(self, logs, key, default):
        if not logs:
            return default
        recent = logs[-self.feature_window :]
        innings = sum(item["innings"] for item in recent)
        if innings <= 0:
            return default
        return 9.0 * sum(item[key] for item in recent) / innings

    def _feature_vector(self, row, team_logs, pitcher_logs):
        home_team = row["home_team"]
        away_team = row["away_team"]
        home_rates = self._recent_team_rates(team_logs.get(home_team, []))
        away_rates = self._recent_team_rates(team_logs.get(away_team, []))
        home_pitcher = row.get("home_pitcher")
        away_pitcher = row.get("away_pitcher")
        home_park = float(self.park_factors.get(home_team, 1.0))

        return np.array([
            home_rates["runs_for"],
            away_rates["runs_for"],
            home_rates["runs_allowed"],
            away_rates["runs_allowed"],
            self._recent_ra9(pitcher_logs.get(home_pitcher, []), "earned_runs", 4.4),
            self._recent_ra9(pitcher_logs.get(away_pitcher, []), "earned_runs", 4.4),
            self._recent_ra9(team_logs.get(home_team, []), "bullpen_earned_runs", 4.2),
            self._recent_ra9(team_logs.get(away_team, []), "bullpen_earned_runs", 4.2),
            home_park,
        ])

    def _fit(self, games):
        required = {
            "home_team", "away_team", "home_goals", "away_goals",
            "home_pitcher", "away_pitcher",
            "home_pitcher_ip", "away_pitcher_ip",
            "home_pitcher_earned_runs", "away_pitcher_earned_runs",
            "home_bullpen_ip", "away_bullpen_ip",
            "home_bullpen_earned_runs", "away_bullpen_earned_runs",
        }
        if games is None or games.empty or not required.issubset(set(games.columns)):
            return

        df = games.sort_values("date").reset_index(drop=True)
        team_logs: dict[str, list[dict]] = {}
        pitcher_logs: dict[str, list[dict]] = {}
        X = []
        y = []

        for _, row in df.iterrows():
            X.append(self._feature_vector(row, team_logs, pitcher_logs))
            y.append(float(row["home_goals"]) + float(row["away_goals"]))

            home_team = row["home_team"]
            away_team = row["away_team"]
            home_pitcher = row.get("home_pitcher")
            away_pitcher = row.get("away_pitcher")

            team_logs.setdefault(home_team, []).append({
                "runs_for": float(row["home_goals"]),
                "runs_allowed": float(row["away_goals"]),
                "innings": float(row.get("home_bullpen_ip", 0.0) or 0.0),
                "bullpen_earned_runs": float(row.get("home_bullpen_earned_runs", 0.0) or 0.0),
            })
            team_logs.setdefault(away_team, []).append({
                "runs_for": float(row["away_goals"]),
                "runs_allowed": float(row["home_goals"]),
                "innings": float(row.get("away_bullpen_ip", 0.0) or 0.0),
                "bullpen_earned_runs": float(row.get("away_bullpen_earned_runs", 0.0) or 0.0),
            })
            if home_pitcher and home_pitcher != "TBD":
                pitcher_logs.setdefault(home_pitcher, []).append({
                    "innings": float(row.get("home_pitcher_ip", 0.0) or 0.0),
                    "earned_runs": float(row.get("home_pitcher_earned_runs", 0.0) or 0.0),
                })
            if away_pitcher and away_pitcher != "TBD":
                pitcher_logs.setdefault(away_pitcher, []).append({
                    "innings": float(row.get("away_pitcher_ip", 0.0) or 0.0),
                    "earned_runs": float(row.get("away_pitcher_earned_runs", 0.0) or 0.0),
                })

        self.team_logs = team_logs
        self.pitcher_logs = pitcher_logs

        if len(X) < self.min_games:
            return

        self.model = LinearRegression()
        self.model.fit(np.array(X), np.array(y))
        preds = self.model.predict(np.array(X))
        residuals = np.array(y) - preds
        if len(residuals) >= 5:
            self.residual_std = max(1.5, float(np.std(residuals)))

    def predict_total(self, fixture: dict) -> float:
        if self.model is None:
            return 8.4
        features = self._feature_vector(fixture, self.team_logs, self.pitcher_logs)
        total = float(self.model.predict(np.array([features]))[0])
        return max(4.5, min(16.0, total))

    def predict_market(self, fixture: dict, total_line: float) -> dict[str, float]:
        expected_total = self.predict_total(fixture)
        sigma = max(1.5, float(self.residual_std or self.default_stddev))
        over_prob = float(1.0 - norm.cdf(total_line, loc=expected_total, scale=sigma))
        over_prob = max(0.01, min(0.99, over_prob))
        under_prob = 1.0 - over_prob
        return {
            "expected_total": expected_total,
            "stddev": sigma,
            "over": over_prob,
            "under": under_prob,
        }


def mlb_totals_predict(model, fixture: dict, total_line: float) -> dict[str, float]:
    """Predict MLB over/under probabilities and expected total."""
    return model.predict_market(fixture, total_line)


# ---------------------------------------------------------------------------
# Recent box-score matchup model
# ---------------------------------------------------------------------------

class RecentBoxScoreModel:
    """Two-way logistic model built from walk-forward team box-score form.

    This captures recent matchup strength more explicitly than the season-level
    efficiency and four-factors layers. Each training row is featurized from
    each team's prior box-score logs only.
    """

    def __init__(self, box_scores, games, feature_window=8, min_games=30):
        self.feature_window = feature_window
        self.min_games = min_games
        self.model = None
        self.team_logs = {}
        self.feature_names = [
            "season_net_rating_diff",
            "recent_net_rating_diff",
            "season_efg_diff",
            "recent_efg_diff",
            "season_to_rate_diff",
            "recent_rebound_pct_diff",
            "season_ft_rate_diff",
            "pace_diff",
        ]
        self._fit(box_scores, games)

    @staticmethod
    def _aggregate(items):
        if not items:
            return {
                "net_rating": 0.0,
                "efg": 0.5,
                "to_rate": 0.14,
                "rebound_pct": 0.5,
                "ft_rate": 0.22,
                "pace": 70.0,
            }

        def _avg(key):
            return float(sum(item[key] for item in items) / len(items))

        return {
            "net_rating": _avg("net_rating"),
            "efg": _avg("efg"),
            "to_rate": _avg("to_rate"),
            "rebound_pct": _avg("rebound_pct"),
            "ft_rate": _avg("ft_rate"),
            "pace": _avg("pace"),
        }

    def _feature_vector(self, home_logs, away_logs):
        home_season = self._aggregate(home_logs)
        away_season = self._aggregate(away_logs)
        home_recent = self._aggregate(home_logs[-self.feature_window :])
        away_recent = self._aggregate(away_logs[-self.feature_window :])

        return np.array([
            home_season["net_rating"] - away_season["net_rating"],
            home_recent["net_rating"] - away_recent["net_rating"],
            home_season["efg"] - away_season["efg"],
            home_recent["efg"] - away_recent["efg"],
            away_season["to_rate"] - home_season["to_rate"],
            home_recent["rebound_pct"] - away_recent["rebound_pct"],
            home_season["ft_rate"] - away_season["ft_rate"],
            (home_recent["pace"] - away_recent["pace"]) / 20.0,
        ])

    def _fit(self, box_scores, games):
        required_box = {"game_id", "team", "pts", "fgm", "fga", "fg3m", "fta", "orb", "drb", "to", "possessions"}
        required_games = {"game_id", "date", "home_team", "away_team", "home_goals", "away_goals"}
        if (
            box_scores is None
            or games is None
            or box_scores.empty
            or games.empty
            or not required_box.issubset(set(box_scores.columns))
            or not required_games.issubset(set(games.columns))
        ):
            return

        bs_lookup = {}
        for _, row in box_scores.iterrows():
            bs_lookup[(row["game_id"], row["team"])] = row

        df = games.sort_values("date").reset_index(drop=True)
        team_logs = {
            team: []
            for team in sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        }
        X = []
        y = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_bs = bs_lookup.get((row["game_id"], home))
            away_bs = bs_lookup.get((row["game_id"], away))
            if home_bs is None or away_bs is None:
                continue
            if int(row["home_goals"]) == int(row["away_goals"]):
                continue

            X.append(self._feature_vector(team_logs.get(home, []), team_logs.get(away, [])))
            y.append(1 if int(row["home_goals"]) > int(row["away_goals"]) else 0)

            home_log = self._boxscore_log(home_bs, away_bs)
            away_log = self._boxscore_log(away_bs, home_bs)
            team_logs.setdefault(home, []).append(home_log)
            team_logs.setdefault(away, []).append(away_log)

        self.team_logs = team_logs

        if len(X) < self.min_games or len(set(y)) < 2:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.array(X), np.array(y))

    @staticmethod
    def _boxscore_log(team_bs, opp_bs):
        fga = max(float(team_bs["fga"]), 1.0)
        poss = max(float(team_bs["possessions"]), 1.0)
        rebound_denom = float(team_bs["orb"]) + float(opp_bs["drb"])
        return {
            "net_rating": ((float(team_bs["pts"]) / poss) - (float(opp_bs["pts"]) / max(float(opp_bs["possessions"]), 1.0))) * 100.0,
            "efg": (float(team_bs["fgm"]) + 0.5 * float(team_bs["fg3m"])) / fga,
            "to_rate": float(team_bs["to"]) / poss,
            "rebound_pct": float(team_bs["orb"]) / max(rebound_denom, 1.0),
            "ft_rate": float(team_bs["fta"]) / fga,
            "pace": poss,
        }

    def predict(self, home_team, away_team):
        if self.model is None:
            return {"home": 0.5, "away": 0.5}

        X = np.array([self._feature_vector(
            self.team_logs.get(home_team, []),
            self.team_logs.get(away_team, []),
        )])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        home_idx = classes.index(1)
        away_idx = classes.index(0)
        return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}


def recent_boxscore_predict(model, home_team, away_team):
    """Predict two-way win probabilities from the recent box-score model."""
    return model.predict(home_team, away_team)


# ---------------------------------------------------------------------------
# NBA matchup context model
# ---------------------------------------------------------------------------

class NbaMatchupModel:
    """NBA-specific walk-forward model using venue, rest, and style context."""

    def __init__(self, box_scores, games, feature_window=8, min_games=30):
        self.feature_window = feature_window
        self.min_games = min_games
        self.model = None
        self.team_logs = {}
        self.feature_names = [
            "season_net_rating_diff",
            "recent_net_rating_diff",
            "venue_net_rating_diff",
            "offense_defense_edge",
            "efg_diff",
            "turnover_edge",
            "rebound_edge",
            "ft_rate_diff",
            "three_point_rate_diff",
            "pace_delta",
            "rest_diff",
            "home_recent_margin",
        ]
        self._fit(box_scores, games)

    @staticmethod
    def _default_summary():
        return {
            "off_rating": 110.0,
            "def_rating": 110.0,
            "net_rating": 0.0,
            "efg": 0.53,
            "to_rate": 0.13,
            "rebound_pct": 0.23,
            "ft_rate": 0.22,
            "three_point_rate": 0.38,
            "pace": 98.0,
            "margin": 0.0,
        }

    def _aggregate(self, logs):
        if not logs:
            return self._default_summary()

        def _avg(key):
            return float(sum(item[key] for item in logs) / len(logs))

        return {
            "off_rating": _avg("off_rating"),
            "def_rating": _avg("def_rating"),
            "net_rating": _avg("net_rating"),
            "efg": _avg("efg"),
            "to_rate": _avg("to_rate"),
            "rebound_pct": _avg("rebound_pct"),
            "ft_rate": _avg("ft_rate"),
            "three_point_rate": _avg("three_point_rate"),
            "pace": _avg("pace"),
            "margin": _avg("margin"),
        }

    def _rest_days(self, logs, game_date):
        if not logs or game_date is None:
            return 2.0
        try:
            current = pd.Timestamp(game_date)
        except (TypeError, ValueError):
            return 2.0

        dated_logs = [item for item in logs if item.get("date") is not None]
        if not dated_logs:
            return 2.0
        last_date = max(pd.Timestamp(item["date"]) for item in dated_logs)
        return max(0.0, float((current - last_date).days))

    def _feature_vector(self, home_logs, away_logs, game_date=None):
        home_season = self._aggregate(home_logs)
        away_season = self._aggregate(away_logs)
        home_recent = self._aggregate(home_logs[-self.feature_window :])
        away_recent = self._aggregate(away_logs[-self.feature_window :])
        home_home = self._aggregate([item for item in home_logs if item.get("venue") == "home"][-self.feature_window :])
        away_away = self._aggregate([item for item in away_logs if item.get("venue") == "away"][-self.feature_window :])

        return np.array([
            home_season["net_rating"] - away_season["net_rating"],
            home_recent["net_rating"] - away_recent["net_rating"],
            home_home["net_rating"] - away_away["net_rating"],
            (home_recent["off_rating"] - away_recent["def_rating"]) - (away_recent["off_rating"] - home_recent["def_rating"]),
            home_recent["efg"] - away_recent["efg"],
            away_recent["to_rate"] - home_recent["to_rate"],
            home_recent["rebound_pct"] - away_recent["rebound_pct"],
            home_recent["ft_rate"] - away_recent["ft_rate"],
            home_recent["three_point_rate"] - away_recent["three_point_rate"],
            (home_recent["pace"] - away_recent["pace"]) / 18.0,
            (self._rest_days(home_logs, game_date) - self._rest_days(away_logs, game_date)) / 3.0,
            home_recent["margin"],
        ])

    @staticmethod
    def _boxscore_log(team_bs, opp_bs, venue, game_date, team_points, opp_points):
        team_poss = max(float(team_bs["possessions"]), 1.0)
        opp_poss = max(float(opp_bs["possessions"]), 1.0)
        fga = max(float(team_bs["fga"]), 1.0)
        return {
            "date": game_date,
            "venue": venue,
            "off_rating": (float(team_bs["pts"]) / team_poss) * 100.0,
            "def_rating": (float(opp_bs["pts"]) / opp_poss) * 100.0,
            "net_rating": ((float(team_bs["pts"]) / team_poss) - (float(opp_bs["pts"]) / opp_poss)) * 100.0,
            "efg": (float(team_bs["fgm"]) + (0.5 * float(team_bs["fg3m"]))) / fga,
            "to_rate": float(team_bs["to"]) / team_poss,
            "rebound_pct": float(team_bs["orb"]) / max(float(team_bs["orb"]) + float(opp_bs["drb"]), 1.0),
            "ft_rate": float(team_bs["fta"]) / fga,
            "three_point_rate": float(team_bs["fg3a"]) / fga,
            "pace": (team_poss + opp_poss) / 2.0,
            "margin": float(team_points - opp_points),
        }

    def _fit(self, box_scores, games):
        required_box = {"game_id", "team", "pts", "fgm", "fga", "fg3m", "possessions"}
        required_games = {"game_id", "date", "home_team", "away_team", "home_goals", "away_goals"}
        if (
            box_scores is None
            or games is None
            or box_scores.empty
            or games.empty
            or not required_box.issubset(set(box_scores.columns))
            or not required_games.issubset(set(games.columns))
        ):
            return

        bs_lookup = {}
        for _, row in box_scores.iterrows():
            bs_lookup[(row["game_id"], row["team"])] = row

        df = games.sort_values("date").reset_index(drop=True)
        team_logs = {
            team: []
            for team in sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        }
        X = []
        y = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_bs = bs_lookup.get((row["game_id"], home))
            away_bs = bs_lookup.get((row["game_id"], away))
            if home_bs is None or away_bs is None:
                continue
            if int(row["home_goals"]) == int(row["away_goals"]):
                continue

            X.append(self._feature_vector(team_logs.get(home, []), team_logs.get(away, []), game_date=row["date"]))
            y.append(1 if int(row["home_goals"]) > int(row["away_goals"]) else 0)

            team_logs.setdefault(home, []).append(
                self._boxscore_log(
                    home_bs,
                    away_bs,
                    "home",
                    row["date"],
                    row["home_goals"],
                    row["away_goals"],
                )
            )
            team_logs.setdefault(away, []).append(
                self._boxscore_log(
                    away_bs,
                    home_bs,
                    "away",
                    row["date"],
                    row["away_goals"],
                    row["home_goals"],
                )
            )

        self.team_logs = team_logs

        if len(X) < self.min_games or len(set(y)) < 2:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.array(X), np.array(y))

    def predict(self, home_team, away_team, game_date=None):
        if self.model is None:
            return {"home": 0.5, "away": 0.5}

        X = np.array([self._feature_vector(
            self.team_logs.get(home_team, []),
            self.team_logs.get(away_team, []),
            game_date=game_date,
        )])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        home_idx = classes.index(1)
        away_idx = classes.index(0)
        return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}


def nba_matchup_predict(model, home_team, away_team, game_date=None):
    """Predict two-way probabilities from the NBA matchup-context model."""
    return model.predict(home_team, away_team, game_date=game_date)


# ---------------------------------------------------------------------------
# NHL matchup context model
# ---------------------------------------------------------------------------

class NhlMatchupModel:
    """NHL walk-forward model using scoring, shooting, and goaltending context."""

    def __init__(self, games, feature_window=10, min_games=40):
        self.feature_window = feature_window
        self.min_games = min_games
        self.model = None
        self.team_logs = {}
        self.feature_names = [
            "season_goal_diff",
            "recent_goal_diff",
            "venue_goal_diff",
            "recent_shot_diff",
            "season_save_pct_diff",
            "recent_save_pct_diff",
            "rest_diff",
        ]
        self._fit(games)

    @staticmethod
    def _default_summary():
        return {
            "goal_diff": 0.0,
            "shot_diff": 0.0,
            "save_pct": 0.91,
        }

    def _aggregate(self, logs):
        if not logs:
            return self._default_summary()

        def _avg(key):
            return float(sum(item[key] for item in logs) / len(logs))

        return {
            "goal_diff": _avg("goal_diff"),
            "shot_diff": _avg("shot_diff"),
            "save_pct": _avg("save_pct"),
        }

    def _rest_days(self, logs, game_date):
        if not logs or game_date is None:
            return 2.0
        try:
            current = pd.Timestamp(game_date)
        except (TypeError, ValueError):
            return 2.0

        dated_logs = [item for item in logs if item.get("date") is not None]
        if not dated_logs:
            return 2.0
        last_date = max(pd.Timestamp(item["date"]) for item in dated_logs)
        return max(0.0, float((current - last_date).days))

    def _feature_vector(self, home_logs, away_logs, game_date=None):
        home_season = self._aggregate(home_logs)
        away_season = self._aggregate(away_logs)
        home_recent = self._aggregate(home_logs[-self.feature_window :])
        away_recent = self._aggregate(away_logs[-self.feature_window :])
        home_home = self._aggregate([item for item in home_logs if item.get("venue") == "home"][-self.feature_window :])
        away_away = self._aggregate([item for item in away_logs if item.get("venue") == "away"][-self.feature_window :])

        return np.array([
            home_season["goal_diff"] - away_season["goal_diff"],
            home_recent["goal_diff"] - away_recent["goal_diff"],
            home_home["goal_diff"] - away_away["goal_diff"],
            home_recent["shot_diff"] - away_recent["shot_diff"],
            home_season["save_pct"] - away_season["save_pct"],
            home_recent["save_pct"] - away_recent["save_pct"],
            (self._rest_days(home_logs, game_date) - self._rest_days(away_logs, game_date)) / 3.0,
        ])

    @staticmethod
    def _game_log(row, venue, goals_for, goals_against, shots_for, shots_against, save_pct):
        return {
            "date": row["date"],
            "venue": venue,
            "goal_diff": float(goals_for - goals_against),
            "shot_diff": float(shots_for - shots_against),
            "save_pct": float(save_pct),
        }

    def _fit(self, games):
        required = {
            "date", "home_team", "away_team", "home_goals", "away_goals",
            "home_shots", "away_shots", "home_save_pct", "away_save_pct",
        }
        if games is None or games.empty or not required.issubset(set(games.columns)):
            return

        df = games.sort_values("date").reset_index(drop=True)
        team_logs = {
            team: []
            for team in sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        }
        X = []
        y = []

        for _, row in df.iterrows():
            if int(row["home_goals"]) == int(row["away_goals"]):
                continue
            home = row["home_team"]
            away = row["away_team"]

            X.append(self._feature_vector(team_logs.get(home, []), team_logs.get(away, []), game_date=row["date"]))
            y.append(1 if int(row["home_goals"]) > int(row["away_goals"]) else 0)

            team_logs.setdefault(home, []).append(
                self._game_log(
                    row,
                    "home",
                    int(row["home_goals"]),
                    int(row["away_goals"]),
                    int(row.get("home_shots", 0)),
                    int(row.get("away_shots", 0)),
                    float(row.get("home_save_pct", 0.91)),
                )
            )
            team_logs.setdefault(away, []).append(
                self._game_log(
                    row,
                    "away",
                    int(row["away_goals"]),
                    int(row["home_goals"]),
                    int(row.get("away_shots", 0)),
                    int(row.get("home_shots", 0)),
                    float(row.get("away_save_pct", 0.91)),
                )
            )

        self.team_logs = team_logs

        if len(X) < self.min_games or len(set(y)) < 2:
            return

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.array(X), np.array(y))

    def predict(self, home_team, away_team, game_date=None):
        if self.model is None:
            return {"home": 0.5, "away": 0.5}

        X = np.array([self._feature_vector(
            self.team_logs.get(home_team, []),
            self.team_logs.get(away_team, []),
            game_date=game_date,
        )])
        proba = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)
        home_idx = classes.index(1)
        away_idx = classes.index(0)
        return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}


def nhl_matchup_predict(model, home_team, away_team, game_date=None):
    """Predict two-way probabilities from the NHL matchup-context model."""
    return model.predict(home_team, away_team, game_date=game_date)


# ---------------------------------------------------------------------------
# NBA totals model
# ---------------------------------------------------------------------------

class NbaTotalsModel:
    """Walk-forward regression model for NBA game totals."""

    def __init__(self, box_scores, games, feature_window=8, min_games=30, default_stddev=13.5):
        self.feature_window = feature_window
        self.min_games = min_games
        self.default_stddev = default_stddev
        self.model = None
        self.team_logs = {}
        self.residual_std = default_stddev
        self.feature_names = [
            "home_recent_points_for",
            "away_recent_points_for",
            "home_recent_points_allowed",
            "away_recent_points_allowed",
            "home_recent_pace",
            "away_recent_pace",
            "home_recent_efg",
            "away_recent_efg",
            "home_recent_ft_rate",
            "away_recent_ft_rate",
            "home_recent_to_rate",
            "away_recent_to_rate",
            "home_home_points_for",
            "away_away_points_for",
            "combined_recent_total",
            "combined_rest",
        ]
        self._fit(box_scores, games)

    @staticmethod
    def _default_summary():
        return {
            "points_for": 112.0,
            "points_allowed": 112.0,
            "pace": 98.0,
            "efg": 0.53,
            "ft_rate": 0.22,
            "to_rate": 0.13,
        }

    def _aggregate(self, logs):
        if not logs:
            return self._default_summary()

        def _avg(key):
            return float(sum(item[key] for item in logs) / len(logs))

        return {
            "points_for": _avg("points_for"),
            "points_allowed": _avg("points_allowed"),
            "pace": _avg("pace"),
            "efg": _avg("efg"),
            "ft_rate": _avg("ft_rate"),
            "to_rate": _avg("to_rate"),
        }

    def _rest_days(self, logs, game_date):
        if not logs or game_date is None:
            return 2.0
        try:
            current = pd.Timestamp(game_date)
        except (TypeError, ValueError):
            return 2.0
        dated_logs = [item for item in logs if item.get("date") is not None]
        if not dated_logs:
            return 2.0
        last_date = max(pd.Timestamp(item["date"]) for item in dated_logs)
        return max(0.0, float((current - last_date).days))

    def _feature_vector(self, home_team, away_team, team_logs, game_date=None):
        home_logs = team_logs.get(home_team, [])
        away_logs = team_logs.get(away_team, [])
        home_recent = self._aggregate(home_logs[-self.feature_window :])
        away_recent = self._aggregate(away_logs[-self.feature_window :])
        home_home = self._aggregate([item for item in home_logs if item.get("venue") == "home"][-self.feature_window :])
        away_away = self._aggregate([item for item in away_logs if item.get("venue") == "away"][-self.feature_window :])

        combined_recent_total = (
            home_recent["points_for"] +
            away_recent["points_for"] +
            away_recent["points_allowed"] +
            home_recent["points_allowed"]
        ) / 2.0
        combined_rest = (self._rest_days(home_logs, game_date) + self._rest_days(away_logs, game_date)) / 4.0

        return np.array([
            home_recent["points_for"],
            away_recent["points_for"],
            home_recent["points_allowed"],
            away_recent["points_allowed"],
            home_recent["pace"],
            away_recent["pace"],
            home_recent["efg"],
            away_recent["efg"],
            home_recent["ft_rate"],
            away_recent["ft_rate"],
            home_recent["to_rate"],
            away_recent["to_rate"],
            home_home["points_for"],
            away_away["points_for"],
            combined_recent_total,
            combined_rest,
        ])

    @staticmethod
    def _boxscore_log(team_bs, opp_bs, venue, game_date):
        team_poss = max(float(team_bs["possessions"]), 1.0)
        opp_poss = max(float(opp_bs["possessions"]), 1.0)
        fga = max(float(team_bs["fga"]), 1.0)
        return {
            "date": game_date,
            "venue": venue,
            "points_for": float(team_bs["pts"]),
            "points_allowed": float(opp_bs["pts"]),
            "pace": (team_poss + opp_poss) / 2.0,
            "efg": (float(team_bs["fgm"]) + (0.5 * float(team_bs["fg3m"]))) / fga,
            "ft_rate": float(team_bs["fta"]) / fga,
            "to_rate": float(team_bs["to"]) / team_poss,
        }

    def _fit(self, box_scores, games):
        required_box = {"game_id", "team", "pts", "possessions"}
        required_games = {"game_id", "date", "home_team", "away_team", "home_goals", "away_goals"}
        if (
            box_scores is None
            or games is None
            or box_scores.empty
            or games.empty
            or not required_box.issubset(set(box_scores.columns))
            or not required_games.issubset(set(games.columns))
        ):
            return

        bs_lookup = {}
        for _, row in box_scores.iterrows():
            bs_lookup[(row["game_id"], row["team"])] = row

        df = games.sort_values("date").reset_index(drop=True)
        team_logs = {
            team: []
            for team in sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        }
        X = []
        y = []

        for _, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_bs = bs_lookup.get((row["game_id"], home))
            away_bs = bs_lookup.get((row["game_id"], away))
            if home_bs is None or away_bs is None:
                continue

            X.append(self._feature_vector(home, away, team_logs, game_date=row["date"]))
            y.append(float(row["home_goals"]) + float(row["away_goals"]))

            team_logs.setdefault(home, []).append(self._boxscore_log(home_bs, away_bs, "home", row["date"]))
            team_logs.setdefault(away, []).append(self._boxscore_log(away_bs, home_bs, "away", row["date"]))

        self.team_logs = team_logs

        if len(X) < self.min_games:
            return

        self.model = LinearRegression()
        self.model.fit(np.array(X), np.array(y))
        preds = self.model.predict(np.array(X))
        residuals = np.array(y) - preds
        if len(residuals) >= 5:
            self.residual_std = max(7.5, float(np.std(residuals)))

    def predict_total(self, fixture: dict) -> float:
        if self.model is None:
            return 224.0
        features = self._feature_vector(
            fixture["home_team"],
            fixture["away_team"],
            self.team_logs,
            game_date=fixture.get("date"),
        )
        total = float(self.model.predict(np.array([features]))[0])
        return max(180.0, min(270.0, total))

    def predict_market(self, fixture: dict, total_line: float) -> dict[str, float]:
        expected_total = self.predict_total(fixture)
        sigma = max(7.5, float(self.residual_std or self.default_stddev))
        over_prob = float(1.0 - norm.cdf(total_line, loc=expected_total, scale=sigma))
        over_prob = max(0.01, min(0.99, over_prob))
        return {
            "expected_total": expected_total,
            "stddev": sigma,
            "over": over_prob,
            "under": 1.0 - over_prob,
        }


def nba_totals_predict(model, fixture: dict, total_line: float) -> dict[str, float]:
    """Predict NBA over/under probabilities and expected total."""
    return model.predict_market(fixture, total_line)


# ---------------------------------------------------------------------------
# Adjusted Efficiency (KenPom-style)
# ---------------------------------------------------------------------------

class AdjustedEfficiency:
    """KenPom-style adjusted efficiency model for college basketball.

    Computes adjusted offensive/defensive efficiency and tempo per team,
    iteratively correcting for opponent strength.

    Parameters
    ----------
    box_scores : pd.DataFrame
        Columns: game_id, team, date, pts, fgm, fga, fg3m, fg3a, ftm, fta,
        orb, drb, to, possessions.
    games : pd.DataFrame
        Columns: game_id, home_team, away_team, home_goals, away_goals.
    iterations : int
        Number of opponent-adjustment iterations.
    """

    def __init__(self, box_scores, games, iterations=10):
        self.off_efficiency = {}
        self.def_efficiency = {}
        self.tempo = {}
        self._fit(box_scores, games, iterations)

    def _fit(self, box_scores, games, iterations):
        # Build opponent mapping from games: for each (game_id, team) -> opponent
        opponents_in_game = {}
        for _, g in games.iterrows():
            gid = g["game_id"]
            opponents_in_game[(gid, g["home_team"])] = g["away_team"]
            opponents_in_game[(gid, g["away_team"])] = g["home_team"]

        # Compute weights based on recency
        weights = _compute_weights(box_scores)

        # Collect per-team raw stats
        teams = box_scores["team"].unique()
        team_total_pts = {t: 0.0 for t in teams}
        team_total_poss = {t: 0.0 for t in teams}
        team_total_pts_allowed = {t: 0.0 for t in teams}
        team_total_poss_against = {t: 0.0 for t in teams}
        team_weight_sum = {t: 0.0 for t in teams}
        team_opponents = {t: [] for t in teams}

        # Index box_scores by (game_id, team) for quick lookup
        bs_lookup = {}
        for idx, row in box_scores.iterrows():
            bs_lookup[(row["game_id"], row["team"])] = (row, weights[box_scores.index.get_loc(idx)])

        for idx, row in box_scores.iterrows():
            team = row["team"]
            gid = row["game_id"]
            poss = row["possessions"]
            w = weights[box_scores.index.get_loc(idx)]

            team_total_pts[team] += row["pts"] * w
            team_total_poss[team] += poss * w
            team_weight_sum[team] += w

            opp = opponents_in_game.get((gid, team))
            if opp is not None:
                team_opponents[team].append(opp)
                lookup_res = bs_lookup.get((gid, opp))
                if lookup_res is not None:
                    opp_row, _ = lookup_res
                    team_total_pts_allowed[team] += opp_row["pts"] * w
                    team_total_poss_against[team] += opp_row["possessions"] * w

        # Raw efficiencies (points per 100 possessions)
        raw_off = {}
        raw_def = {}
        raw_tempo = {}
        for t in teams:
            if team_total_poss[t] > 0:
                raw_off[t] = (team_total_pts[t] / team_total_poss[t]) * 100.0
            else:
                raw_off[t] = 100.0
            if team_total_poss_against[t] > 0:
                raw_def[t] = (team_total_pts_allowed[t] / team_total_poss_against[t]) * 100.0
            else:
                raw_def[t] = 100.0
            if team_weight_sum[t] > 0:
                raw_tempo[t] = team_total_poss[t] / team_weight_sum[t]
            else:
                raw_tempo[t] = 68.0

        # Iterative opponent adjustment
        adj_off = dict(raw_off)
        adj_def = dict(raw_def)

        for _ in range(iterations):
            league_avg_off = np.mean(list(adj_off.values()))
            league_avg_def = np.mean(list(adj_def.values()))

            new_off = {}
            new_def = {}
            for t in teams:
                opps = team_opponents[t]
                if opps:
                    avg_def_of_opps = np.mean([adj_def[o] for o in opps])
                    avg_off_of_opps = np.mean([adj_off[o] for o in opps])
                else:
                    avg_def_of_opps = league_avg_def
                    avg_off_of_opps = league_avg_off

                if avg_def_of_opps > 0:
                    new_off[t] = raw_off[t] * (league_avg_def / avg_def_of_opps)
                else:
                    new_off[t] = raw_off[t]
                if avg_off_of_opps > 0:
                    new_def[t] = raw_def[t] * (league_avg_off / avg_off_of_opps)
                else:
                    new_def[t] = raw_def[t]

            adj_off = new_off
            adj_def = new_def

        self.off_efficiency = adj_off
        self.def_efficiency = adj_def
        self.tempo = raw_tempo


def efficiency_predict(model, home_team, away_team, home_bonus=3.5, sigma=11.0):
    """Predict 2-way win probabilities from an AdjustedEfficiency model.

    Parameters
    ----------
    model : AdjustedEfficiency
        Fitted model.
    home_team, away_team : str
        Team names.
    home_bonus : float
        Points added to home team's expected score.
    sigma : float
        Logistic spread parameter for converting point spread to probability.

    Returns
    -------
    dict
        ``{"home": float, "away": float}``
    """
    league_avg = np.mean(list(model.off_efficiency.values()))
    if league_avg == 0:
        league_avg = 100.0

    home_off = model.off_efficiency.get(home_team, league_avg)
    away_off = model.off_efficiency.get(away_team, league_avg)
    home_def = model.def_efficiency.get(home_team, league_avg)
    away_def = model.def_efficiency.get(away_team, league_avg)
    home_tempo = model.tempo.get(home_team, 68.0)
    away_tempo = model.tempo.get(away_team, 68.0)

    expected_tempo = (home_tempo + away_tempo) / 2.0
    pace_factor = expected_tempo / 100.0

    home_pts = (home_off * away_def / league_avg) * pace_factor + home_bonus
    away_pts = (away_off * home_def / league_avg) * pace_factor

    spread = home_pts - away_pts
    home_prob = 1.0 / (1.0 + math.exp(-spread / sigma))

    return {"home": home_prob, "away": 1.0 - home_prob}


# ---------------------------------------------------------------------------
# Four Factors Logistic Regression
# ---------------------------------------------------------------------------

class FourFactorsModel:
    """Four Factors logistic regression model for college basketball.

    Computes Dean Oliver's four factors (offensive and defensive) per team
    as season averages, then trains a logistic regression on historical games.

    Parameters
    ----------
    box_scores : pd.DataFrame
        Columns: game_id, team, date, pts, fgm, fga, fg3m, fg3a, ftm, fta,
        orb, drb, to, possessions.
    games : pd.DataFrame
        Columns: game_id, home_team, away_team, home_goals, away_goals.
    """

    def __init__(self, box_scores, games):
        self.team_stats = {}
        self.model = None
        self._fit(box_scores, games)

    def _fit(self, box_scores, games):
        # Build opponent mapping
        opponents_in_game = {}
        for _, g in games.iterrows():
            gid = g["game_id"]
            opponents_in_game[(gid, g["home_team"])] = g["away_team"]
            opponents_in_game[(gid, g["away_team"])] = g["home_team"]

        # Compute weights based on recency
        weights = _compute_weights(box_scores)

        # Index box_scores by (game_id, team)
        bs_lookup = {}
        for idx, row in box_scores.iterrows():
            bs_lookup[(row["game_id"], row["team"])] = (row, weights[box_scores.index.get_loc(idx)])

        # Accumulate per-team stats
        teams = box_scores["team"].unique()
        accum = {t: {
            "fgm": 0.0, "fga": 0.0, "fg3m": 0.0, "fta": 0.0, "orb": 0.0, "to": 0.0,
            "possessions": 0.0,
            "opp_fgm": 0.0, "opp_fga": 0.0, "opp_fg3m": 0.0, "opp_fta": 0.0,
            "opp_orb": 0.0, "opp_drb": 0.0, "opp_to": 0.0, "opp_possessions": 0.0,
            "drb": 0.0, "weight_sum": 0.0,
        } for t in teams}

        for idx, row in box_scores.iterrows():
            team = row["team"]
            gid = row["game_id"]
            w = weights[box_scores.index.get_loc(idx)]
            a = accum[team]
            a["fgm"] += row["fgm"] * w
            a["fga"] += row["fga"] * w
            a["fg3m"] += row["fg3m"] * w
            a["fta"] += row["fta"] * w
            a["orb"] += row["orb"] * w
            a["drb"] += row["drb"] * w
            a["to"] += row["to"] * w
            a["possessions"] += row["possessions"] * w
            a["weight_sum"] += w

            opp = opponents_in_game.get((gid, team))
            if opp is not None:
                lookup_res = bs_lookup.get((gid, opp))
                if lookup_res is not None:
                    opp_row, _ = lookup_res
                    a["opp_fgm"] += opp_row["fgm"] * w
                    a["opp_fga"] += opp_row["fga"] * w
                    a["opp_fg3m"] += opp_row["fg3m"] * w
                    a["opp_fta"] += opp_row["fta"] * w
                    a["opp_orb"] += opp_row["orb"] * w
                    a["opp_drb"] += opp_row["drb"] * w
                    a["opp_to"] += opp_row["to"] * w
                    a["opp_possessions"] += opp_row["possessions"] * w

        # Compute four factors per team
        for t in teams:
            a = accum[t]
            fga = max(a["fga"], 1)
            poss = max(a["possessions"], 1)
            opp_fga = max(a["opp_fga"], 1)
            opp_poss = max(a["opp_possessions"], 1)

            off_efg = (a["fgm"] + 0.5 * a["fg3m"]) / fga
            off_to_rate = a["to"] / poss
            # ORB% = team ORB / (team ORB + opponent DRB)
            orb_denom = a["orb"] + a["opp_drb"]
            off_orb_pct = a["orb"] / max(orb_denom, 1)
            off_ft_rate = a["fta"] / fga

            def_efg = (a["opp_fgm"] + 0.5 * a["opp_fg3m"]) / opp_fga
            def_to_rate = a["opp_to"] / opp_poss
            # Opponent ORB% = opponent ORB / (opponent ORB + team DRB)
            def_orb_denom = a["opp_orb"] + a["drb"]
            def_orb_pct = a["opp_orb"] / max(def_orb_denom, 1)
            def_ft_rate = a["opp_fta"] / opp_fga

            self.team_stats[t] = {
                "off_efg": off_efg,
                "off_to_rate": off_to_rate,
                "off_orb_pct": off_orb_pct,
                "off_ft_rate": off_ft_rate,
                "def_efg": def_efg,
                "def_to_rate": def_to_rate,
                "def_orb_pct": def_orb_pct,
                "def_ft_rate": def_ft_rate,
            }

        # Train logistic regression on historical games
        feature_keys = [
            "off_efg", "off_to_rate", "off_orb_pct", "off_ft_rate",
            "def_efg", "def_to_rate", "def_orb_pct", "def_ft_rate",
        ]
        X = []
        y = []
        for _, g in games.iterrows():
            ht = g["home_team"]
            at = g["away_team"]
            if ht not in self.team_stats or at not in self.team_stats:
                continue
            home_feats = [self.team_stats[ht][k] for k in feature_keys]
            away_feats = [self.team_stats[at][k] for k in feature_keys]
            X.append(home_feats + away_feats)
            y.append(1 if g["home_goals"] > g["away_goals"] else 0)

        if len(X) >= 5 and len(set(y)) >= 2:
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(np.array(X), np.array(y))


def four_factors_predict(model, home_team, away_team):
    """Predict 2-way win probabilities from a FourFactorsModel.

    Parameters
    ----------
    model : FourFactorsModel
        Fitted model.
    home_team, away_team : str
        Team names.

    Returns
    -------
    dict
        ``{"home": float, "away": float}``
    """
    if model.model is None:
        return {"home": 0.5, "away": 0.5}
    if home_team not in model.team_stats or away_team not in model.team_stats:
        return {"home": 0.5, "away": 0.5}

    feature_keys = [
        "off_efg", "off_to_rate", "off_orb_pct", "off_ft_rate",
        "def_efg", "def_to_rate", "def_orb_pct", "def_ft_rate",
    ]
    home_feats = [model.team_stats[home_team][k] for k in feature_keys]
    away_feats = [model.team_stats[away_team][k] for k in feature_keys]
    X = np.array([home_feats + away_feats])
    proba = model.model.predict_proba(X)[0]

    # proba[1] = P(home wins), proba[0] = P(away wins)
    # Ensure correct class ordering
    classes = list(model.model.classes_)
    home_idx = classes.index(1)
    away_idx = classes.index(0)

    return {"home": float(proba[home_idx]), "away": float(proba[away_idx])}
