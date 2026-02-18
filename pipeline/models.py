"""Dixon-Coles prediction model, Elo rating system, Adjusted Efficiency,
and Four Factors logistic regression for SLOP LOCKS."""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

from pipeline.config import (
    CONGESTION_PENALTY,
    ELO_HOME_ADVANTAGE,
    ELO_K_FACTOR,
    FORM_WEIGHT_MULTIPLIER,
    FORM_WINDOW,
    MAX_GOALS,
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

    def process_season(self, matches):
        """Process a DataFrame of matches in chronological order.

        Parameters
        ----------
        matches : pd.DataFrame
            Must contain: home_team, away_team, home_goals, away_goals, date.
        """
        df = matches.sort_values("date").reset_index(drop=True)
        for _, row in df.iterrows():
            self.update(
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
            )


def elo_predict(elo, home_team, away_team, outcomes=None,
                home_rest_adj=0.0, away_rest_adj=0.0):
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

    Returns
    -------
    dict
        Probability for each outcome.
    """
    if outcomes is None:
        outcomes = ["home", "draw", "away"]

    r_home = elo.get_rating(home_team) + elo.home_advantage + home_rest_adj
    r_away = elo.get_rating(away_team) + away_rest_adj

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

        # Collect per-team raw stats
        teams = box_scores["team"].unique()
        team_total_pts = {t: 0.0 for t in teams}
        team_total_poss = {t: 0.0 for t in teams}
        team_total_pts_allowed = {t: 0.0 for t in teams}
        team_total_poss_against = {t: 0.0 for t in teams}
        team_game_count = {t: 0 for t in teams}
        team_opponents = {t: [] for t in teams}

        # Index box_scores by (game_id, team) for quick lookup
        bs_lookup = {}
        for _, row in box_scores.iterrows():
            bs_lookup[(row["game_id"], row["team"])] = row

        for _, row in box_scores.iterrows():
            team = row["team"]
            gid = row["game_id"]
            poss = row["possessions"]

            team_total_pts[team] += row["pts"]
            team_total_poss[team] += poss
            team_game_count[team] += 1

            opp = opponents_in_game.get((gid, team))
            if opp is not None:
                team_opponents[team].append(opp)
                opp_row = bs_lookup.get((gid, opp))
                if opp_row is not None:
                    team_total_pts_allowed[team] += opp_row["pts"]
                    team_total_poss_against[team] += opp_row["possessions"]

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
            if team_game_count[t] > 0:
                raw_tempo[t] = team_total_poss[t] / team_game_count[t]
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

        # Index box_scores by (game_id, team)
        bs_lookup = {}
        for _, row in box_scores.iterrows():
            bs_lookup[(row["game_id"], row["team"])] = row

        # Accumulate per-team stats
        teams = box_scores["team"].unique()
        accum = {t: {
            "fgm": 0, "fga": 0, "fg3m": 0, "fta": 0, "orb": 0, "to": 0,
            "possessions": 0,
            "opp_fgm": 0, "opp_fga": 0, "opp_fg3m": 0, "opp_fta": 0,
            "opp_orb": 0, "opp_drb": 0, "opp_to": 0, "opp_possessions": 0,
            "drb": 0, "game_count": 0,
        } for t in teams}

        for _, row in box_scores.iterrows():
            team = row["team"]
            gid = row["game_id"]
            a = accum[team]
            a["fgm"] += row["fgm"]
            a["fga"] += row["fga"]
            a["fg3m"] += row["fg3m"]
            a["fta"] += row["fta"]
            a["orb"] += row["orb"]
            a["drb"] += row["drb"]
            a["to"] += row["to"]
            a["possessions"] += row["possessions"]
            a["game_count"] += 1

            opp = opponents_in_game.get((gid, team))
            if opp is not None:
                opp_row = bs_lookup.get((gid, opp))
                if opp_row is not None:
                    a["opp_fgm"] += opp_row["fgm"]
                    a["opp_fga"] += opp_row["fga"]
                    a["opp_fg3m"] += opp_row["fg3m"]
                    a["opp_fta"] += opp_row["fta"]
                    a["opp_orb"] += opp_row["orb"]
                    a["opp_drb"] += opp_row["drb"]
                    a["opp_to"] += opp_row["to"]
                    a["opp_possessions"] += opp_row["possessions"]

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
