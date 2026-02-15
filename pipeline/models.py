"""Dixon-Coles prediction model and Elo rating system for SLOP LOCKS."""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

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
    """

    def __init__(self, teams, initial_rating=1500):
        self.ratings = {t: float(initial_rating) for t in teams}

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

        # Expected scores — home team gets ELO_HOME_ADVANTAGE boost
        e_home = self.expected_score(r_home + ELO_HOME_ADVANTAGE, r_away)
        e_away = 1.0 - e_home

        # Goal-difference multiplier
        goal_diff = abs(hg - ag)
        if goal_diff <= 1:
            gd_mult = 1.0
        elif goal_diff == 2:
            gd_mult = 1.5
        else:
            gd_mult = (11.0 + goal_diff) / 8.0

        k = ELO_K_FACTOR * gd_mult

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


def elo_predict(elo, home_team, away_team):
    """Convert Elo ratings into match-outcome probabilities.

    The draw probability is modelled as a function of rating closeness,
    calibrated to produce roughly 25% draws for typical EPL rating gaps.

    Parameters
    ----------
    elo : EloRatings
        Fitted Elo system.
    home_team, away_team : str
        Team names.

    Returns
    -------
    dict
        Keys: ``home``, ``draw``, ``away`` — each a float probability.
    """
    r_home = elo.get_rating(home_team) + ELO_HOME_ADVANTAGE
    r_away = elo.get_rating(away_team)

    diff = r_home - r_away

    # Expected score for home (logistic)
    e_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    # Draw probability: peaks when teams are close, base ~25%
    # Using a Gaussian-like function of the rating difference
    draw_base = 0.25
    draw_prob = draw_base * math.exp(-(diff ** 2) / (2 * 300 ** 2))
    # Ensure draw_prob stays in a reasonable range
    draw_prob = max(0.10, min(0.35, draw_prob))

    # Remaining probability split by expected score
    remaining = 1.0 - draw_prob
    home_prob = remaining * e_home
    away_prob = remaining * (1.0 - e_home)

    return {"home": home_prob, "draw": draw_prob, "away": away_prob}
