from datetime import date, datetime, timedelta
from pathlib import Path
import math
import pandas as pd
import nflreadpy as nfl

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


app = FastAPI(title="NFL Receiving Yard Projections")


# ---------------------------------------------------------
# TEAM INFORMATION
# ---------------------------------------------------------

NFL_TEAMS = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

SPORTSBOOKS = [
    "DraftKings",
    "FanDuel",
    "BetMGM",
    "Caesars",
    "ESPN BET",
    "Other",
]


# ---------------------------------------------------------
# CACHING
# ---------------------------------------------------------

stats_cache = {
    "data": None,
    "season": None,
    "expires": None,
}

roster_cache = {
    "data": None,
    "season": None,
    "expires": None,
}

STATS_CACHE_MINUTES = 60
ROSTER_CACHE_HOURS = 12


# ---------------------------------------------------------
# CURRENT NFL SEASON
# ---------------------------------------------------------

def get_current_nfl_season():
    """
    NFL regular seasons generally begin in September.
    January through July games belong to the previous season.
    """

    today = date.today()

    if today.month >= 8:
        return today.year

    return today.year - 1


# ---------------------------------------------------------
# GENERAL DATA HELPERS
# ---------------------------------------------------------

def find_column(df, possible_names):
    """Find a column without depending on capitalization."""

    if df is None or df.empty:
        return None

    lookup = {
        str(column).lower(): column
        for column in df.columns
    }

    for name in possible_names:
        found = lookup.get(name.lower())

        if found is not None:
            return found

    return None


def numeric_series(df, column):
    """Return a numeric column, replacing missing values with zero."""

    if column is None or column not in df.columns:
        return pd.Series(0.0, index=df.index)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0.0)


def safe_float(value, default=0.0):
    try:
        result = float(value)

        if math.isnan(result) or math.isinf(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def find_name_column(df):
    return find_column(
        df,
        [
            "player_display_name",
            "player_name",
            "full_name",
            "display_name",
            "player"
        ]
    )


def get_team_column(df):
    return find_column(
        df,
        [
            "recent_team",
            "team",
            "team_abbr",
            "posteam"
        ]
    )


def get_opponent_column(df):
    return find_column(
        df,
        [
            "opponent_team",
            "defteam",
            "opponent"
        ]
    )


def get_position_column(df):
    return find_column(
        df,
        [
            "position",
            "player_position"
        ]
    )


def get_week_column(df):
    return find_column(
        df,
        [
            "week"
        ]
    )


# ---------------------------------------------------------
# WEIGHTED AVERAGE
# ---------------------------------------------------------

def weighted_average(values):
    """
    Give more importance to recent performance.

    50% = full-season average
    30% = last five games
    20% = last three games
    """

    values = [
        safe_float(value)
        for value in values
    ]

    values = [
        value for value in values
        if value >= 0
    ]

    if not values:
        return 0.0

    full_average = sum(values) / len(values)

    last_five = values[-5:]
    last_three = values[-3:]

    five_average = sum(last_five) / len(last_five)
    three_average = sum(last_three) / len(last_three)

    return (
        full_average * 0.50
        + five_average * 0.30
        + three_average * 0.20
    )


# ---------------------------------------------------------
# LOAD NFL PLAYER STATS
# ---------------------------------------------------------

def load_stats():
    season = get_current_nfl_season()
    now = datetime.now()

    if (
        stats_cache["data"] is not None
        and stats_cache["season"] == season
        and stats_cache["expires"] is not None
        and now < stats_cache["expires"]
    ):
        return stats_cache["data"]

    try:
        df = nfl.load_player_stats(
            seasons=[season],
            summary_level="week"
        ).to_pandas()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to load NFL player statistics: {str(exc)}"
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No NFL player statistics are currently available for {season}."
        )

    # Keep only the requested season if the column exists.
    season_column = find_column(df, ["season"])

    if season_column:
        df = df[
            pd.to_numeric(
                df[season_column],
                errors="coerce"
            ) == season
        ]

    # Keep regular-season games if season_type is available.
    season_type_column = find_column(
        df,
        ["season_type"]
    )

    if season_type_column:
        df = df[
            df[season_type_column]
            .astype(str)
            .str.upper()
            .isin(["REG", "REGULAR"])
        ]

    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="No regular-season statistics are available yet."
        )

    stats_cache["data"] = df
    stats_cache["season"] = season
    stats_cache["expires"] = now + timedelta(
        minutes=STATS_CACHE_MINUTES
    )

    return df


# ---------------------------------------------------------
# LOAD NFL ROSTERS
# ---------------------------------------------------------

def load_rosters():
    season = get_current_nfl_season()
    now = datetime.now()

    if (
        roster_cache["data"] is not None
        and roster_cache["season"] == season
        and roster_cache["expires"] is not None
        and now < roster_cache["expires"]
    ):
        return roster_cache["data"]

    try:
        df = nfl.load_rosters(
            seasons=[season]
        ).to_pandas()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to load NFL rosters: {str(exc)}"
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail="NFL roster information is not currently available."
        )

    roster_cache["data"] = df
    roster_cache["season"] = season
    roster_cache["expires"] = now + timedelta(
        hours=ROSTER_CACHE_HOURS
    )

    return df


# ---------------------------------------------------------
# PLAYER STAT FILTERING
# ---------------------------------------------------------

def get_player_rows(df, player_name, team=None):
    name_column = find_name_column(df)
    team_column = get_team_column(df)
    week_column = get_week_column(df)

    if name_column is None:
        raise HTTPException(
            status_code=500,
            detail="Could not identify the player-name column in NFL statistics."
        )

    rows = df[
        df[name_column]
        .astype(str)
        .str.casefold()
        == str(player_name).casefold()
    ].copy()

    if team and team_column:
        rows = rows[
            rows[team_column]
            .astype(str)
            .str.upper()
            == team.upper()
        ]

    if week_column:
        rows["_sort_week"] = pd.to_numeric(
            rows[week_column],
            errors="coerce"
        )

        rows = rows.sort_values(
            "_sort_week"
        )

    return rows


# ---------------------------------------------------------
# API: HOME PAGE
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    html_file = Path("static/index.html")

    if not html_file.exists():
        return """
        <h2>NFL Projection App</h2>
        <p>The frontend file static/index.html was not found.</p>
        """

    return html_file.read_text(encoding="utf-8")


# ---------------------------------------------------------
# API: TEAMS
# ---------------------------------------------------------

@app.get("/api/teams")
def get_teams():
    return [
        {
            "code": code,
            "name": name
        }
        for code, name in NFL_TEAMS.items()
    ]


# ---------------------------------------------------------
# API: SPORTSBOOKS
# ---------------------------------------------------------

@app.get("/api/sportsbooks")
def get_sportsbooks():
    return SPORTSBOOKS


# ---------------------------------------------------------
# API: TEAM ROSTER
# ---------------------------------------------------------

@app.get("/api/roster")
def get_roster(team: str):
    team = team.upper().strip()

    if team not in NFL_TEAMS:
        raise HTTPException(
            status_code=400,
            detail="Invalid NFL team."
        )

    df = load_rosters()

    team_column = find_column(
        df,
        [
            "team",
            "team_abbr",
            "recent_team"
        ]
    )

    name_column = find_name_column(df)
    position_column = get_position_column(df)

    if not team_column or not name_column or not position_column:
        raise HTTPException(
            status_code=500,
            detail="Could not identify the required roster columns."
        )

    team_rows = df[
        df[team_column]
        .astype(str)
        .str.upper()
        == team
    ].copy()

    if team_rows.empty:
        return {
            "team": team,
            "quarterbacks": [],
            "receivers": []
        }

    # Exclude players who are not on the active roster when
    # a status field is available.
    status_column = find_column(
        team_rows,
        [
            "status",
            "roster_status"
        ]
    )

    if status_column:
        active_statuses = [
            "ACT",
            "ACTIVE",
            "RES",
            "PUP",
            "IR"
        ]

        team_rows = team_rows[
            team_rows[status_column]
            .astype(str)
            .str.upper()
            .isin(active_statuses)
        ]

    team_rows[name_column] = (
        team_rows[name_column].astype(str)
    )

    team_rows[position_column] = (
        team_rows[position_column]
        .astype(str)
        .str.upper()
    )

    quarterbacks = team_rows[
        team_rows[position_column] == "QB"
    ]

    receivers = team_rows[
        team_rows[position_column].isin(
            ["WR", "TE", "RB"]
        )
    ]

    def format_players(rows):
        players = []

        for _, row in rows.iterrows():
            players.append({
                "name": str(row[name_column]),
                "position": str(row[position_column]),
                "status": (
                    str(row[status_column])
                    if status_column
                    else ""
                )
            })

        return sorted(
            players,
            key=lambda player: player["name"]
        )

    return {
        "team": team,
        "quarterbacks": format_players(quarterbacks),
        "receivers": format_players(receivers)
    }


# ---------------------------------------------------------
# API: HEALTH CHECK
# ---------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "season": get_current_nfl_season()
    }


# ---------------------------------------------------------
# API: RECEIVING YARD PROJECTION
# ---------------------------------------------------------

@app.get("/api/project")
def project(
    offense_team: str,
    qb: str,
    receiver: str,
    defense: str,
    sportsbook: str = "Other",
    line: float = None
):
    offense_team = offense_team.upper().strip()
    defense = defense.upper().strip()

    if offense_team not in NFL_TEAMS:
        raise HTTPException(
            status_code=400,
            detail="Invalid offensive team."
        )

    if defense not in NFL_TEAMS:
        raise HTTPException(
            status_code=400,
            detail="Invalid defensive team."
        )

    if offense_team == defense:
        raise HTTPException(
            status_code=400,
            detail="The offense and defense teams cannot be the same."
        )

    df = load_stats()

    name_column = find_name_column(df)
    team_column = get_team_column(df)
    opponent_column = get_opponent_column(df)
    week_column = get_week_column(df)

    if not name_column or not team_column or not week_column:
        raise HTTPException(
            status_code=500,
            detail="Required player-stat columns could not be identified."
        )

    # -----------------------------------------------------
    # SELECTED QB STATS
    # -----------------------------------------------------

    qb_rows = get_player_rows(
        df,
        qb,
        offense_team
    )

    if qb_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No current-season statistics found for QB {qb}."
        )

    attempts_column = find_column(
        qb_rows,
        [
            "attempts",
            "passing_attempts"
        ]
    )

    completions_column = find_column(
        qb_rows,
        [
            "completions",
            "passing_completions"
        ]
    )

    if not attempts_column or not completions_column:
        raise HTTPException(
            status_code=500,
            detail="Could not identify QB passing attempts or completions."
        )

    qb_rows["_attempts"] = numeric_series(
        qb_rows,
        attempts_column
    )

    qb_rows["_completions"] = numeric_series(
        qb_rows,
        completions_column
    )

    qb_weekly = (
        qb_rows.groupby(week_column)
        .agg(
            attempts=("_attempts", "sum"),
            completions=("_completions", "sum")
        )
        .reset_index()
    )

    qb_attempts = weighted_average(
        qb_weekly["attempts"].tolist()
    )

    qb_completions = weighted_average(
        qb_weekly["completions"].tolist()
    )

    # -----------------------------------------------------
    # SELECTED RECEIVER STATS
    # -----------------------------------------------------

    wr_rows = get_player_rows(
        df,
        receiver,
        offense_team
    )

    if wr_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No current-season statistics found for receiver {receiver}."
        )

    targets_column = find_column(
        wr_rows,
        ["targets"]
    )

    receptions_column = find_column(
        wr_rows,
        ["receptions"]
    )

    yards_column = find_column(
        wr_rows,
        [
            "receiving_yards",
            "rec_yards"
        ]
    )

    if not targets_column or not receptions_column or not yards_column:
        raise HTTPException(
            status_code=500,
            detail="Could not identify receiver targets, receptions, or yards."
        )

    wr_rows["_targets"] = numeric_series(
        wr_rows,
        targets_column
    )

    wr_rows["_receptions"] = numeric_series(
        wr_rows,
        receptions_column
    )

    wr_rows["_yards"] = numeric_series(
        wr_rows,
        yards_column
    )

    receiver_weekly = (
        wr_rows.groupby(week_column)
        .agg(
            targets=("_targets", "sum"),
            receptions=("_receptions", "sum"),
            yards=("_yards", "sum")
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # TARGET SHARE
    # -----------------------------------------------------
    # This is an estimate using the selected QB's attempts
    # and the receiver's weekly targets in matching weeks.
    #
    # nflverse weekly player stats do not reliably attribute
    # every receiver target to an individual QB when a team
    # uses multiple quarterbacks in one game.

    matched = pd.merge(
        qb_weekly,
        receiver_weekly,
        on=week_column,
        how="inner"
    )

    matched = matched[
        matched["attempts"] > 0
    ].copy()

    if matched.empty:
        target_share_decimal = 0.0
    else:
        matched["weekly_target_share"] = (
            matched["targets"] / matched["attempts"]
        )

        target_share_decimal = weighted_average(
            matched["weekly_target_share"].tolist()
        )

    # Return percentage units, e.g. 24.5 means 24.5%.
    target_share_percentage = (
        target_share_decimal * 100
    )

    # -----------------------------------------------------
    # CATCH RATE
    # -----------------------------------------------------

    total_targets = wr_rows["_targets"].sum()
    total_receptions = wr_rows["_receptions"].sum()

    if total_targets > 0:
        catch_rate_decimal = (
            total_receptions / total_targets
        )
    else:
        catch_rate_decimal = 0.0

    # Return percentage units, e.g. 68.2 means 68.2%.
    catch_rate_percentage = (
        catch_rate_decimal * 100
    )

    # -----------------------------------------------------
    # RECENT RECEIVING YARDS
    # -----------------------------------------------------

    receiver_recent_yards_average = weighted_average(
        receiver_weekly["yards"].tolist()
    )

    # -----------------------------------------------------
    # DEFENSIVE PASSING STATS
    # -----------------------------------------------------

    defense_games_sampled = 0
    yards_per_completion = 0.0

    if opponent_column:
        defense_rows = df[
            df[opponent_column]
            .astype(str)
            .str.upper()
            == defense
        ].copy()

        position_column = get_position_column(
            defense_rows
        )

        if position_column:
            defense_rows = defense_rows[
                defense_rows[position_column]
                .astype(str)
                .str.upper()
                == "QB"
            ]

        passing_yards_column = find_column(
            defense_rows,
            [
                "passing_yards",
                "pass_yards"
            ]
        )

        defense_completions_column = find_column(
            defense_rows,
            [
                "completions",
                "passing_completions"
            ]
        )

        if (
            not defense_rows.empty
            and passing_yards_column
            and defense_completions_column
        ):
            defense_rows["_pass_yards"] = numeric_series(
                defense_rows,
                passing_yards_column
            )

            defense_rows["_pass_completions"] = numeric_series(
                defense_rows,
                defense_completions_column
            )

            defense_weekly = (
                defense_rows.groupby(week_column)
                .agg(
                    pass_yards=("_pass_yards", "sum"),
                    completions=("_pass_completions", "sum")
                )
                .reset_index()
            )

            defense_weekly = defense_weekly[
                defense_weekly["completions"] > 0
            ]

            defense_games_sampled = len(
                defense_weekly
            )

            if not defense_weekly.empty:
                defense_weekly["yards_per_completion"] = (
                    defense_weekly["pass_yards"]
                    / defense_weekly["completions"]
                )

                yards_per_completion = weighted_average(
                    defense_weekly[
                        "yards_per_completion"
                    ].tolist()
                )

    # Fallback if defensive data is not available.
    if yards_per_completion <= 0:
        yards_per_completion = 10.5

    # -----------------------------------------------------
    # PROJECTION CALCULATIONS
    # -----------------------------------------------------

    projected_targets = (
        qb_attempts * target_share_decimal
    )

    projected_receptions = (
        projected_targets * catch_rate_decimal
    )

    original_projection = (
        projected_targets * yards_per_completion
    )

    improved_projection = (
        projected_receptions * yards_per_completion
    )

    # -----------------------------------------------------
    # SPORTSBOOK LINE / EDGE
    # -----------------------------------------------------

    sportsbook = (
        sportsbook
        if sportsbook in SPORTSBOOKS
        else "Other"
    )

    if line is not None:
        line = safe_float(line, None)

    if line is not None:
        improved_edge = (
            improved_projection - line
        )

        if improved_edge > 0:
            lean = "OVER"
        elif improved_edge < 0:
            lean = "UNDER"
        else:
            lean = "NO EDGE"
    else:
        improved_edge = None
        lean = "NO LINE ENTERED"

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    return {
        "offense_team": offense_team,
        "quarterback": qb,
        "receiver": receiver,
        "defense": defense,
        "season": get_current_nfl_season(),

        "improved_projection": round(
            improved_projection, 2
        ),

        "original_projection": round(
            original_projection, 2
        ),

        "qb_attempts_weighted": round(
            qb_attempts, 2
        ),

        "qb_completions_weighted": round(
            qb_completions, 2
        ),

        "target_share": round(
            target_share_percentage, 2
        ),

        "catch_rate": round(
            catch_rate_percentage, 2
        ),

        "receiver_recent_yards_average": round(
            receiver_recent_yards_average, 2
        ),

        "projected_targets": round(
            projected_targets, 2
        ),

        "projected_receptions": round(
            projected_receptions, 2
        ),

        "yards_per_completion_allowed": round(
            yards_per_completion, 2
        ),

        "matched_games": int(len(matched)),

        "defense_games_sampled": int(
            defense_games_sampled
        ),

        "sportsbook": sportsbook,

        "line": line,

        "improved_edge": (
            round(improved_edge, 2)
            if improved_edge is not None
            else None
        ),

        "lean": lean
    }


# ---------------------------------------------------------
# STATIC FILES
# ---------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
