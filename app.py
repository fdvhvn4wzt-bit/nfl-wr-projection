from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import date
from functools import lru_cache
import pandas as pd
import nflreadpy as nfl

app = FastAPI(title="NFL Receiving Yard Projection App")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------
# AUTOMATIC NFL SEASON DETECTION
# --------------------------------------------------

def get_current_nfl_season():
    """
    NFL seasons begin in the fall and continue into January/February.
    August is used as the transition month for the upcoming season.
    """

    today = date.today()

    if today.month >= 8:
        return today.year
    else:
        return today.year - 1


# --------------------------------------------------
# NFL TEAMS
# --------------------------------------------------

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
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
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


# --------------------------------------------------
# LOAD CURRENT-SEASON STATISTICS ONLY
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_stats():

    season = get_current_nfl_season()

    try:
        df = nfl.load_player_stats(
            seasons=[season],
            summary_level="week"
        ).to_pandas()

        if df.empty:
            raise ValueError(
                f"No player statistics available for the {season} season yet."
            )

        df = df[df["season"] == season].copy()

        if "season_type" in df.columns:
            df = df[df["season_type"] == "REG"].copy()

        return df

    except Exception as e:
        raise RuntimeError(
            f"Unable to load {season} NFL statistics: {str(e)}"
        )


# --------------------------------------------------
# LOAD CURRENT-SEASON ROSTERS ONLY
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_rosters():

    season = get_current_nfl_season()

    try:
        rosters = nfl.load_rosters(
            seasons=[season]
        ).to_pandas()

        if rosters.empty:
            raise ValueError(
                f"No roster information available for {season}."
            )

        if "season" in rosters.columns:
            rosters = rosters[
                rosters["season"] == season
            ].copy()

        return rosters

    except Exception as e:
        raise RuntimeError(
            f"Unable to load {season} NFL rosters: {str(e)}"
        )


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def weighted_average(values):

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).dropna()

    if values.empty:
        return 0.0

    full_average = values.mean()
    last_five = values.tail(5).mean()
    last_three = values.tail(3).mean()

    return float(
        (full_average * 0.50)
        + (last_five * 0.30)
        + (last_three * 0.20)
    )


def find_name_column(df):

    for column in [
        "player_display_name",
        "player_name",
        "full_name"
    ]:
        if column in df.columns:
            return column

    raise ValueError("Could not find player name column.")


def get_player_rows(df, player_name, team=None):

    name_column = find_name_column(df)

    rows = df[
        df[name_column].astype(str).str.lower()
        == player_name.lower()
    ].copy()

    if team and "recent_team" in rows.columns:
        rows = rows[
            rows["recent_team"] == team
        ].copy()

    if "season" in rows.columns:
        rows = rows[
            rows["season"] == get_current_nfl_season()
        ].copy()

    if "week" in rows.columns:
        rows = rows.sort_values("week")

    return rows


def safe_number(value):

    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="index.html not found in static folder."
        )

    return index_file.read_text(encoding="utf-8")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "online",
        "season": get_current_nfl_season(),
        "message": "NFL projection app is running."
    }


# --------------------------------------------------
# GET NFL TEAMS
# --------------------------------------------------

@app.get("/api/teams")
def get_teams():

    return [
        {
            "code": code,
            "name": name
        }
        for code, name in NFL_TEAMS.items()
    ]


# --------------------------------------------------
# GET SPORTSBOOKS
# --------------------------------------------------

@app.get("/api/sportsbooks")
def get_sportsbooks():

    return SPORTSBOOKS


# --------------------------------------------------
# GET TEAM ROSTER
# --------------------------------------------------

@app.get("/api/roster")
def get_roster(team: str):

    team = team.upper()

    if team not in NFL_TEAMS:
        raise HTTPException(
            status_code=400,
            detail="Invalid NFL team."
        )

    try:
        rosters = load_rosters()

        team_column = (
            "team"
            if "team" in rosters.columns
            else "recent_team"
        )

        team_roster = rosters[
            rosters[team_column] == team
        ].copy()

        if team_roster.empty:
            raise HTTPException(
                status_code=404,
                detail="No roster information available for this team."
            )

        name_column = find_name_column(team_roster)

        quarterbacks = []
        receivers = []

        for _, player in team_roster.iterrows():

            name = str(player[name_column])
            position = str(player.get("position", ""))
            status = str(player.get("status", ""))

            player_info = {
                "name": name,
                "position": position,
                "status": status
            }

            if position == "QB":
                quarterbacks.append(player_info)

            elif position in ["WR", "TE", "RB"]:
                receivers.append(player_info)

        quarterbacks.sort(key=lambda x: x["name"])
        receivers.sort(key=lambda x: x["name"])

        return {
            "team": team,
            "quarterbacks": quarterbacks,
            "receivers": receivers
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# RECEIVING YARD PROJECTION
# --------------------------------------------------

@app.get("/api/project")
def project_receiving_yards(
    offense_team: str,
    qb: str,
    defense: str,
    receiver: str,
    line: float = None,
    sportsbook: str = None
):

    offense_team = offense_team.upper()
    defense = defense.upper()

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
            detail="Offense and defense cannot be the same team."
        )

    try:

        df = load_stats()

        # ------------------------------------------
        # GET QUARTERBACK STATISTICS
        # ------------------------------------------

        qb_rows = get_player_rows(
            df,
            qb,
            offense_team
        )

        if qb_rows.empty:
            raise HTTPException(
                status_code=404,
                detail="No current-season statistics found for this quarterback."
            )

        if "attempts" not in qb_rows.columns:
            raise ValueError("QB passing attempts are unavailable.")

        if "completions" not in qb_rows.columns:
            raise ValueError("QB completions are unavailable.")

        qb_attempts = weighted_average(
            qb_rows["attempts"]
        )

        qb_completions = weighted_average(
            qb_rows["completions"]
        )

        # ------------------------------------------
        # GET RECEIVER STATISTICS
        # ------------------------------------------

        wr_rows = get_player_rows(
            df,
            receiver,
            offense_team
        )

        if wr_rows.empty:
            raise HTTPException(
                status_code=404,
                detail="No current-season statistics found for this receiver."
            )

        required_columns = [
            "targets",
            "receptions",
            "receiving_yards"
        ]

        for column in required_columns:
            if column not in wr_rows.columns:
                raise ValueError(
                    f"Receiver statistics missing: {column}"
                )

        targets = weighted_average(
            wr_rows["targets"]
        )

        receptions = weighted_average(
            wr_rows["receptions"]
        )

        receiving_yards = weighted_average(
            wr_rows["receiving_yards"]
        )

        total_targets = safe_number(
            wr_rows["targets"].sum()
        )

        total_receptions = safe_number(
            wr_rows["receptions"].sum()
        )

        # ------------------------------------------
        # CALCULATE TRUE TEAM TARGET SHARE
        # ------------------------------------------

        team_rows = df[
            df["recent_team"] == offense_team
        ].copy()

        if "week" in team_rows.columns:
            team_target_totals = (
                team_rows.groupby("week")["targets"]
                .sum()
            )

            receiver_target_totals = (
                wr_rows.groupby("week")["targets"]
                .sum()
            )

            matching_weeks = receiver_target_totals.index.intersection(
                team_target_totals.index
            )

            if len(matching_weeks) > 0:

                weekly_shares = (
                    receiver_target_totals.loc[matching_weeks]
                    / team_target_totals.loc[matching_weeks].replace(0, pd.NA)
                ).dropna()

                target_share = (
                    weighted_average(weekly_shares) * 100
                    if not weekly_shares.empty
                    else 0.0
                )

            else:
                target_share = 0.0

        else:
            target_share = 0.0

        # ------------------------------------------
        # CALCULATE CATCH RATE
        # ------------------------------------------

        catch_rate = (
            total_receptions / total_targets
            if total_targets > 0
            else 0.0
        )

        # ------------------------------------------
        # GET DEFENSIVE PASSING STATISTICS
        # ------------------------------------------

        if "opponent_team" not in df.columns:
            raise ValueError(
                "Opponent team information is unavailable."
            )

        defense_rows = df[
            df["opponent_team"] == defense
        ].copy()

        if defense_rows.empty:
            raise HTTPException(
                status_code=404,
                detail="No current-season defensive statistics found."
            )

        if "passing_yards" not in defense_rows.columns:
            raise ValueError("Passing yards data unavailable.")

        if "completions" not in defense_rows.columns:
            raise ValueError("Defensive completions data unavailable.")

        defense_games = defense_rows.groupby("week").agg(
            passing_yards=("passing_yards", "sum"),
            completions=("completions", "sum")
        )

        defense_games = defense_games[
            defense_games["completions"] > 0
        ]

        if defense_games.empty:
            yards_per_completion = 10.5
        else:
            defense_games["yards_per_completion"] = (
                defense_games["passing_yards"]
                / defense_games["completions"]
            )

            yards_per_completion = weighted_average(
                defense_games["yards_per_completion"]
            )

        # ------------------------------------------
        # PROJECTION FORMULAS
        # ------------------------------------------

        target_share_decimal = target_share / 100

        projected_receptions = (
            qb_completions
            * target_share_decimal
            * catch_rate
        )

        # Original formula:
        # QB completions x target share x defensive YPC

        original_projection = (
            qb_completions
            * target_share_decimal
            * yards_per_completion
        )

        # Improved formula:
        # QB completions x target share x catch rate x defensive YPC

        improved_projection = (
            projected_receptions
            * yards_per_completion
        )

        # ------------------------------------------
        # SPORTSBOOK COMPARISON
        # ------------------------------------------

        original_edge = None
        improved_edge = None
        lean = None

        if line is not None:

            original_edge = round(
                original_projection - line,
                2
            )

            improved_edge = round(
                improved_projection - line,
                2
            )

            if improved_projection > line:
                lean = "OVER"

            elif improved_projection < line:
                lean = "UNDER"

            else:
                lean = "PASS"

        # ------------------------------------------
        # RETURN PROJECTION
        # ------------------------------------------

        return {
            "season": get_current_nfl_season(),

            "offense_team": offense_team,
            "quarterback": qb,
            "defense": defense,
            "receiver": receiver,

            "original_projection": round(
                original_projection, 2
            ),

            "improved_projection": round(
                improved_projection, 2
            ),

            "projected_receptions": round(
                projected_receptions, 2
            ),

            "target_share": round(
                target_share, 2
            ),

            "catch_rate": round(
                catch_rate * 100, 2
            ),

            "yards_per_completion_allowed": round(
                yards_per_completion, 2
            ),

            "qb_attempts_weighted": round(
                qb_attempts, 2
            ),

            "qb_completions_weighted": round(
                qb_completions, 2
            ),

            "receiver_recent_yards_average": round(
                receiving_yards, 2
            ),

            "matched_games": len(wr_rows),

            "defense_games_sampled": len(
                defense_games
            ),

            "sportsbook": sportsbook,

            "line": line,

            "original_edge": original_edge,

            "improved_edge": improved_edge,

            "lean": lean
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
