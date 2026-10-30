from datetime import date, datetime, timedelta
from pathlib import Path
import math
import pandas as pd
import nflreadpy as nfl

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="NFL Player Prop Projections")

NFL_TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos",
    "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots",
    "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

SPORTSBOOKS = [
    "DraftKings", "FanDuel", "BetMGM",
    "Caesars", "ESPN BET", "Other"
]

stats_cache = {"data": None, "season": None, "expires": None}
roster_cache = {"data": None, "season": None, "expires": None}

STATS_CACHE_MINUTES = 60
ROSTER_CACHE_HOURS = 12


def get_current_nfl_season():
    today = date.today()
    return today.year if today.month >= 8 else today.year - 1


def find_column(df, possible_names):
    if df is None or df.empty:
        return None

    lookup = {str(c).lower(): c for c in df.columns}

    for name in possible_names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def numeric_series(df, column):
    if column is None or column not in df.columns:
        return pd.Series(0.0, index=df.index)

    return pd.to_numeric(
        df[column], errors="coerce"
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
    return find_column(df, [
        "player_display_name", "player_name",
        "full_name", "display_name", "player"
    ])


def get_team_column(df):
    return find_column(df, [
        "recent_team", "team", "team_abbr", "posteam"
    ])


def get_opponent_column(df):
    return find_column(df, [
        "opponent_team", "defteam", "opponent"
    ])


def get_position_column(df):
    return find_column(df, ["position", "player_position"])


def get_week_column(df):
    return find_column(df, ["week"])


def weighted_average(values):
    values = [
        safe_float(v)
        for v in values
        if safe_float(v) >= 0
    ]

    if not values:
        return 0.0

    full = sum(values) / len(values)
    last_five = values[-5:]
    last_three = values[-3:]

    five = sum(last_five) / len(last_five)
    three = sum(last_three) / len(last_three)

    return full * 0.50 + five * 0.30 + three * 0.20


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
            detail=f"Unable to load NFL player statistics: {exc}"
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"No NFL player statistics are currently available for {season}."
        )

    season_col = find_column(df, ["season"])

    if season_col:
        df = df[
            pd.to_numeric(df[season_col], errors="coerce") == season
        ]

    type_col = find_column(df, ["season_type"])

    if type_col:
        df = df[
            df[type_col].astype(str).str.upper().isin(["REG", "REGULAR"])
        ]

    if df.empty:
        raise HTTPException(
            status_code=503,
            detail="No regular-season statistics are available yet."
        )

    stats_cache.update({
        "data": df,
        "season": season,
        "expires": now + timedelta(minutes=STATS_CACHE_MINUTES)
    })

    return df


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
        df = nfl.load_rosters(seasons=[season]).to_pandas()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to load NFL rosters: {exc}"
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail="NFL roster information is not currently available."
        )

    roster_cache.update({
        "data": df,
        "season": season,
        "expires": now + timedelta(hours=ROSTER_CACHE_HOURS)
    })

    return df


def get_player_rows(df, player_name, team=None):
    name_col = find_name_column(df)
    team_col = get_team_column(df)
    week_col = get_week_column(df)

    if name_col is None:
        raise HTTPException(
            status_code=500,
            detail="Could not identify the player-name column."
        )

    rows = df[
        df[name_col].astype(str).str.casefold()
        == str(player_name).casefold()
    ].copy()

    if team and team_col:
        rows = rows[
            rows[team_col].astype(str).str.upper() == team.upper()
        ]

    if week_col:
        rows["_sort_week"] = pd.to_numeric(
            rows[week_col], errors="coerce"
        )
        rows = rows.sort_values("_sort_week")

    return rows


@app.get("/", response_class=HTMLResponse)
def home():
    html_file = Path("static/index.html")

    if not html_file.exists():
        return "<h2>Frontend file static/index.html was not found.</h2>"

    return html_file.read_text(encoding="utf-8")


@app.get("/api/teams")
def get_teams():
    return [
        {"code": code, "name": name}
        for code, name in NFL_TEAMS.items()
    ]


@app.get("/api/sportsbooks")
def get_sportsbooks():
    return SPORTSBOOKS


@app.get("/api/roster")
def get_roster(team: str):
    team = team.upper().strip()

    if team not in NFL_TEAMS:
        raise HTTPException(status_code=400, detail="Invalid NFL team.")

    df = load_rosters()

    team_col = find_column(df, ["team", "team_abbr", "recent_team"])
    name_col = find_name_column(df)
    pos_col = get_position_column(df)

    if not team_col or not name_col or not pos_col:
        raise HTTPException(
            status_code=500,
            detail="Could not identify required roster columns."
        )

    rows = df[
        df[team_col].astype(str).str.upper() == team
    ].copy()

    status_col = find_column(rows, ["status", "roster_status"])

    if status_col:
        rows = rows[
            rows[status_col].astype(str).str.upper().isin(
                ["ACT", "ACTIVE", "RES", "PUP", "IR"]
            )
        ]

    rows[name_col] = rows[name_col].astype(str)
    rows[pos_col] = rows[pos_col].astype(str).str.upper()

    def format_players(player_rows):
        result = []

        for _, row in player_rows.iterrows():
            result.append({
                "name": str(row[name_col]),
                "position": str(row[pos_col]),
                "status": str(row[status_col]) if status_col else ""
            })

        return sorted(result, key=lambda p: p["name"])

    quarterbacks = rows[rows[pos_col] == "QB"]
    receivers = rows[rows[pos_col].isin(["WR", "TE"])]
    running_backs = rows[rows[pos_col] == "RB"]

    return {
        "team": team,
        "quarterbacks": format_players(quarterbacks),
        "receivers": format_players(receivers),
        "running_backs": format_players(running_backs)
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "season": get_current_nfl_season()
    }


# ---------------------------------------------------------
# RECEIVING PROJECTION
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

    if offense_team not in NFL_TEAMS or defense not in NFL_TEAMS:
        raise HTTPException(status_code=400, detail="Invalid NFL team.")

    if offense_team == defense:
        raise HTTPException(
            status_code=400,
            detail="The offense and defense teams cannot be the same."
        )

    df = load_stats()
    week_col = get_week_column(df)
    opponent_col = get_opponent_column(df)

    if not week_col:
        raise HTTPException(
            status_code=500,
            detail="Could not identify the week column."
        )

    qb_rows = get_player_rows(df, qb, offense_team)

    if qb_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No current-season statistics found for QB {qb}."
        )

    attempts_col = find_column(qb_rows, ["attempts", "passing_attempts"])
    completions_col = find_column(
        qb_rows, ["completions", "passing_completions"]
    )

    if not attempts_col or not completions_col:
        raise HTTPException(
            status_code=500,
            detail="Could not identify QB passing attempts or completions."
        )

    qb_rows["_attempts"] = numeric_series(qb_rows, attempts_col)
    qb_rows["_completions"] = numeric_series(qb_rows, completions_col)

    qb_weekly = qb_rows.groupby(week_col).agg(
        attempts=("_attempts", "sum"),
        completions=("_completions", "sum")
    ).reset_index()

    qb_attempts = weighted_average(qb_weekly["attempts"].tolist())
    qb_completions = weighted_average(qb_weekly["completions"].tolist())

    wr_rows = get_player_rows(df, receiver, offense_team)

    if wr_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No current-season statistics found for receiver {receiver}."
        )

    targets_col = find_column(wr_rows, ["targets"])
    receptions_col = find_column(wr_rows, ["receptions"])
    yards_col = find_column(wr_rows, ["receiving_yards", "rec_yards"])

    if not targets_col or not receptions_col or not yards_col:
        raise HTTPException(
            status_code=500,
            detail="Could not identify receiver targets, receptions, or yards."
        )

    wr_rows["_targets"] = numeric_series(wr_rows, targets_col)
    wr_rows["_receptions"] = numeric_series(wr_rows, receptions_col)
    wr_rows["_yards"] = numeric_series(wr_rows, yards_col)

    wr_weekly = wr_rows.groupby(week_col).agg(
        targets=("_targets", "sum"),
        receptions=("_receptions", "sum"),
        yards=("_yards", "sum")
    ).reset_index()

    matched = pd.merge(qb_weekly, wr_weekly, on=week_col, how="inner")
    matched = matched[matched["attempts"] > 0].copy()

    if matched.empty:
        target_share_decimal = 0.0
    else:
        matched["share"] = matched["targets"] / matched["attempts"]
        target_share_decimal = weighted_average(matched["share"].tolist())

    target_share_percentage = target_share_decimal * 100

    total_targets = wr_rows["_targets"].sum()
    total_receptions = wr_rows["_receptions"].sum()

    catch_rate_decimal = (
        total_receptions / total_targets if total_targets > 0 else 0.0
    )

    recent_yards = weighted_average(wr_weekly["yards"].tolist())

    defense_games = 0
    yards_per_completion = 10.5

    if opponent_col:
        defense_rows = df[
            df[opponent_col].astype(str).str.upper() == defense
        ].copy()

        pos_col = get_position_column(defense_rows)

        if pos_col:
            defense_rows = defense_rows[
                defense_rows[pos_col].astype(str).str.upper() == "QB"
            ]

        pass_yards_col = find_column(
            defense_rows, ["passing_yards", "pass_yards"]
        )
        defense_comp_col = find_column(
            defense_rows, ["completions", "passing_completions"]
        )

        if (
            not defense_rows.empty
            and pass_yards_col
            and defense_comp_col
        ):
            defense_rows["_yards"] = numeric_series(
                defense_rows, pass_yards_col
            )
            defense_rows["_completions"] = numeric_series(
                defense_rows, defense_comp_col
            )

            defense_weekly = defense_rows.groupby(week_col).agg(
                yards=("_yards", "sum"),
                completions=("_completions", "sum")
            ).reset_index()

            defense_weekly = defense_weekly[
                defense_weekly["completions"] > 0
            ]

            defense_games = len(defense_weekly)

            if not defense_weekly.empty:
                defense_weekly["ypc"] = (
                    defense_weekly["yards"] / defense_weekly["completions"]
                )
                yards_per_completion = weighted_average(
                    defense_weekly["ypc"].tolist()
                )

    projected_targets = qb_attempts * target_share_decimal
    projected_receptions = projected_targets * catch_rate_decimal

    original_projection = projected_targets * yards_per_completion
    improved_projection = projected_receptions * yards_per_completion

    sportsbook = sportsbook if sportsbook in SPORTSBOOKS else "Other"
    line = safe_float(line, None) if line is not None else None

    edge = improved_projection - line if line is not None else None

    lean = (
        "OVER" if edge > 0 else
        "UNDER" if edge < 0 else
        "NO EDGE" if edge is not None else
        "NO LINE ENTERED"
    )

    return {
        "offense_team": offense_team,
        "quarterback": qb,
        "receiver": receiver,
        "defense": defense,
        "opponent_name": NFL_TEAMS[defense],
        "season": get_current_nfl_season(),
        "improved_projection": round(improved_projection, 2),
        "original_projection": round(original_projection, 2),
        "qb_attempts_weighted": round(qb_attempts, 2),
        "qb_completions_weighted": round(qb_completions, 2),
        "target_share": round(target_share_percentage, 2),
        "catch_rate": round(catch_rate_decimal * 100, 2),
        "receiver_recent_yards_average": round(recent_yards, 2),
        "projected_targets": round(projected_targets, 2),
        "projected_receptions": round(projected_receptions, 2),
        "yards_per_completion_allowed": round(yards_per_completion, 2),
        "matched_games": int(len(matched)),
        "defense_games_sampled": int(defense_games),
        "sportsbook": sportsbook,
        "line": line,
        "improved_edge": round(edge, 2) if edge is not None else None,
        "lean": lean
    }


# ---------------------------------------------------------
# RUNNING BACK RUSHING PROJECTION
# ---------------------------------------------------------

@app.get("/api/project/rushing")
def project_rushing(
    offense_team: str,
    running_back: str,
    defense: str,
    sportsbook: str = "Other",
    yards_line: float = None,
    attempts_line: float = None
):
    offense_team = offense_team.upper().strip()
    defense = defense.upper().strip()

    if offense_team not in NFL_TEAMS or defense not in NFL_TEAMS:
        raise HTTPException(status_code=400, detail="Invalid NFL team.")

    if offense_team == defense:
        raise HTTPException(
            status_code=400,
            detail="The offense and defense teams cannot be the same."
        )

    df = load_stats()

    week_col = get_week_column(df)
    team_col = get_team_column(df)
    opponent_col = get_opponent_column(df)

    if not week_col or not team_col:
        raise HTTPException(
            status_code=500,
            detail="Could not identify team or week columns."
        )

    carries_col = find_column(
        df, ["carries", "rushing_attempts", "rush_attempts", "attempts"]
    )
    rush_yards_col = find_column(
        df, ["rushing_yards", "rushing_yards_gained", "rush_yards"]
    )

    if not carries_col or not rush_yards_col:
        raise HTTPException(
            status_code=500,
            detail="Could not identify rushing attempts or rushing yards in NFL statistics."
        )

    # Selected running back
    rb_rows = get_player_rows(df, running_back, offense_team)

    if rb_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No current-season rushing statistics found for {running_back}."
        )

    rb_rows["_carries"] = numeric_series(rb_rows, carries_col)
    rb_rows["_yards"] = numeric_series(rb_rows, rush_yards_col)

    rb_weekly = rb_rows.groupby(week_col).agg(
        carries=("_carries", "sum"),
        yards=("_yards", "sum")
    ).reset_index()

    rb_weekly = rb_weekly[rb_weekly["carries"] > 0]

    if rb_weekly.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No rushing attempts found for {running_back} this season."
        )

    rb_weekly["ypc"] = rb_weekly["yards"] / rb_weekly["carries"]

    rb_attempts_average = weighted_average(
        rb_weekly["carries"].tolist()
    )

    rb_yards_average = weighted_average(
        rb_weekly["yards"].tolist()
    )

    rb_ypc = weighted_average(rb_weekly["ypc"].tolist())

    # Team rushing attempts by week.
    team_rows = df[
        df[team_col].astype(str).str.upper() == offense_team
    ].copy()

    team_rows["_carries"] = numeric_series(team_rows, carries_col)

    team_weekly = team_rows.groupby(week_col).agg(
        team_carries=("_carries", "sum")
    ).reset_index()

    team_weekly = team_weekly[team_weekly["team_carries"] > 0]

    # Carry share in weeks where both player and team data exist.
    carry_share_data = pd.merge(
        rb_weekly[["week" if "week" in rb_weekly.columns else week_col, "carries"]],
        team_weekly,
        on=week_col,
        how="inner"
    )

    if carry_share_data.empty:
        carry_share = 0.0
    else:
        carry_share_data["share"] = (
            carry_share_data["carries"]
            / carry_share_data["team_carries"]
        )
        carry_share = weighted_average(
            carry_share_data["share"].tolist()
        )

    team_attempts_average = weighted_average(
        team_weekly["team_carries"].tolist()
    )

    # Estimate the player's attempts from team volume and carry share.
    projected_attempts = team_attempts_average * carry_share

    # Opponent rushing yards allowed per attempt.
    defense_ypc = 4.3
    defense_games = 0

    if opponent_col:
        defense_rows = df[
            df[opponent_col].astype(str).str.upper() == defense
        ].copy()

        defense_rows["_carries"] = numeric_series(
            defense_rows, carries_col
        )
        defense_rows["_yards"] = numeric_series(
            defense_rows, rush_yards_col
        )

        defense_weekly = defense_rows.groupby(week_col).agg(
            carries=("_carries", "sum"),
            yards=("_yards", "sum")
        ).reset_index()

        defense_weekly = defense_weekly[
            defense_weekly["carries"] > 0
        ]

        defense_games = len(defense_weekly)

        if not defense_weekly.empty:
            defense_weekly["ypc"] = (
                defense_weekly["yards"] / defense_weekly["carries"]
            )
            defense_ypc = weighted_average(
                defense_weekly["ypc"].tolist()
            )

    # Blend player efficiency with opponent defensive efficiency.
    adjusted_ypc = (rb_ypc + defense_ypc) / 2

    projected_yards = projected_attempts * adjusted_ypc

    sportsbook = sportsbook if sportsbook in SPORTSBOOKS else "Other"

    yards_line = (
        safe_float(yards_line, None)
        if yards_line is not None else None
    )
    attempts_line = (
        safe_float(attempts_line, None)
        if attempts_line is not None else None
    )

    yards_edge = (
        projected_yards - yards_line
        if yards_line is not None else None
    )

    attempts_edge = (
        projected_attempts - attempts_line
        if attempts_line is not None else None
    )

    def get_lean(edge):
        if edge is None:
            return "NO LINE ENTERED"
        if edge > 0:
            return "OVER"
        if edge < 0:
            return "UNDER"
        return "NO EDGE"

    return {
        "offense_team": offense_team,
        "running_back": running_back,
        "defense": defense,
        "opponent_name": NFL_TEAMS[defense],
        "season": get_current_nfl_season(),

        "projected_attempts": round(projected_attempts, 2),
        "projected_rushing_yards": round(projected_yards, 2),

        "player_attempts_average": round(rb_attempts_average, 2),
        "player_rushing_yards_average": round(rb_yards_average, 2),
        "player_yards_per_carry": round(rb_ypc, 2),

        "team_attempts_average": round(team_attempts_average, 2),
        "carry_share": round(carry_share * 100, 2),

        "defense_yards_per_attempt_allowed": round(defense_ypc, 2),
        "adjusted_yards_per_carry": round(adjusted_ypc, 2),

        "player_games_sampled": int(len(rb_weekly)),
        "defense_games_sampled": int(defense_games),

        "sportsbook": sportsbook,

        "yards_line": yards_line,
        "yards_edge": round(yards_edge, 2) if yards_edge is not None else None,
        "yards_lean": get_lean(yards_edge),

        "attempts_line": attempts_line,
        "attempts_edge": round(attempts_edge, 2) if attempts_edge is not None else None,
        "attempts_lean": get_lean(attempts_edge)
    }


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)
