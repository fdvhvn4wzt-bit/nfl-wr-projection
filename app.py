
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
from functools import lru_cache
from typing import Optional
import pandas as pd

app = FastAPI(title="NFL WR Receiving Yard Projection")

BASE = Path(__file__).parent
STATIC_DIR = BASE / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    "LA": "Los Angeles Rams",
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
    "WAS": "Washington Commanders"
}

SPORTSBOOKS = [
    "DraftKings",
    "FanDuel",
    "BetMGM",
    "Caesars",
    "ESPN BET",
    "Other"
]


@lru_cache(maxsize=1)
def load_stats():
    import nflreadpy as nfl

    year = datetime.now().year

    df = nfl.load_player_stats(
        seasons=list(range(year - 2, year + 1)),
        summary_level="week"
    ).to_pandas()

    if df.empty:
        raise ValueError("No NFL statistics were returned.")

    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    return df


@lru_cache(maxsize=1)
def load_rosters():
    import nflreadpy as nfl

    year = datetime.now().year

    roster = nfl.load_rosters(
        seasons=[year]
    ).to_pandas()

    if roster.empty:
        raise ValueError("No NFL roster data was returned.")

    return roster


def weighted_average(values):
    values = pd.to_numeric(
        values, errors="coerce"
    ).dropna()

    if values.empty:
        return 0.0

    season_avg = values.mean()
    last5_avg = values.tail(5).mean()
    last3_avg = values.tail(3).mean()

    return (
        0.50 * season_avg
        + 0.30 * last5_avg
        + 0.20 * last3_avg
    )


def get_player_game_rows(df, name, position, team):
    team_column = (
        "recent_team"
        if "recent_team" in df.columns
        else "team"
    )

    rows = df[
        (
            df["player_display_name"]
            .astype(str)
            .str.casefold() == name.casefold()
        )
        & (df["position"].astype(str) == position)
        & (
            df[team_column]
            .astype(str)
            .str.upper() == team.upper()
        )
    ].copy()

    if rows.empty:
        raise ValueError(
            f"Could not find {name} as a {position} "
            f"for {team} in the loaded statistics."
        )

    rows = rows.sort_values(["season", "week"])

    latest_season = rows["season"].max()

    return rows[
        rows["season"] == latest_season
    ].copy()


@app.get("/", response_class=HTMLResponse)
def home():
    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail="The static/index.html file is missing."
        )

    return index_file.read_text()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "message": "NFL projection server is running."
    }


@app.get("/api/teams")
def get_teams():
    return [
        {
            "code": code,
            "name": name
        }
        for code, name in sorted(
            NFL_TEAMS.items(),
            key=lambda item: item[1]
        )
    ]


@app.get("/api/sportsbooks")
def get_sportsbooks():
    return SPORTSBOOKS


@app.get("/api/roster")
def get_roster(team: str):
    try:
        roster = load_rosters()

        team = team.upper()

        if team not in NFL_TEAMS:
            raise ValueError("Invalid NFL team selection.")

        roster = roster[
            roster["team"].astype(str).str.upper() == team
        ].copy()

        roster = roster[
            roster["position"].isin(["QB", "WR"])
        ]

        def player_list(position):
            group = roster[
                roster["position"] == position
            ]

            players = []

            for _, row in group.iterrows():
                name = row.get("full_name")

                if pd.isna(name):
                    continue

                players.append({
                    "name": str(name),
                    "position": position,
                    "status": str(row.get("status", "Unknown"))
                })

            return sorted(
                players,
                key=lambda player: player["name"].lower()
            )

        return {
            "team": team,
            "team_name": NFL_TEAMS[team],
            "quarterbacks": player_list("QB"),
            "receivers": player_list("WR")
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )


@app.get("/api/project")
def project(
    offense_team: str,
    qb: str,
    defense: str,
    receiver: str,
    sportsbook: Optional[str] = None,
    line: Optional[float] = None
):
    try:
        df = load_stats()

        offense_team = offense_team.upper()
        defense = defense.upper()

        if offense_team not in NFL_TEAMS:
            raise ValueError("Invalid offense team.")

        if defense not in NFL_TEAMS:
            raise ValueError("Invalid opposing defense.")

        if offense_team == defense:
            raise ValueError(
                "The offense and defense cannot be the same team."
            )

        if line is not None and line < 0:
            raise ValueError("The sportsbook line cannot be negative.")

        required = [
            "player_display_name",
            "position",
            "season",
            "week",
            "attempts",
            "completions",
            "targets",
            "receptions",
            "receiving_yards",
            "passing_yards",
            "opponent_team"
        ]

        missing = [
            column for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"NFL data is missing columns: {missing}"
            )

        # Load selected QB and receiver statistics.

        wr = get_player_game_rows(
            df, receiver, "WR", offense_team
        )

        qb_rows = get_player_game_rows(
            df, qb, "QB", offense_team
        )

        # Match weeks in which the selected QB recorded statistics
        # with weeks in which the selected receiver recorded statistics.

        matched = wr.merge(
            qb_rows[
                [
                    "season",
                    "week",
                    "attempts",
                    "completions"
                ]
            ],
            on=["season", "week"],
            how="inner",
            suffixes=("_wr", "_qb")
        )

        if matched.empty:
            raise ValueError(
                "The selected QB and receiver have no matching "
                "game weeks in the loaded season."
            )

        # Receiver statistics.

        targets = weighted_average(
            matched["targets"]
        )

        receptions = weighted_average(
            matched["receptions"]
        )

        receiver_yards = weighted_average(
            matched["receiving_yards"]
        )

        # QB statistics.

        qb_attempts = weighted_average(
            matched["attempts_qb"]
        )

        qb_completions = weighted_average(
            matched["completions_qb"]
        )

        catch_rate = (
            receptions / targets
            if targets > 0 else 0
        )

        target_share = (
            targets / qb_attempts
            if qb_attempts > 0 else 0
        )

        # Opposing defense statistics.

        defense_rows = df[
            (df["position"] == "QB")
            & (
                df["opponent_team"]
                .astype(str)
                .str.upper() == defense
            )
        ].copy()

        defense_rows["passing_yards"] = pd.to_numeric(
            defense_rows["passing_yards"],
            errors="coerce"
        ).fillna(0)

        defense_rows["completions"] = pd.to_numeric(
            defense_rows["completions"],
            errors="coerce"
        ).fillna(0)

        defense_games = (
            defense_rows.groupby(
                ["season", "week"],
                as_index=False
            )[
                ["passing_yards", "completions"]
            ]
            .sum()
            .sort_values(["season", "week"])
        )

        if defense_games.empty:
            raise ValueError(
                f"No passing defense statistics found for {defense}."
            )

        latest_defense_season = defense_games["season"].max()

        defense_games = defense_games[
            defense_games["season"] == latest_defense_season
        ].copy()

        defense_games = defense_games[
            defense_games["completions"] > 0
        ]

        if defense_games.empty:
            raise ValueError(
                "No valid defensive completion statistics found."
            )

        defense_games["yards_per_completion"] = (
            defense_games["passing_yards"]
            / defense_games["completions"]
        )

        yards_per_completion = weighted_average(
            defense_games["yards_per_completion"]
        )

        # ORIGINAL FORMULA.

        original_projection = (
            qb_completions
            * target_share
            * yards_per_completion
        )

        # IMPROVED FORMULA.

        projected_receptions = (
            qb_completions
            * target_share
            * catch_rate
        )

        improved_projection = (
            projected_receptions
            * yards_per_completion
        )

        result = {
            "offense_team": offense_team,
            "offense_name": NFL_TEAMS[offense_team],
            "quarterback": qb,
            "receiver": receiver,
            "opponent_defense": defense,
            "opponent_name": NFL_TEAMS[defense],

            "original_projection": round(
                original_projection, 1
            ),

            "improved_projection": round(
                improved_projection, 1
            ),

            "projected_receptions": round(
                projected_receptions, 1
            ),

            "target_share": round(
                target_share, 3
            ),

            "catch_rate": round(
                catch_rate, 3
            ),

            "yards_per_completion_allowed": round(
                yards_per_completion, 2
            ),

            "qb_attempts_weighted": round(
                qb_attempts, 1
            ),

            "qb_completions_weighted": round(
                qb_completions, 1
            ),

            "receiver_recent_yards_average": round(
                receiver_yards, 1
            ),

            "matched_games": len(matched),

            "defense_games_sampled": len(defense_games)
        }

        # OPTIONAL SPORTSBOOK COMPARISON.

        if line is not None:

            result["sportsbook"] = (
                sportsbook or "Unspecified"
            )

            result["line"] = line

            result["original_edge"] = round(
                original_projection - line, 1
            )

            result["improved_edge"] = round(
                improved_projection - line, 1
            )

            result["lean"] = (
                "OVER"
                if improved_projection > line
                else "UNDER"
                if improved_projection < line
                else "NO EDGE"
            )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
