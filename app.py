from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd

app = FastAPI(title=“NFL WR Receiving Yard Projection”)

BASE = Path(file).parent
STATIC_DIR = BASE / “static”

if STATIC_DIR.exists():
app.mount(”/static”, StaticFiles(directory=STATIC_DIR), name=“static”)

def load_stats():
import nflreadpy as nfl

year = datetime.now().year
seasons = list(range(year - 2, year + 1))
df = nfl.load_player_stats(
    seasons=seasons,
    summary_level="week"
).to_pandas()
if df.empty:
    raise ValueError("No NFL statistics were returned.")
return df

def weighted_average(values):
values = pd.to_numeric(values, errors=“coerce”).dropna()

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

def player_rows(df, name, position):
required = [“player_display_name”, “position”]

for column in required:
    if column not in df.columns:
        raise ValueError(
            f"NFL data is missing the required column: {column}"
        )
rows = df[
    (df["player_display_name"].astype(str).str.lower() == name.lower())
    & (df["position"].astype(str) == position)
].copy()
if rows.empty:
    raise ValueError(
        f"Could not find {name} as a {position}. "
        "Check the spelling of the player name."
    )
if "season_type" in rows.columns:
    rows = rows[rows["season_type"] == "REG"]
if rows.empty:
    raise ValueError("No regular-season statistics were found.")
rows = rows.sort_values(
    [c for c in ["season", "week"] if c in rows.columns]
)
latest_season = rows["season"].max() if "season" in rows else None
if latest_season is not None:
    rows = rows[rows["season"] == latest_season]
return rows

@app.get(”/”, response_class=HTMLResponse)
def home():
index_file = STATIC_DIR / “index.html”

if not index_file.exists():
    raise HTTPException(
        status_code=500,
        detail="The static/index.html file is missing."
    )
return index_file.read_text()

@app.get(”/api/health”)
def health():
return {
“status”: “ok”,
“message”: “NFL projection server is running.”
}

@app.get(”/api/players”)
def players():
try:
df = load_stats()

    if "player_display_name" not in df.columns:
        raise ValueError("Player names are unavailable in the NFL data.")
    if "position" in df.columns:
        wr = df[df["position"] == "WR"]
    else:
        wr = df
    names = sorted(
        wr["player_display_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    teams = []
    if "recent_team" in wr.columns:
        teams = sorted(
            wr["recent_team"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    return {
        "players": names,
        "teams": teams
    }
except Exception as e:
    raise HTTPException(status_code=503, detail=str(e))

@app.get(”/api/project”)
def project(
qb: str,
defense: str,
receiver: str,
line: Optional[float] = None
):
try:
df = load_stats()

    needed = [
        "player_display_name",
        "position",
        "attempts",
        "completions",
        "targets",
        "receptions",
        "receiving_yards"
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"NFL data is missing these columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    wr = player_rows(df, receiver, "WR")
    qb_rows = player_rows(df, qb, "QB")
    for column in [
        "attempts",
        "completions",
        "targets",
        "receptions",
        "receiving_yards"
    ]:
        wr[column] = pd.to_numeric(
            wr[column], errors="coerce"
        ).fillna(0)
    qb_rows["attempts"] = pd.to_numeric(
        qb_rows["attempts"], errors="coerce"
    ).fillna(0)
    qb_rows["completions"] = pd.to_numeric(
        qb_rows["completions"], errors="coerce"
    ).fillna(0)
    targets = weighted_average(wr["targets"])
    receptions = weighted_average(wr["receptions"])
    receiving_yards = weighted_average(wr["receiving_yards"])
    qb_attempts = weighted_average(qb_rows["attempts"])
    qb_completions = weighted_average(qb_rows["completions"])
    catch_rate = (
        receptions / targets if targets > 0 else 0
    )
    target_share = (
        targets / qb_attempts if qb_attempts > 0 else 0
    )
    # Find opponent passing production against the selected defense.
    yards_per_completion = 10.5
    if (
        "opponent_team" in df.columns
        and "passing_yards" in df.columns
    ):
        defense_rows = df[
            (df["opponent_team"].astype(str).str.upper()
             == defense.upper())
            & (df["position"].astype(str) == "QB")
        ].copy()
        if "season_type" in defense_rows.columns:
            defense_rows = defense_rows[
                defense_rows["season_type"] == "REG"
            ]
        if not defense_rows.empty:
            defense_rows["passing_yards"] = pd.to_numeric(
                defense_rows["passing_yards"],
                errors="coerce"
            ).fillna(0)
            defense_rows["completions"] = pd.to_numeric(
                defense_rows["completions"],
                errors="coerce"
            ).fillna(0)
            total_completions = defense_rows["completions"].sum()
            if total_completions > 0:
                yards_per_completion = (
                    defense_rows["passing_yards"].sum()
                    / total_completions
                )
    projected_receptions = qb_completions * target_share * catch_rate
    projection = projected_receptions * yards_per_completion
    result = {
        "projection_yards": round(projection, 1),
        "projected_receptions": round(projected_receptions, 1),
        "target_share": round(target_share, 3),
        "catch_rate": round(catch_rate, 3),
        "yards_per_completion_allowed": round(
            yards_per_completion, 2
        ),
        "qb_attempts_weighted": round(qb_attempts, 1),
        "qb_completions_weighted": round(qb_completions, 1),
        "receiver_recent_yards_average": round(
            receiving_yards, 1
        )
    }
    if line is not None:
        edge = projection - line
        result["line"] = line
        result["edge"] = round(edge, 1)
        result["lean"] = (
            "OVER" if edge > 0
            else "UNDER" if edge < 0
            else "NO EDGE"
        )
    return result
except Exception as e:
    raise HTTPException(
        status_code=400,
        detail=str(e)
    )
