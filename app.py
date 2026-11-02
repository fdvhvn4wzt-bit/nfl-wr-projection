
# ============================================================
# NFL WEEKLY MATCHUP DROPDOWN
# Uses nflverse schedules, not Odds API credits.
# Includes games already played in the selected NFL week.
# ============================================================

@app.get("/api/matchups")
def get_matchups():
    from zoneinfo import ZoneInfo

    now = datetime.now(timezone.utc)
    current_season = season()

    cached = cache.get("matchup_schedule")
    if cached and cached[0] == current_season and now < cached[1]:
        return cached[2]

    try:
        schedule = nfl.load_schedules(
            seasons=[current_season]
        ).to_pandas()
    except Exception as exc:
        raise HTTPException(
            503,
            f"Unable to load NFL matchups: {exc}"
        )

    home_col = column(schedule, ["home_team"])
    away_col = column(schedule, ["away_team"])
    week_col = column(schedule, ["week"])
    date_col = column(schedule, ["gameday", "game_date"])
    time_col = column(schedule, ["gametime", "game_time"])
    type_col = column(schedule, ["game_type", "season_type"])

    if not all([home_col, away_col, week_col, date_col]):
        raise HTTPException(
            503,
            "NFL schedule is missing required fields."
        )

    schedule = schedule.copy()

    if type_col:
        schedule = schedule[
            schedule[type_col].astype(str).str.upper().isin(
                ["REG", "REGULAR"]
            )
        ].copy()

    kickoff_col = column(
        schedule,
        ["start_time", "game_datetime", "kickoff"]
    )

    if kickoff_col:
        kickoff_times = pd.to_datetime(
            schedule[kickoff_col],
            errors="coerce",
            utc=True
        )
    elif time_col:
        local_times = pd.to_datetime(
            schedule[date_col].astype(str)
            + " "
            + schedule[time_col].astype(str),
            errors="coerce"
        )
        kickoff_times = (
            local_times.dt.tz_localize(
                "America/New_York",
                ambiguous="NaT",
                nonexistent="NaT"
            ).dt.tz_convert("UTC")
        )
    else:
        raise HTTPException(
            503,
            "NFL schedule kickoff times are unavailable."
        )

    schedule["_kickoff"] = kickoff_times
    schedule["_week"] = pd.to_numeric(
        schedule[week_col],
        errors="coerce"
    )

    schedule = schedule.dropna(
        subset=["_kickoff", "_week"]
    ).sort_values("_kickoff")

    if schedule.empty:
        return {
            "season": current_season,
            "week": None,
            "games": [],
            "count": 0,
            "uses_odds_credits": False
        }

    now_ts = pd.Timestamp(now)

    # Keep the current week visible while its games are
    # being played, including games already completed.
    weeks = []

    for week_number, rows in schedule.groupby("_week"):
        first = rows["_kickoff"].min()
        last = rows["_kickoff"].max()

        weeks.append({
            "week": int(week_number),
            "first": first,
            "last": last
        })

    weeks.sort(key=lambda item: item["first"])

    active = next(
        (
            item for item in weeks
            if item["first"] - pd.Timedelta(days=2)
            <= now_ts
            <= item["last"] + pd.Timedelta(days=1)
        ),
        None
    )

    if active is None:
        active = next(
            (
                item for item in weeks
                if item["last"] >= now_ts
            ),
            None
        )

    if active is None:
        active = weeks[-1]

    selected_week = active["week"]

    selected = schedule[
        schedule["_week"] == selected_week
    ].sort_values("_kickoff")

    hawaii = ZoneInfo("Pacific/Honolulu")
    games = []

    for _, game in selected.iterrows():
        home = str(game[home_col]).upper().strip()
        away = str(game[away_col]).upper().strip()

        if home == "LA":
            home = "LAR"
        if away == "LA":
            away = "LAR"

        if home not in NFL_TEAMS or away not in NFL_TEAMS:
            continue

        kickoff = game["_kickoff"].to_pydatetime()
        kickoff_hawaii = kickoff.astimezone(hawaii)

        games.append({
            "id": f"{selected_week}-{away}-{home}",
            "week": selected_week,
            "away": away,
            "home": home,
            "away_name": NFL_TEAMS[away],
            "home_name": NFL_TEAMS[home],
            "kickoff_utc": kickoff.isoformat(),
            "kickoff_hst": kickoff_hawaii.strftime(
                "%a %b %-d, %-I:%M %p HST"
            ),
            "started": kickoff <= now
        })

    result = {
        "season": current_season,
        "week": selected_week,
        "games": games,
        "count": len(games),
        "uses_odds_credits": False
    }

    cache["matchup_schedule"] = (
        current_season,
        now + timedelta(minutes=30),
        result
    )

    return result
