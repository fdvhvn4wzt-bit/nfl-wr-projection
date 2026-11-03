
# ============================================================
# POSITION-SPECIFIC TOUCHDOWNS ALLOWED
# Uses completed current-season NFL player statistics.
# No Odds API requests. Existing projections are unchanged.
# ============================================================

TD_CACHE_MINUTES = 60


def defensive_touchdown_data():
    now = datetime.now(timezone.utc)
    current_season = season()
    saved = cache.get("defensive_touchdowns")

    if saved and saved[0] == current_season and now < saved[1]:
        return saved[2]

    df = stats().copy()

    position_col = column(df, ["position", "player_position"])
    opponent_col = column(
        df, ["opponent_team", "defteam", "opponent"]
    )
    week_col = column(df, ["week"])

    if not all([position_col, opponent_col, week_col]):
        raise HTTPException(
            503,
            "NFL statistics are missing position, opponent, or week."
        )

    df["_td_position"] = (
        df[position_col].astype(str).str.upper().str.strip()
    )
    df["_td_opponent"] = (
        df[opponent_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"LA": "LAR"})
    )
    df["_td_week"] = pd.to_numeric(
        df[week_col], errors="coerce"
    )

    df = df[
        df["_td_opponent"].isin(NFL_TEAMS)
        & df["_td_week"].notna()
    ].copy()

    if df.empty:
        raise HTTPException(
            503, "No completed NFL games available for TD analysis."
        )

    # Every observed opponent/week is a completed game.
    # This lets us include zero-TD games rather than only
    # counting games where a touchdown was recorded.
    all_games = (
        df[["_td_opponent", "_td_week"]]
        .drop_duplicates()
        .copy()
    )

    touchdown_columns = {
        "receiving": column(
            df, ["receiving_tds", "receiving_touchdowns"]
        ),
        "rushing": column(
            df, ["rushing_tds", "rushing_touchdowns"]
        ),
        "passing": column(
            df, ["passing_tds", "passing_touchdowns"]
        ),
    }

    if not any(touchdown_columns.values()):
        raise HTTPException(
            503,
            "Touchdown columns are unavailable in NFL statistics."
        )

    result = {}

    definitions = {
        "WR": [("receiving", "Receiving TDs Allowed")],
        "TE": [("receiving", "Receiving TDs Allowed")],
        "RB": [
            ("rushing", "RB Rushing TDs Allowed"),
            ("receiving", "RB Receiving TDs Allowed"),
        ],
        "QB": [
            ("passing", "Passing TDs Allowed"),
            ("rushing", "QB Rushing TDs Allowed"),
        ],
    }

    for position, metrics_to_build in definitions.items():
        position_rows = df[
            df["_td_position"] == position
        ].copy()

        position_result = {}

        for touchdown_type, label in metrics_to_build:
            source_column = touchdown_columns[touchdown_type]

            if source_column is None:
                position_result[touchdown_type] = {
                    "label": label,
                    "available": False,
                    "reason": "Touchdown statistic unavailable",
                }
                continue

            position_rows["_td_value"] = numeric(
                position_rows, source_column
            )

            totals = (
                position_rows.groupby(
                    ["_td_opponent", "_td_week"],
                    as_index=False
                )["_td_value"]
                .sum()
            )

            weekly = (
                all_games.merge(
                    totals,
                    on=["_td_opponent", "_td_week"],
                    how="left"
                )
                .fillna({"_td_value": 0})
                .sort_values("_td_week")
            )

            by_defense = {}

            for defense, games in weekly.groupby(
                "_td_opponent"
            ):
                games = games.sort_values("_td_week")
                values = games["_td_value"].tolist()

                if not values:
                    continue

                total = sum(values)
                game_count = len(values)

                by_defense[defense] = {
                    "label": label,
                    "available": True,
                    "season_total": int(round(total)),
                    "per_game": round(total / game_count, 2),
                    "last_5_total": int(
                        round(sum(values[-5:]))
                    ),
                    "last_5_per_game": round(
                        sum(values[-5:]) / len(values[-5:]), 2
                    ),
                    "games": game_count,
                    "last_5_games": min(5, game_count),
                }

            # Rank defenses from most TDs allowed per game
            # to fewest. Equal rates receive the same rank.
            ranked = sorted(
                by_defense,
                key=lambda team: (
                    -by_defense[team]["per_game"],
                    team
                )
            )

            previous_rate = None
            previous_rank = None

            for index, defense in enumerate(ranked):
                rate = by_defense[defense]["per_game"]

                if previous_rate is not None and rate == previous_rate:
                    rank = previous_rank
                else:
                    rank = index + 1

                by_defense[defense]["nfl_rank"] = rank
                by_defense[defense]["ranked_defenses"] = len(ranked)

                previous_rate = rate
                previous_rank = rank

            position_result[touchdown_type] = by_defense

        result[position] = position_result

    cache["defensive_touchdowns"] = (
        current_season,
        now + timedelta(minutes=TD_CACHE_MINUTES),
        result
    )

    return result


@app.get("/api/defense-touchdowns")
def defense_touchdowns(defense: str, position: str):
    defense = defense.upper().strip()
    position = position.upper().strip()

    if defense not in NFL_TEAMS:
        raise HTTPException(400, "Invalid NFL defense.")

    if position not in ("WR", "TE", "RB", "QB"):
        raise HTTPException(
            400, "Position must be WR, TE, RB, or QB."
        )

    data = defensive_touchdown_data()
    position_data = data.get(position, {})

    result = {}

    for touchdown_type, defenses in position_data.items():
        if not isinstance(defenses, dict):
            continue

        if defenses.get("available") is False:
            result[touchdown_type] = defenses
            continue

        result[touchdown_type] = defenses.get(
            defense,
            {
                "available": False,
                "reason": "No completed games for this defense",
            }
        )

    return {
        "season": season(),
        "defense": defense,
        "defense_name": NFL_TEAMS[defense],
        "position": position,
        "touchdowns": result,
        "uses_odds_credits": False,
        "ranking_note": (
            "Rank 1 means the most touchdowns allowed "
            "per game to this position."
        ),
    }
