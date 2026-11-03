
/* ============================================================
   POSITION-SPECIFIC DEFENSIVE TOUCHDOWNS
   Adds actual TD statistics to existing defense sections.
   Existing projections and features remain unchanged.
   ============================================================ */

function touchdownMetricItems(stat, label) {
    if (!stat || stat.available === false) {
        return [[label, "Data unavailable"]];
    }

    const rank = stat.nfl_rank != null
        ? "#" + stat.nfl_rank + " of " + stat.ranked_defenses
        : "Not ranked";

    return [
        [label + " — Season Total", stat.season_total],
        [label + " — Per Game", stat.per_game],
        [
            label + " — Last " + stat.last_5_games + " Games",
            stat.last_5_total
        ],
        [
            label + " — Last " + stat.last_5_games + " Per Game",
            stat.last_5_per_game
        ],
        [label + " — NFL Rank", rank]
    ];
}

function appendTouchdownMetrics(containerId, items) {
    const container = $(containerId);
    if (!container) return;

    container.querySelectorAll(".td-added-metric")
        .forEach(element => element.remove());

    for (const [label, value] of items) {
        const element = metric(label, value);
        element.classList.add("td-added-metric");
        container.append(element);
    }
}

async function loadDefenseTouchdowns(position, defense) {
    const params = new URLSearchParams({
        position: position,
        defense: defense
    });

    return await jsonFetch(
        "/api/defense-touchdowns?" + params.toString()
    );
}

async function showReceivingTouchdowns(data) {
    const defense = data.defense;
    const player = data.receiver;

    // Determine the receiver's actual WR or TE position
    // from the selected roster option.
    const selected = $("receiver").selectedOptions[0];
    const position = selected?.dataset?.position || "WR";

    const containerId = "receivingDefenseMetrics";

    try {
        const result = await loadDefenseTouchdowns(
            position, defense
        );

        // Prevent an older request from updating a new matchup.
        if (
            $("defense").value !== defense ||
            $("receiver").value !== player
        ) return;

        const stat = result.touchdowns?.receiving;

        appendTouchdownMetrics(
            containerId,
            touchdownMetricItems(
                stat,
                position + " Receiving TDs Allowed"
            )
        );
    } catch (error) {
        appendTouchdownMetrics(containerId, [
            ["Receiving TDs Allowed", "Unavailable: " + error.message]
        ]);
    }
}

async function showRbTouchdowns(data) {
    const defense = data.defense;
    const player = data.running_back;

    try {
        const result = await loadDefenseTouchdowns("RB", defense);

        if (
            $("defense").value !== defense ||
            $("runningBack").value !== player
        ) return;

        appendTouchdownMetrics(
            "rushDefenseMetrics",
            touchdownMetricItems(
                result.touchdowns?.rushing,
                "RB Rushing TDs Allowed"
            )
        );

        appendTouchdownMetrics(
            "rbReceivingDefense",
            touchdownMetricItems(
                result.touchdowns?.receiving,
                "RB Receiving TDs Allowed"
            )
        );
    } catch (error) {
        appendTouchdownMetrics("rushDefenseMetrics", [
            ["RB Rushing TDs Allowed", "Unavailable: " + error.message]
        ]);
        appendTouchdownMetrics("rbReceivingDefense", [
            ["RB Receiving TDs Allowed", "Unavailable: " + error.message]
        ]);
    }
}

async function showQbTouchdowns(data) {
    const defense = data.defense;
    const player = data.quarterback;

    try {
        const result = await loadDefenseTouchdowns("QB", defense);

        if (
            $("defense").value !== defense ||
            $("qbPlayer").value !== player
        ) return;

        const items = [
            ...touchdownMetricItems(
                result.touchdowns?.passing,
                "Passing TDs Allowed"
            ),
            ...touchdownMetricItems(
                result.touchdowns?.rushing,
                "QB Rushing TDs Allowed"
            )
        ];

        appendTouchdownMetrics("qbDefense", items);
    } catch (error) {
        appendTouchdownMetrics("qbDefense", [
            ["QB TDs Allowed", "Unavailable: " + error.message]
        ]);
    }
}

/*
   Save the existing projection functions and extend them.
   This does not replace their calculations.
*/

const originalReceivingProjection = runProjection;
runProjection = async function () {
    await originalReceivingProjection();

    if (projections.receiving && !$("results").hidden) {
        await showReceivingTouchdowns(projections.receiving);
    }
};

const originalRushingProjection = runRushingProjection;
runRushingProjection = async function () {
    await originalRushingProjection();

    if (projections.rushing && !$("rushResults").hidden) {
        await showRbTouchdowns(projections.rushing);
    }
};

const originalQbProjection = runQbProjection;
runQbProjection = async function () {
    await originalQbProjection();

    if (qbProjection && !$("qbResults").hidden) {
        await showQbTouchdowns(qbProjection);
    }
};

/*
   Preserve player positions in the roster dropdown.
   The existing loadRoster() populates the dropdown;
   this wrapper adds WR/TE position metadata afterward.
*/

const originalLoadRoster = loadRoster;

loadRoster = async function () {
    await originalLoadRoster();

    const offense = $("offense").value;
    if (!offense) return;

    try {
        const roster = await jsonFetch(
            "/api/roster?team=" + encodeURIComponent(offense)
        );

        if ($("offense").value !== offense) return;

        const positions = new Map(
            (roster.receivers || []).map(player => [
                player.name,
                player.position
            ])
        );

        for (const option of $("receiver").options) {
            const position = positions.get(option.value);

            if (position === "WR" || position === "TE") {
                option.dataset.position = position;
            }
        }
    } catch (error) {
        console.warn(
            "Could not load WR/TE position metadata:",
            error.message
        );
    }
};
