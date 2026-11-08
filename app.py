"""NFL Player Prop Projections - backend.

Backward compatible with the previous version: every endpoint, response field and
projection formula the dashboard already uses is kept. Additions:
  * bug fixes (LA/LAC team codes, stale weekly cache, rushing double-count, under prices)
  * speed (cached defensive tables, no per-player recomputation, gzip)
  * /api/analyze-line  -> probability of OVER/UNDER, de-vigged market odds, EV
  * /api/injury-status -> official injury designation for the selected player
  * /api/live-odds returns DraftKings only by default (set ODDS_BOOKMAKERS to add other books)
"""
import hmac, json, math, os, re, time
from statistics import NormalDist
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import nflreadpy as nfl
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title='NFL Player Prop Projections')
app.add_middleware(GZipMiddleware, minimum_size=800)

NAMES = '''ARI|Arizona Cardinals;ATL|Atlanta Falcons;BAL|Baltimore Ravens;BUF|Buffalo Bills;CAR|Carolina Panthers;CHI|Chicago Bears;CIN|Cincinnati Bengals;CLE|Cleveland Browns;DAL|Dallas Cowboys;DEN|Denver Broncos;DET|Detroit Lions;GB|Green Bay Packers;HOU|Houston Texans;IND|Indianapolis Colts;JAX|Jacksonville Jaguars;KC|Kansas City Chiefs;LAC|Los Angeles Chargers;LAR|Los Angeles Rams;LV|Las Vegas Raiders;MIA|Miami Dolphins;MIN|Minnesota Vikings;NE|New England Patriots;NO|New Orleans Saints;NYG|New York Giants;NYJ|New York Jets;PHI|Philadelphia Eagles;PIT|Pittsburgh Steelers;SEA|Seattle Seahawks;SF|San Francisco 49ers;TB|Tampa Bay Buccaneers;TEN|Tennessee Titans;WAS|Washington Commanders'''
NFL_TEAMS = dict(x.split('|') for x in NAMES.split(';'))
SPORTSBOOKS = ['DraftKings', 'Other']
MARKETS = {'receiving_yards': 'player_reception_yds', 'receptions': 'player_receptions',
           'rushing_yards': 'player_rush_yds', 'rushing_attempts': 'player_rush_attempts',
           'passing_yards': 'player_pass_yds', 'passing_attempts': 'player_pass_attempts',
           'passing_completions': 'player_pass_completions', 'passing_touchdowns': 'player_pass_tds',
           'interceptions': 'player_pass_interceptions'}
PROP_BY_MARKET = {v: k for k, v in MARKETS.items()}

cache = {}
TABLES = {}                      # derived defensive tables; cleared whenever stats reload
odds_lock = Lock()
load_lock = Lock()
ODDS_CACHE_MINUTES = int(os.getenv('ODDS_CACHE_MINUTES', '360'))
RATE = {}

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
TEAM_ALIASES = {'LA': 'LAR', 'JAC': 'JAX', 'WSH': 'WAS'}
NAME_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}


def norm_team(x):
    s = str(x).upper().strip()
    return TEAM_ALIASES.get(s, s)


def norm_name(x):
    s = re.sub(r'[^a-z\s]', '', str(x).lower().replace('-', ' '))
    return ' '.join(p for p in s.split() if p not in NAME_SUFFIXES)


def season():
    today = date.today()
    return today.year if today.month >= 8 else today.year - 1


def column(df, choices):
    found = {str(c).lower(): c for c in df.columns}
    return next((found[x.lower()] for x in choices if x.lower() in found), None)


def numeric(df, c):
    return pd.to_numeric(df[c], errors='coerce').fillna(0) if c else pd.Series(0.0, index=df.index)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def rate_limit(request: Request, bucket: str, limit: int, window: int = 60):
    fwd = request.headers.get('x-forwarded-for', '').split(',')[0].strip()
    ip = fwd or (request.client.host if request.client else 'unknown')
    now = time.time()
    key = (bucket, ip)
    hits = [t for t in RATE.get(key, []) if now - t < window]
    if len(hits) >= limit:
        raise HTTPException(429, 'Too many requests. Please wait a moment and try again.')
    hits.append(now)
    RATE[key] = hits


FAILED_CODES = {}


def check_access(request: Request):
    """Odds lookups spend paid API credits, so they need the owner's access code.

    The code lives in the ODDS_ACCESS_CODE environment variable (Render > Environment).
    If it is not set, lookups stay locked rather than open, so credits are safe by default.
    """
    expected = os.getenv('ODDS_ACCESS_CODE', '')
    if not expected:
        raise HTTPException(503, 'Live odds are locked. The owner needs to set ODDS_ACCESS_CODE on the server.')
    fwd = request.headers.get('x-forwarded-for', '').split(',')[0].strip()
    ip = fwd or (request.client.host if request.client else 'unknown')
    now = time.time()
    fails = [x for x in FAILED_CODES.get(ip, []) if now - x < 600]
    if len(fails) >= 8:
        FAILED_CODES[ip] = fails
        raise HTTPException(429, 'Too many wrong codes. Try again in 10 minutes.')
    supplied = request.headers.get('x-access-code', '')
    if not hmac.compare_digest(supplied.encode('utf-8'), expected.encode('utf-8')):
        fails.append(now)
        FAILED_CODES[ip] = fails
        raise HTTPException(401, 'Incorrect access code.')


def avg(values):
    v = [float(x) for x in values if pd.notna(x) and math.isfinite(float(x)) and float(x) >= 0]
    return (sum(v) / len(v) * .5 + sum(v[-5:]) / len(v[-5:]) * .3 + sum(v[-3:]) / len(v[-3:]) * .2) if v else 0.0


def trend(values):
    """Automatic last-3/last-5/season baseline; handles short samples."""
    v = [float(x) for x in values if pd.notna(x) and math.isfinite(float(x))]
    if not v:
        return {'last_3': 0.0, 'last_5': 0.0, 'season': 0.0, 'weighted': 0.0, 'games': 0}
    season_mean = sum(v) / len(v)
    last3 = sum(v[-3:]) / len(v[-3:])
    last5 = sum(v[-5:]) / len(v[-5:])
    return {'last_3': round(last3, 2), 'last_5': round(last5, 2), 'season': round(season_mean, 2),
            'weighted': round(.40 * last3 + .35 * last5 + .25 * season_mean, 2), 'games': len(v)}


def confidence(values, defense_games=0):
    """Heuristic data-quality label, NOT a probability or betting recommendation."""
    v = [float(x) for x in values if pd.notna(x) and math.isfinite(float(x))]
    n = len(v)
    if n < 3:
        return 'LOW'
    mean = sum(v) / n
    if mean <= 0:
        return 'LOW'
    sd = (sum((x - mean) ** 2 for x in v) / n) ** 0.5
    cv = sd / max(abs(mean), 1.0)
    if n >= 8 and defense_games >= 5 and cv <= 0.35:
        return 'HIGH'
    if n >= 5 and defense_games >= 3 and cv <= 0.65:
        return 'MEDIUM'
    return 'LOW'


def matchup_factor(defense_rate, league_rate, minimum=.80, maximum=1.20):
    if league_rate <= 0:
        return 1.0
    return max(minimum, min(maximum, defense_rate / league_rate))


def team_codes(code):
    return ['LAR', 'LA'] if code in ('LAR', 'LA') else [code]


def usage_adjustment(values):
    v = [float(x) for x in values if pd.notna(x) and math.isfinite(float(x))]
    if len(v) < 4:
        return 1.0, 'Limited workload history'
    baseline = sum(v) / len(v)
    if baseline <= 0:
        return 1.0, 'Limited workload history'
    recent = sum(v[-3:]) / min(3, len(v))
    change = recent / baseline - 1.0
    factor = 1.0 + max(-.10, min(.10, change * .30))
    return factor, ('Usage rising' if factor > 1.025 else 'Usage falling' if factor < .975 else 'Usage steady')


def lean(e):
    return 'NO LINE' if e is None else ('OVER' if e > 0 else 'UNDER' if e < 0 else 'EVEN')


def validate(offense, defense):
    offense = norm_team(offense)
    defense = norm_team(defense)
    if offense not in NFL_TEAMS or defense not in NFL_TEAMS or offense == defense:
        raise HTTPException(400, 'Select two different valid NFL teams.')
    return offense, defense


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
def stats():
    now = datetime.now()
    s = season()
    item = cache.get('stats')
    if item and item[0] == s and now < item[1]:
        return item[2]
    with load_lock:
        item = cache.get('stats')
        if item and item[0] == s and now < item[1]:
            return item[2]
        try:
            d = nfl.load_player_stats(seasons=[s], summary_level='week').to_pandas()
        except Exception as exc:
            if item:                       # serve slightly stale data rather than fail
                return item[2]
            raise HTTPException(503, f'Unable to load NFL statistics: {exc}')
        if d is None or d.empty:
            if item:
                return item[2]
            raise HTTPException(503, f'No current-season stats for {s}.')
        sc = column(d, ['season'])
        st = column(d, ['season_type'])
        if sc:
            d = d[pd.to_numeric(d[sc], errors='coerce') == s]
        if st:
            d = d[d[st].astype(str).str.upper().isin(['REG', 'REGULAR'])]
        if d.empty:
            if item:
                return item[2]
            raise HTTPException(503, 'No regular-season stats available yet.')
        d = d.copy()
        nc = column(d, ['player_display_name', 'player_name', 'full_name', 'display_name', 'player'])
        tc = column(d, ['recent_team', 'team', 'team_abbr', 'posteam'])
        oc = column(d, ['opponent_team', 'defteam', 'opponent'])
        pc = column(d, ['position', 'player_position'])
        wc = column(d, ['week'])
        if not all([nc, tc, wc]):
            raise HTTPException(503, 'NFL statistics are missing player, team or week columns.')
        # Normalised helper columns computed ONCE per load (not per request).
        d['_name'] = d[nc].map(norm_name)
        d['_team'] = d[tc].map(norm_team)
        d['_opp'] = d[oc].map(norm_team) if oc else ''
        d['_pos'] = d[pc].astype(str).str.upper().str.strip() if pc else ''
        d['_week'] = pd.to_numeric(d[wc], errors='coerce')
        TABLES.clear()
        cache['stats'] = (s, now + timedelta(minutes=60), d)
        return d


def rosters():
    now = datetime.now()
    s = season()
    item = cache.get('rosters')
    if item and item[0] == s and now < item[1]:
        return item[2]
    try:
        d = nfl.load_rosters(seasons=[s]).to_pandas()
    except Exception as exc:
        if item:
            return item[2]
        raise HTTPException(503, f'Unable to load NFL rosters: {exc}')
    if d is None or d.empty:
        raise HTTPException(503, 'NFL roster data unavailable.')
    cache['rosters'] = (s, now + timedelta(hours=12), d)
    return d


def schedule_df():
    """Current-season regular-season schedule with UTC kickoff times (cached 1h)."""
    now = datetime.now()
    s = season()
    item = cache.get('schedule')
    if item and item[0] == s and now < item[1]:
        return item[2]
    try:
        schedule = nfl.load_schedules(seasons=[s]).to_pandas()
    except Exception as exc:
        if item:
            return item[2]
        raise HTTPException(503, f'Unable to load NFL schedule: {exc}')
    hc = column(schedule, ['home_team'])
    ac = column(schedule, ['away_team'])
    dc = column(schedule, ['gameday', 'game_date'])
    tc = column(schedule, ['gametime', 'game_time'])
    wc = column(schedule, ['week'])
    st = column(schedule, ['game_type', 'season_type'])
    if not all([hc, ac, dc, wc]):
        raise HTTPException(503, 'NFL schedule is missing team, date, or week fields.')
    schedule = schedule.copy()
    if st:
        schedule = schedule[schedule[st].astype(str).str.upper().isin(['REG', 'REGULAR'])].copy()
    kickoff = column(schedule, ['start_time', 'game_datetime', 'kickoff'])
    if kickoff:
        times = pd.to_datetime(schedule[kickoff], errors='coerce', utc=True)
    elif tc:
        local = pd.to_datetime(schedule[dc].astype(str) + ' ' + schedule[tc].astype(str), errors='coerce')
        times = local.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    else:
        raise HTTPException(503, 'NFL schedule has no kickoff times; refusing to guess matchups.')
    schedule['_kickoff'] = times
    schedule['_week'] = pd.to_numeric(schedule[wc], errors='coerce')
    schedule['_home'] = schedule[hc].map(norm_team)
    schedule['_away'] = schedule[ac].map(norm_team)
    cache['schedule'] = (s, now + timedelta(hours=1), schedule)
    return schedule


def injury_data():
    """Official injury report designations (cached 30 min). Never raises."""
    now = datetime.now()
    s = season()
    item = cache.get('injuries')
    if item and item[0] == s and now < item[1]:
        return item[2]
    empty = {'df': pd.DataFrame(columns=['_team', '_name', '_display', '_week', '_status', '_injury']), 'weeks': set()}
    try:
        d = nfl.load_injuries(seasons=[s]).to_pandas()
        tc = column(d, ['team'])
        nc = column(d, ['full_name', 'player_display_name', 'player_name'])
        wc = column(d, ['week'])
        sc = column(d, ['report_status'])
        gt = column(d, ['game_type', 'season_type'])
        ic = column(d, ['report_primary_injury'])
        if not all([tc, nc, wc, sc]):
            raise ValueError('injury columns missing')
        if gt:
            d = d[d[gt].astype(str).str.upper().isin(['REG', 'REGULAR'])]
        out = pd.DataFrame({
            '_team': d[tc].map(norm_team), '_name': d[nc].map(norm_name), '_display': d[nc].astype(str),
            '_week': pd.to_numeric(d[wc], errors='coerce'),
            '_status': d[sc].astype(str).str.upper().str.strip(),
            '_injury': d[ic].astype(str) if ic else ''})
        weeks = {int(w) for w in out['_week'].dropna().unique()}
        out = out[out['_week'].notna() & out['_status'].isin(['OUT', 'DOUBTFUL', 'QUESTIONABLE'])]
        data = {'df': out, 'weeks': weeks}
        ttl = 30
    except Exception:
        data, ttl = empty, 10
    cache['injuries'] = (s, now + timedelta(minutes=ttl), data)
    return data


def upcoming_week():
    """(week number, next kickoff) of the nearest unplayed game, or (None, None)."""
    try:
        d = schedule_df()
    except HTTPException:
        return None, None
    now = pd.Timestamp.now(tz='UTC')
    up = d[(d['_kickoff'] > now) & d['_week'].notna()]
    if up.empty:
        return None, None
    first = up.sort_values('_kickoff').iloc[0]
    return int(first['_week']), first['_kickoff']


def injury_status_for(team, name, week):
    """Returns (status, injury_text, verified)."""
    data = injury_data()
    if week is None or int(week) not in data['weeks']:
        return 'UNKNOWN', '', False
    d = data['df']
    r = d[(d['_team'] == norm_team(team)) & (d['_name'] == norm_name(name)) & (d['_week'] == week)]
    if r.empty:
        return 'HEALTHY', '', True
    x = r.iloc[-1]
    return x['_status'], x['_injury'] if x['_injury'] not in ('nan', 'None') else '', True


def player(df, name, team):
    """Player's weekly rows. Matches punctuation/suffix variants and keeps traded players."""
    r = df[df['_name'] == norm_name(name)]
    at_team = r[r['_team'].isin([norm_team(team)])]
    r = (at_team if not at_team.empty else r).copy()
    return r.sort_values('_week')


# ----------------------------------------------------------------------------
# Defensive tables (cached - previously recomputed on every call)
# ----------------------------------------------------------------------------
def _pos_table(df, position, metric, week_col, opponent_col):
    yc = column(df, metric)
    key = ('pos', position, yc)
    hit = TABLES.get(key)
    if hit is not None:
        return hit
    z = df[df['_pos'] == position]
    if not yc or z.empty or not week_col:
        TABLES[key] = None
        return None
    z = z.assign(_yards=numeric(z, yc))
    grouped = z.groupby(['_opp', '_week'], as_index=False)['_yards'].sum()
    if grouped.empty:
        TABLES[key] = None
        return None
    league = float(grouped['_yards'].mean())
    per_team = grouped.groupby('_opp')['_yards'].agg(['mean', 'count'])
    TABLES[key] = (league, per_team)
    return TABLES[key]


def position_defense(df, opponent, position, metric, week_col, opponent_col):
    """Per-game yards allowed to a position vs league average (shrunk, capped)."""
    if not opponent_col or not week_col:
        return 1.0, 0, None, None
    t = _pos_table(df, position, metric, week_col, opponent_col)
    if t is None:
        return 1.0, 0, None, None
    league, per_team = t
    opp = norm_team(opponent)
    if opp not in per_team.index or league <= 0:
        return 1.0, 0, None, round(league, 2) if league > 0 else None
    allowed = float(per_team.loc[opp, 'mean'])
    games = int(per_team.loc[opp, 'count'])
    raw = matchup_factor(allowed, league, .80, 1.20)
    factor = 1.0 + (raw - 1.0) * .60 * games / (games + 4)
    return factor, games, round(allowed, 2), round(league, 2)


def rb_receiving_defense(df, opponent, week_col, opponent_col):
    """Per-game RB targets, catches and receiving yards allowed; zeros included."""
    key = ('rbrec',)
    t = TABLES.get(key)
    if t is None:
        tc = column(df, ['targets'])
        rc = column(df, ['receptions'])
        yc = column(df, ['receiving_yards', 'rec_yards'])
        if not all([tc, rc, yc, week_col, opponent_col]):
            t = False
        else:
            games = df[['_opp', '_week']].dropna().drop_duplicates()
            rb = df[df['_pos'] == 'RB']
            rb = rb.assign(_t=numeric(rb, tc), _r=numeric(rb, rc), _y=numeric(rb, yc))
            totals = rb.groupby(['_opp', '_week'], as_index=False).agg(
                targets=('_t', 'sum'), receptions=('_r', 'sum'), yards=('_y', 'sum'))
            t = games.merge(totals, on=['_opp', '_week'], how='left').fillna(
                {'targets': 0, 'receptions': 0, 'yards': 0})
        TABLES[key] = t
    if t is False:
        return {'games': 0, 'targets': None, 'receptions': None, 'yards': None,
                'league_receptions': None, 'league_yards': None}
    selected = t[t['_opp'] == norm_team(opponent)]

    def m(frame, col):
        return round(float(frame[col].mean()), 2) if not frame.empty else None
    return {'games': len(selected), 'targets': m(selected, 'targets'), 'receptions': m(selected, 'receptions'),
            'yards': m(selected, 'yards'), 'league_receptions': m(t, 'receptions'), 'league_yards': m(t, 'yards')}


# ----------------------------------------------------------------------------
# Basic routes
# ----------------------------------------------------------------------------
@app.get('/', response_class=HTMLResponse)
def home():
    return (BASE_DIR / 'static' / 'index.html').read_text(encoding='utf-8')


@app.get('/api/teams')
def teams():
    return [{'code': k, 'name': v} for k, v in NFL_TEAMS.items()]


@app.get('/api/sportsbooks')
def books():
    return SPORTSBOOKS


@app.get('/api/health')
def health():
    return {'status': 'ok', 'season': season(), 'live_odds_configured': bool(os.getenv('ODDS_API_KEY')),
            'odds_lock_set': bool(os.getenv('ODDS_ACCESS_CODE'))}


@app.get('/api/matchups')
def matchups():
    """Every regular-season game (upcoming and played) for the matchup picker."""
    item = cache.get('matchups')
    if item and time.time() < item[0]:
        return item[1]
    d = schedule_df()
    now = pd.Timestamp.now(tz='UTC')
    home_score = column(d, ['home_score'])
    away_score = column(d, ['away_score'])
    current_week, _ = upcoming_week()
    games = []
    for _, g in d.sort_values('_kickoff').iterrows():
        home, away = g['_home'], g['_away']
        if home not in NFL_TEAMS or away not in NFL_TEAMS or pd.isna(g['_week']):
            continue
        hs = g[home_score] if home_score else None
        a_s = g[away_score] if away_score else None
        final = hs is not None and a_s is not None and pd.notna(hs) and pd.notna(a_s)
        kickoff = g['_kickoff']
        started = pd.notna(kickoff) and kickoff <= now
        games.append({
            'week': int(g['_week']), 'away': away, 'home': home,
            'away_name': NFL_TEAMS[away], 'home_name': NFL_TEAMS[home],
            'kickoff': kickoff.isoformat() if pd.notna(kickoff) else None,
            'status': 'final' if final else 'started' if started else 'upcoming',
            'away_score': int(a_s) if final else None,
            'home_score': int(hs) if final else None})
    result = {'season': season(), 'current_week': current_week, 'games': games}
    cache['matchups'] = (time.time() + 120, result)
    return result


@app.get('/api/roster')
def roster(team: str):
    team = norm_team(team)
    if team not in NFL_TEAMS:
        raise HTTPException(400, 'Invalid NFL team.')
    d = rosters()
    tc = column(d, ['team', 'team_abbr', 'recent_team'])
    nc = column(d, ['player_display_name', 'player_name', 'full_name', 'display_name', 'player'])
    pc = column(d, ['position', 'player_position'])
    if not all([tc, nc, pc]):
        raise HTTPException(500, 'Roster columns missing.')
    r = d[d[tc].map(norm_team) == team].copy()
    sc = column(r, ['status', 'roster_status'])
    if sc:
        r = r[r[sc].astype(str).str.upper().isin(['ACT', 'ACTIVE', 'RES', 'PUP', 'IR'])]
    r[pc] = r[pc].astype(str).str.upper()
    # Workload per player so starters sort first (fixes "first QB is a backup").
    usage = {}
    try:
        s = stats()
        t = s[s['_team'] == team]
        for col, key in (('attempts', 'QB'), ('targets', 'REC'), ('carries', 'RB')):
            c = column(t, [col, 'passing_attempts' if col == 'attempts' else 'rushing_attempts' if col == 'carries' else col])
            if c:
                usage[key] = numeric(t, c).groupby(t['_name']).sum().to_dict()
    except HTTPException:
        pass
    week, _ = upcoming_week()
    inj = injury_data()
    injmap = {}
    if week is not None and int(week) in inj['weeks']:
        x = inj['df']
        x = x[(x['_team'] == team) & (x['_week'] == week)]
        injmap = dict(zip(x['_name'], x['_status']))

    def listing(positions, key):
        u = usage.get(key, {})
        rows = []
        for _, x in r[r[pc].isin(positions)].iterrows():
            nm = str(x[nc])
            rows.append({'name': nm, 'position': str(x[pc]), 'status': str(x[sc]) if sc else '',
                         'injury': injmap.get(norm_name(nm), ''), '_u': float(u.get(norm_name(nm), 0))})
        rows.sort(key=lambda z: (-z['_u'], z['name']))
        for z in rows:
            z.pop('_u')
        return rows
    return {'team': team, 'quarterbacks': listing(['QB'], 'QB'), 'receivers': listing(['WR', 'TE'], 'REC'),
            'running_backs': listing(['RB'], 'RB')}


# ----------------------------------------------------------------------------
# Projections (formulas unchanged except the rushing fix noted below)
# ----------------------------------------------------------------------------
@app.get('/api/project')
def project(offense_team: str, qb: str, receiver: str, defense: str, sportsbook: str = 'Other',
            line: float | None = None, receptions_line: float | None = None):
    offense_team, defense = validate(offense_team, defense)
    d = stats()
    w = column(d, ['week'])
    oc = column(d, ['opponent_team', 'defteam', 'opponent'])
    if not w:
        raise HTTPException(500, 'Week column missing.')
    q = player(d, qb, offense_team)
    r = player(d, receiver, offense_team)
    if q.empty or r.empty:
        raise HTTPException(404, 'Selected QB or receiver has no current-season statistics.')
    ac = column(q, ['attempts', 'passing_attempts'])
    cc = column(q, ['completions', 'passing_completions'])
    tc = column(r, ['targets'])
    rc = column(r, ['receptions'])
    yc = column(r, ['receiving_yards', 'rec_yards'])
    if not all([ac, cc, tc, rc, yc]):
        raise HTTPException(500, 'Receiving/passing stat columns missing.')
    q['_a'] = numeric(q, ac)
    q['_c'] = numeric(q, cc)
    r['_t'] = numeric(r, tc)
    r['_r'] = numeric(r, rc)
    r['_y'] = numeric(r, yc)
    qw = q.groupby(w).agg(attempts=('_a', 'sum'), completions=('_c', 'sum')).reset_index()
    rw = r.groupby(w).agg(targets=('_t', 'sum'), receptions=('_r', 'sum'), yards=('_y', 'sum')).reset_index()
    matched = qw.merge(rw, on=w)
    matched = matched[matched.attempts > 0]
    share = avg((matched.targets / matched.attempts).tolist()) if not matched.empty else 0
    catch = r['_r'].sum() / r['_t'].sum() if r['_t'].sum() > 0 else 0
    qa = avg(qw.attempts.tolist())
    qc = avg(qw.completions.tolist())
    recent = avg(rw.yards.tolist())
    dypc = 10.5
    dg = 0
    if oc:
        z = d[d['_opp'] == defense].copy()
        z = z[z['_pos'] == 'QB']
        py = column(z, ['passing_yards', 'pass_yards'])
        pcomp = column(z, ['completions', 'passing_completions'])
        if py and pcomp:
            z['_y'] = numeric(z, py)
            z['_c'] = numeric(z, pcomp)
            zw = z.groupby(w).agg(y=('_y', 'sum'), c=('_c', 'sum')).reset_index()
            zw = zw[zw.c > 0]
            dg = len(zw)
            if dg:
                dypc = avg((zw.y / zw.c).tolist())
    targets = qa * share
    usage_receptions = targets * catch
    original = targets * dypc
    receptions_trend = trend(rw.receptions.tolist())
    yards_trend = trend(rw.yards.tolist())
    receptions = .55 * usage_receptions + .45 * receptions_trend['weighted']
    base_yards = .55 * (receptions * dypc) + .45 * yards_trend['weighted']
    position_col = column(r, ['position', 'player_position'])
    position = str(r.iloc[-1][position_col]).upper() if position_col else 'WR'
    position = position if position in ('WR', 'TE') else 'WR'
    usage_factor, usage_note = usage_adjustment(rw.targets.tolist())
    defense_factor, pos_games, pos_allowed, league_allowed = position_defense(
        d, defense, position, ['receiving_yards', 'rec_yards'], w, oc)
    improved = max(0, base_yards * usage_factor * (1 + .5 * (defense_factor - 1)))
    reasons = [usage_note]
    if pos_games:
        reasons.append(('Favorable' if defense_factor > 1.015 else 'Tough' if defense_factor < .985 else 'Neutral')
                       + ' matchup vs ' + position)
    else:
        reasons.append('Position matchup data limited')
    ye = improved - line if line is not None else None
    re_ = receptions - receptions_line if receptions_line is not None else None
    return {'offense_team': offense_team, 'quarterback': qb, 'receiver': receiver, 'defense': defense,
            'opponent_name': NFL_TEAMS[defense], 'season': season(),
            'improved_projection': round(improved, 2), 'original_projection': round(original, 2),
            'qb_attempts_weighted': round(qa, 2), 'qb_completions_weighted': round(qc, 2),
            'target_share': round(share * 100, 2), 'catch_rate': round(catch * 100, 2),
            'receiver_recent_yards_average': round(recent, 2), 'projected_targets': round(targets, 2),
            'projected_receptions': round(receptions, 2), 'yards_per_completion_allowed': round(dypc, 2),
            'matched_games': len(matched), 'defense_games_sampled': dg, 'quick_factors': reasons,
            'usage_adjustment_pct': round((usage_factor - 1) * 100, 1),
            'position_matchup_adjustment_pct': round((defense_factor - 1) * 50 * 100, 1),
            'position_defense_yards_allowed': pos_allowed, 'league_position_yards_allowed': league_allowed,
            'position_defense_games': pos_games,
            'recent_form': {'receiving_yards': yards_trend, 'receptions': receptions_trend},
            'confidence': {'receiving_yards': confidence(rw.yards.tolist(), dg),
                           'receptions': confidence(rw.receptions.tolist(), dg)},
            'sportsbook': sportsbook if sportsbook in SPORTSBOOKS else 'Other',
            'line': line, 'improved_edge': round(ye, 2) if ye is not None else None, 'lean': lean(ye),
            'receptions_line': receptions_line,
            'receptions_edge': round(re_, 2) if re_ is not None else None, 'receptions_lean': lean(re_)}


@app.get('/api/project/rushing')
def rushing(offense_team: str, running_back: str, defense: str, sportsbook: str = 'Other',
            yards_line: float | None = None, attempts_line: float | None = None,
            receiving_yards_line: float | None = None, receptions_line: float | None = None):
    offense_team, defense = validate(offense_team, defense)
    d = stats()
    w = column(d, ['week'])
    tc = column(d, ['recent_team', 'team', 'team_abbr', 'posteam'])
    oc = column(d, ['opponent_team', 'defteam', 'opponent'])
    cc = column(d, ['carries', 'rushing_attempts', 'rush_attempts'])
    yc = column(d, ['rushing_yards', 'rushing_yards_gained', 'rush_yards'])
    if not all([w, tc, cc, yc]):
        raise HTTPException(500, 'Rushing data columns missing.')
    r = player(d, running_back, offense_team)
    if r.empty:
        raise HTTPException(404, 'No current-season stats for selected running back.')
    r['_c'] = numeric(r, cc)
    r['_y'] = numeric(r, yc)
    rw = r.groupby(w).agg(c=('_c', 'sum'), y=('_y', 'sum')).reset_index()
    rw = rw[rw.c > 0]
    if rw.empty:
        raise HTTPException(404, 'No rushing attempts for this player this season.')
    rw['ypc'] = rw.y / rw.c
    team = d[d['_team'] == offense_team].copy()
    team['_c'] = numeric(team, cc)
    tw = team.groupby(w).agg(team_c=('_c', 'sum')).reset_index()
    tw = tw[tw.team_c > 0]
    joined = rw[[w, 'c']].merge(tw, on=w)
    share = avg((joined.c / joined.team_c).tolist()) if not joined.empty else 0
    teamavg = avg(tw.team_c.tolist())
    attempts = teamavg * share
    player_ypc = avg(rw.ypc.tolist())
    def_ypc = 4.3
    dg = 0
    if oc:
        z = d[d['_opp'] == defense].copy()
        z['_c'] = numeric(z, cc)
        z['_y'] = numeric(z, yc)
        zw = z.groupby(w).agg(c=('_c', 'sum'), y=('_y', 'sum')).reset_index()
        zw = zw[zw.c > 0]
        dg = len(zw)
        if dg:
            def_ypc = avg((zw.y / zw.c).tolist())
    attempts_trend = trend(rw.c.tolist())
    yards_trend = trend(rw.y.tolist())
    attempts = .55 * attempts + .45 * attempts_trend['weighted']
    blended = (player_ypc + def_ypc) / 2
    original = attempts * blended
    usage_factor, usage_note = usage_adjustment(rw.c.tolist())
    defense_factor, pos_games, pos_allowed, league_allowed = position_defense(
        d, defense, 'RB', ['rushing_yards', 'rushing_yards_gained', 'rush_yards'], w, oc)
    # FIX: the old line used ONLY the defense's yards/carry, so every back with the
    # same carries projected identically, and the defense was counted twice (YPC and
    # position factor). Now the back's own (shrunk) efficiency drives yards, and the
    # opponent enters once via the RB position factor.
    shrunk_ypc = (float(rw.y.sum()) + 4.3 * 30) / (float(rw.c.sum()) + 30)
    improved = max(0, attempts * shrunk_ypc * usage_factor * (1 + .5 * (defense_factor - 1)))
    reasons = [usage_note]
    if pos_games:
        reasons.append(('Favorable' if defense_factor > 1.015 else 'Tough' if defense_factor < .985 else 'Neutral')
                       + ' matchup vs RB')
    else:
        reasons.append('Position matchup data limited')
    target_col = column(r, ['targets'])
    reception_col = column(r, ['receptions'])
    receiving_col = column(r, ['receiving_yards', 'rec_yards'])
    if not all([target_col, reception_col, receiving_col]):
        raise HTTPException(500, 'RB receiving stat columns missing.')
    rec = r.copy()
    rec['_rt'] = numeric(rec, target_col)
    rec['_rr'] = numeric(rec, reception_col)
    rec['_ry'] = numeric(rec, receiving_col)
    rec = rec.groupby(w).agg(targets=('_rt', 'sum'), receptions=('_rr', 'sum'), yards=('_ry', 'sum')).reset_index()
    rec = rec.sort_values(w)
    defense_rec = rb_receiving_defense(d, defense, w, oc)
    rb_targets = rec.targets.tolist()
    rb_receptions = rec.receptions.tolist()
    rb_yards = rec.yards.tolist()
    target_trend = trend(rb_targets)
    reception_trend = trend(rb_receptions)
    receiving_yards_trend = trend(rb_yards)
    target_factor, target_note = usage_adjustment(rb_targets)
    projected_targets = max(0, target_trend['weighted'] * target_factor)
    player_catch = sum(rb_receptions) / sum(rb_targets) if sum(rb_targets) > 0 else 0
    defense_catch = (defense_rec['receptions'] / defense_rec['targets']
                     if defense_rec['targets'] and defense_rec['receptions'] is not None else None)
    expected_catch = (.8 * player_catch + .2 * defense_catch) if defense_catch is not None else player_catch
    expected_catch = max(0, min(1, expected_catch))
    base_receptions = .6 * (projected_targets * expected_catch) + .4 * reception_trend['weighted']

    def rec_matchup(metric, league):
        if metric is None or league is None or league <= 0:
            return 1.0
        raw = matchup_factor(metric, league, .8, 1.2)
        return 1 + (.35 * (raw - 1) * defense_rec['games'] / (defense_rec['games'] + 4))
    reception_factor = rec_matchup(defense_rec['receptions'], defense_rec['league_receptions'])
    projected_rb_receptions = max(0, base_receptions * reception_factor)
    player_ypr = sum(rb_yards) / sum(rb_receptions) if sum(rb_receptions) > 0 else 0
    defense_ypr = (defense_rec['yards'] / defense_rec['receptions']
                   if defense_rec['receptions'] and defense_rec['yards'] is not None else None)
    expected_ypr = (.8 * player_ypr + .2 * defense_ypr) if defense_ypr is not None else player_ypr
    projected_rb_receiving_yards = max(0, .6 * (projected_rb_receptions * expected_ypr)
                                       + .4 * receiving_yards_trend['weighted'])
    rb_receiving_reasons = [target_note,
                            'Opponent RB receptions allowed: ' + str(defense_rec['receptions']) + ' per game'
                            if defense_rec['receptions'] is not None else 'RB receiving defense data limited']
    rb_receiving_confidence = {'receiving_yards': confidence(rb_yards, defense_rec['games']),
                               'receptions': confidence(rb_receptions, defense_rec['games'])}
    ye = improved - yards_line if yards_line is not None else None
    ae = attempts - attempts_line if attempts_line is not None else None
    return {'offense_team': offense_team, 'running_back': running_back, 'defense': defense,
            'opponent_name': NFL_TEAMS[defense], 'season': season(),
            'projected_attempts': round(attempts, 2), 'projected_rushing_yards': round(improved, 2),
            'improved_projected_rushing_yards': round(improved, 2),
            'original_projected_rushing_yards': round(original, 2),
            'player_attempts_average': round(avg(rw.c.tolist()), 2),
            'player_rushing_yards_average': round(avg(rw.y.tolist()), 2),
            'player_yards_per_carry': round(player_ypc, 2), 'team_attempts_average': round(teamavg, 2),
            'carry_share': round(share * 100, 2), 'defense_yards_per_attempt_allowed': round(def_ypc, 2),
            'adjusted_yards_per_carry': round(blended, 2), 'player_games_sampled': len(rw),
            'defense_games_sampled': dg, 'quick_factors': reasons,
            'usage_adjustment_pct': round((usage_factor - 1) * 100, 1),
            'position_matchup_adjustment_pct': round((defense_factor - 1) * 50 * 100, 1),
            'position_defense_yards_allowed': pos_allowed, 'league_position_yards_allowed': league_allowed,
            'position_defense_games': pos_games,
            'recent_form': {'rushing_yards': yards_trend, 'rushing_attempts': attempts_trend},
            'confidence': {'rushing_yards': confidence(rw.y.tolist(), dg),
                           'rushing_attempts': confidence(rw.c.tolist(), dg)},
            'rb_projected_targets': round(projected_targets, 2),
            'rb_receiving_yards_projection': round(projected_rb_receiving_yards, 2),
            'rb_receptions_projection': round(projected_rb_receptions, 2),
            'rb_receiving_yards_average': round(avg(rb_yards), 2),
            'rb_receptions_average': round(avg(rb_receptions), 2),
            'rb_targets_average': round(avg(rb_targets), 2), 'rb_catch_rate': round(player_catch * 100, 2),
            'rb_yards_per_reception': round(player_ypr, 2), 'rb_receiving_defense': defense_rec,
            'rb_receiving_reasons': rb_receiving_reasons, 'rb_receiving_confidence': rb_receiving_confidence,
            'rb_receiving_recent_form': {'targets': target_trend, 'receptions': reception_trend,
                                         'receiving_yards': receiving_yards_trend},
            'sportsbook': sportsbook if sportsbook in SPORTSBOOKS else 'Other',
            'yards_line': yards_line, 'yards_edge': round(ye, 2) if ye is not None else None,
            'yards_lean': lean(ye), 'attempts_line': attempts_line,
            'attempts_edge': round(ae, 2) if ae is not None else None, 'attempts_lean': lean(ae)}


def qb_defensive_rates(df, defense, week_col, opponent_col):
    cols = {
        'passing_yards': column(df, ['passing_yards', 'pass_yards']),
        'passing_attempts': column(df, ['attempts', 'passing_attempts']),
        'passing_completions': column(df, ['completions', 'passing_completions']),
        'passing_touchdowns': column(df, ['passing_tds', 'passing_touchdowns']),
        'interceptions': column(df, ['interceptions', 'passing_interceptions']),
        'qb_rushing_yards': column(df, ['rushing_yards', 'rush_yards']),
        'qb_rushing_attempts': column(df, ['carries', 'rushing_attempts', 'rush_attempts'])}
    key = ('qbdef',)
    weekly = TABLES.get(key)
    if weekly is None:
        z = df[df['_pos'] == 'QB']
        if z.empty or not week_col or not opponent_col:
            weekly = False
        else:
            z = z.copy()
            for k, c in cols.items():
                z[k] = numeric(z, c)
            weekly = z.groupby(['_opp', '_week'], as_index=False)[list(cols)].sum()
        TABLES[key] = weekly
    if weekly is False:
        return {'games': 0, 'league_games': 0, 'allowed': {}, 'league': {}}
    selected = weekly[weekly['_opp'] == norm_team(defense)]

    def means(frame):
        return {k: round(float(frame[k].mean()), 2) if not frame.empty else None for k in cols}
    return {'games': len(selected), 'league_games': len(weekly), 'allowed': means(selected), 'league': means(weekly)}


@app.get('/api/project/qb')
def project_qb(offense_team: str, quarterback: str, defense: str,
               passing_yards_line: float | None = None, passing_attempts_line: float | None = None,
               passing_completions_line: float | None = None, passing_touchdowns_line: float | None = None,
               interceptions_line: float | None = None, rushing_yards_line: float | None = None,
               rushing_attempts_line: float | None = None):
    offense_team, defense = validate(offense_team, defense)
    df = stats()
    wc = column(df, ['week'])
    oc = column(df, ['opponent_team', 'defteam', 'opponent'])
    if not wc:
        raise HTTPException(500, 'Week column missing.')
    r = player(df, quarterback, offense_team)
    if r.empty:
        raise HTTPException(404, 'No current-season stats for selected quarterback.')
    mapping = {
        'passing_yards': ['passing_yards', 'pass_yards'], 'passing_attempts': ['attempts', 'passing_attempts'],
        'passing_completions': ['completions', 'passing_completions'],
        'passing_touchdowns': ['passing_tds', 'passing_touchdowns'],
        'interceptions': ['interceptions', 'passing_interceptions'],
        'rushing_yards': ['rushing_yards', 'rush_yards'],
        'rushing_attempts': ['carries', 'rushing_attempts', 'rush_attempts']}
    missing = [k for k, v in mapping.items() if not column(r, v)]
    if missing:
        raise HTTPException(503, 'QB statistics unavailable for: ' + ', '.join(missing))
    for k, v in mapping.items():
        r[k + '_model'] = numeric(r, column(r, v))
    weekly = r.groupby(wc, as_index=False)[[k + '_model' for k in mapping]].sum().sort_values(wc)
    weekly = weekly[weekly['passing_attempts_model'] > 0]
    if weekly.empty:
        raise HTTPException(404, 'No passing games recorded for selected quarterback.')
    defense_rates = qb_defensive_rates(df, defense, wc, oc)
    allowed = defense_rates['allowed']
    league = defense_rates['league']
    dg = defense_rates['games']
    trends = {k: trend(weekly[k + '_model'].tolist()) for k in mapping}

    def adjustment(k, limit=.12):
        a = allowed.get(k)
        l = league.get(k)
        if a is None or l is None or l <= 0 or dg < 2:
            return 1.0
        raw = max(1 - limit, min(1 + limit, a / l))
        return 1 + (raw - 1) * dg / (dg + 5) * .65

    def ratio(a, b, default):
        return a / b if b > 0 else default
    attempts_base = trends['passing_attempts']['weighted']
    attempts = max(0, attempts_base * adjustment('passing_attempts'))
    total_att = float(weekly['passing_attempts_model'].sum())
    total_comp = float(weekly['passing_completions_model'].sum())
    total_yds = float(weekly['passing_yards_model'].sum())
    completion_rate = ratio(total_comp, total_att, .65)
    ypa = ratio(total_yds, total_att, 7.0)
    opponent_completion_rate = ratio(allowed.get('passing_completions') or 0, allowed.get('passing_attempts') or 0, completion_rate)
    league_completion_rate = ratio(league.get('passing_completions') or 0, league.get('passing_attempts') or 0, completion_rate)
    completion_shift = max(-.045, min(.045, (opponent_completion_rate - league_completion_rate) * .35)) if dg >= 2 else 0
    expected_completion_rate = max(.40, min(.80, completion_rate + completion_shift))
    completions = max(0, attempts * expected_completion_rate)
    opponent_ypa = ratio(allowed.get('passing_yards') or 0, allowed.get('passing_attempts') or 0, ypa)
    league_ypa = ratio(league.get('passing_yards') or 0, league.get('passing_attempts') or 0, ypa)
    efficiency_factor = max(.90, min(1.10, 1 + .35 * (opponent_ypa / max(league_ypa, .1) - 1))) if dg >= 2 else 1
    passing_yards = max(0, .72 * (attempts * ypa * efficiency_factor) + .28 * trends['passing_yards']['weighted'])
    td_rate = ratio(float(weekly['passing_touchdowns_model'].sum()), total_att, .04)
    int_rate = ratio(float(weekly['interceptions_model'].sum()), total_att, .025)
    league_td_rate = ratio(league.get('passing_touchdowns') or 0, league.get('passing_attempts') or 0, .045)
    league_int_rate = ratio(league.get('interceptions') or 0, league.get('passing_attempts') or 0, .025)
    td_rate = (td_rate * total_att + league_td_rate * 70) / (total_att + 70)
    int_rate = (int_rate * total_att + league_int_rate * 90) / (total_att + 90)
    touchdowns = max(0, attempts * td_rate * adjustment('passing_touchdowns', .15))
    interceptions = max(0, attempts * int_rate * adjustment('interceptions', .15))
    rush_attempts = max(0, trends['rushing_attempts']['weighted'] * adjustment('qb_rushing_attempts'))
    rush_ypa = ratio(float(weekly['rushing_yards_model'].sum()), float(weekly['rushing_attempts_model'].sum()), 4.0)
    rush_yards = max(0, .65 * rush_attempts * rush_ypa * adjustment('qb_rushing_yards') + .35 * trends['rushing_yards']['weighted'])
    projections = {'passing_yards': passing_yards, 'passing_attempts': attempts, 'passing_completions': completions,
                   'passing_touchdowns': touchdowns, 'interceptions': interceptions,
                   'rushing_yards': rush_yards, 'rushing_attempts': rush_attempts}
    lines = {'passing_yards': passing_yards_line, 'passing_attempts': passing_attempts_line,
             'passing_completions': passing_completions_line, 'passing_touchdowns': passing_touchdowns_line,
             'interceptions': interceptions_line, 'rushing_yards': rushing_yards_line,
             'rushing_attempts': rushing_attempts_line}
    for k, v in lines.items():
        if v is not None and (not math.isfinite(v) or v < 0):
            raise HTTPException(400, 'Prop lines must be nonnegative finite numbers.')
    edges = {k: round(projections[k] - v, 2) if v is not None else None for k, v in lines.items()}
    return {'offense_team': offense_team, 'quarterback': quarterback, 'defense': defense,
            'opponent_name': NFL_TEAMS[defense], 'season': season(),
            'projections': {k: round(v, 2) for k, v in projections.items()},
            'recent_form': trends, 'defense_matchup': defense_rates,
            'confidence': {k: confidence(weekly[k + '_model'].tolist(), dg) for k in mapping},
            'lines': lines, 'edges': edges, 'leans': {k: lean(edges[k]) for k in mapping},
            'player_games_sampled': len(weekly), 'defense_games_sampled': dg,
            'notes': ['Passing volume drives completions and yards projections.',
                      'Opponent QB statistics use current-season completed games.',
                      'Touchdowns and interceptions are high-variance estimates, not probabilities.',
                      'QB rushing includes official rushing totals, including kneel-down effects.']}


# ----------------------------------------------------------------------------
# Live odds (DraftKings only by default; see ODDS_BOOKMAKERS)
# ----------------------------------------------------------------------------
# DraftKings only. A bookmakers filter counts like one region, so credit cost is the same as before, but other
# sportsbooks are no longer fetched or shown. Set ODDS_BOOKMAKERS (comma separated keys) to add more.
ODDS_BOOKMAKERS = os.getenv('ODDS_BOOKMAKERS', 'draftkings')


def odds_request(path, params):
    key = os.getenv('ODDS_API_KEY')
    if not key:
        raise HTTPException(503, 'Live odds not configured. Add ODDS_API_KEY to Render environment variables.')
    url = 'https://api.the-odds-api.com/v4/' + path + '?' + urlencode({**params, 'apiKey': key})
    try:
        with urlopen(UrlRequest(url, headers={'User-Agent': 'NFL-Prop-Dashboard/1.0'}), timeout=20) as response:
            return json.loads(response.read())
    except Exception:
        raise HTTPException(502, 'Odds provider request failed. Check market availability and your Odds API plan.')


def american_to_decimal(price):
    price = float(price)
    return 1 + price / 100 if price > 0 else 1 + 100 / abs(price)


def implied_prob(price):
    return 1 / american_to_decimal(price)


def parse_event_odds(data, player_name):
    pn = norm_name(player_name)
    result = {}
    allowed = {b.strip().lower() for b in ODDS_BOOKMAKERS.split(',') if b.strip()}
    for book in data.get('bookmakers', []):
        if str(book.get('key', '')).lower() not in allowed:
            continue
        title = 'DraftKings' if book.get('key', '').lower() == 'draftkings' else (book.get('title') or book.get('key'))
        for market in book.get('markets', []):
            prop = PROP_BY_MARKET.get(market.get('key'))
            if not prop:
                continue
            found = {}
            for o in market.get('outcomes', []):
                if o.get('point') is None or norm_name(o.get('description', '')) != pn:
                    continue
                side = str(o.get('name', '')).lower()
                if side in ('over', 'under'):
                    found.setdefault(o['point'], {})[side] = o.get('price')
            if not found:
                continue

            def closeness(item):
                v = item[1]
                if v.get('over') is None or v.get('under') is None:
                    return 9
                return abs(implied_prob(v['over']) - implied_prob(v['under']))
            point, v = min(found.items(), key=closeness)       # main line = closest to even money
            result.setdefault(title, {})[prop] = {
                'line': point, 'over_price': v.get('over'), 'under_price': v.get('under'),
                'last_update': market.get('last_update') or book.get('last_update')}
    return result


@app.get('/api/live-odds')
def live_odds(request: Request, offense_team: str, defense: str, player: str):
    check_access(request)
    rate_limit(request, 'odds', 40)
    offense_team, defense = validate(offense_team, defense)
    now = datetime.now(timezone.utc)
    item = cache.get('events')
    if item and now < item[0]:
        events = item[1]
    else:
        events = odds_request('sports/americanfootball_nfl/events', {})
        cache['events'] = (now + timedelta(minutes=3), events)
    event = next((e for e in events if {e.get('home_team'), e.get('away_team')} == {NFL_TEAMS[offense_team], NFL_TEAMS[defense]}), None)
    if not event:
        return {'available': False, 'message': 'No upcoming matching NFL game found.',
                'books': {'DraftKings': {}}, 'updated_at': now.isoformat(), 'stale': False}
    event_key = 'event_odds:' + str(event['id'])
    stale = False
    with odds_lock:
        saved = cache.get(event_key)
        if saved and now < saved[0]:
            data, fetched = saved[1], saved[2]
        else:
            try:
                data = odds_request(f"sports/americanfootball_nfl/events/{event['id']}/odds",
                                    {'bookmakers': ODDS_BOOKMAKERS, 'markets': ','.join(MARKETS.values()), 'oddsFormat': 'american'})
                fetched = now
                cache[event_key] = (now + timedelta(minutes=ODDS_CACHE_MINUTES), data, now)
            except HTTPException:
                if not saved or now - saved[2] > timedelta(hours=24):
                    raise
                data, fetched, stale = saved[1], saved[2], True
    books = parse_event_odds(data, player)
    books.setdefault('DraftKings', {})
    available = any(books[b] for b in books)
    return {'available': available, 'books': books, 'event': event.get('commence_time'),
            'updated_at': fetched.isoformat(), 'stale': stale, 'cache_minutes': ODDS_CACHE_MINUTES,
            'message': ('⚠️ Showing cached odds; live refresh failed.' if stale
                        else ('' if available else 'No matching player prop markets currently posted.'))}


# ----------------------------------------------------------------------------
# Probability and expected value for a line
# ----------------------------------------------------------------------------
PROP_MODEL = {   # hint, prior coefficient of variation
    'receiving_yards': ('yards', .75), 'receptions': ('count', .55),
    'rushing_yards': ('yards', .75), 'rushing_attempts': ('count', .38),
    'passing_yards': ('yards', .27), 'passing_attempts': ('count', .18),
    'passing_completions': ('count', .23), 'passing_touchdowns': ('rate', .8), 'interceptions': ('rate', 1.2)}
MARKET_WEIGHT = .30     # how much the posted line is blended into the projection


def _ncdf(x):
    return .5 * (1 + math.erf(x / math.sqrt(2)))


def make_dist(hint, mu, sd):
    mu = max(float(mu), .01)
    if hint == 'yards' or (hint == 'count' and mu >= 12):
        return ('normal', mu, max(sd, .1 * mu, 1.0))
    if hint == 'rate':
        return ('poisson', mu, math.sqrt(mu))
    var = max(sd * sd, mu)
    return ('nbinom', mu, var) if var > mu * 1.02 else ('poisson', mu, math.sqrt(mu))


def dist_cdf(dist, k):
    fam, mu, par = dist
    if k < 0:
        return 0.0
    if fam == 'normal':
        a = _ncdf((-.5 - mu) / par)
        return clamp((_ncdf((k + .5 - mu) / par) - a) / max(1 - a, 1e-9), 0.0, 1.0)
    k = int(min(k, 600))
    total = 0.0
    if fam == 'poisson':
        lp = math.log(mu)
        for i in range(k + 1):
            total += math.exp(-mu + i * lp - math.lgamma(i + 1))
    else:
        r = mu * mu / (par - mu)
        p = r / (r + mu)
        lp, lq = math.log(p), math.log(1 - p)
        for i in range(k + 1):
            total += math.exp(math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) + r * lp + i * lq)
    return min(1.0, total)


def line_probs(dist, line):
    over = 1 - dist_cdf(dist, math.floor(line))
    under = dist_cdf(dist, math.ceil(line) - 1)
    return over, under, max(0.0, 1 - over - under)


def blended_sd(values, mu, prior_cv):
    v = [float(x) for x in values if pd.notna(x) and math.isfinite(float(x))]
    n = len(v)
    prior_var = (max(mu, 1.0) * prior_cv) ** 2
    if n >= 2:
        m = sum(v) / n
        sample_var = sum((x - m) ** 2 for x in v) / (n - 1)
    else:
        sample_var = prior_var
    w = n / (n + 4)
    return math.sqrt(w * sample_var + (1 - w) * prior_var)


def check_price(p, name):
    if p is None:
        return None
    if not math.isfinite(p) or abs(p) < 100:
        raise HTTPException(400, f'{name} must be American odds such as -110 or +120.')
    return float(p)


@app.get('/api/analyze-line')
def analyze_line(request: Request, team: str, player_name: str, prop: str, line: float, projection: float,
                 over_price: float | None = None, under_price: float | None = None, defense_games: int = 0):
    """Probability of OVER/UNDER at a line, de-vigged market odds and expected value."""
    rate_limit(request, 'analyze', 240)
    team = norm_team(team)
    if team not in NFL_TEAMS:
        raise HTTPException(400, 'Invalid team.')
    if prop not in PROP_MODEL:
        raise HTTPException(400, 'Unsupported prop.')
    if not (math.isfinite(line) and line >= 0 and math.isfinite(projection) and projection >= 0):
        raise HTTPException(400, 'Invalid line or projection.')
    over_price = check_price(over_price, 'over_price')
    under_price = check_price(under_price, 'under_price')
    hint, cv = PROP_MODEL[prop]
    p = player(stats(), player_name, team)
    stat_col = column(p, {'receiving_yards': ['receiving_yards', 'rec_yards'], 'receptions': ['receptions'],
                          'rushing_yards': ['rushing_yards', 'rush_yards'],
                          'rushing_attempts': ['carries', 'rushing_attempts', 'rush_attempts'],
                          'passing_yards': ['passing_yards', 'pass_yards'],
                          'passing_attempts': ['attempts', 'passing_attempts'],
                          'passing_completions': ['completions', 'passing_completions'],
                          'passing_touchdowns': ['passing_tds', 'passing_touchdowns'],
                          'interceptions': ['interceptions', 'passing_interceptions']}[prop])
    values = []
    if stat_col is not None and not p.empty:
        values = p.assign(_v=numeric(p, stat_col)).groupby('_week')['_v'].sum().sort_index().tolist()
    sd = blended_sd(values, projection, cv)
    conf = confidence(values, defense_games)
    if hint == 'rate' and conf == 'HIGH':
        conf = 'MEDIUM'
    blended = (1 - MARKET_WEIGHT) * projection + MARKET_WEIGHT * line
    po, pu, pp = line_probs(make_dist(hint, blended, sd), line)
    po, pu = clamp(po, .02, .98), clamp(pu, .02, .98)
    mo = line_probs(make_dist(hint, projection, sd), line)[0]
    notes = []
    if over_price is not None and under_price is not None:
        io, iu = implied_prob(over_price), implied_prob(under_price)
        fair_o = io / (io + iu)
    elif over_price is not None:
        fair_o = clamp(implied_prob(over_price) / 1.045, .03, .97)
        notes.append('Only the over price is posted; the under price is estimated.')
    elif under_price is not None:
        fair_o = 1 - clamp(implied_prob(under_price) / 1.045, .03, .97)
        notes.append('Only the under price is posted; the over price is estimated.')
    else:
        fair_o = .5
        notes.append('No odds provided: assuming -110 on both sides.')
    dec_o = american_to_decimal(over_price if over_price is not None else -110)
    dec_u = american_to_decimal(under_price if under_price is not None else -110)
    ev_o = po * (dec_o - 1) - pu
    ev_u = pu * (dec_u - 1) - po
    side = 'OVER' if ev_o >= ev_u else 'UNDER'
    ev = ev_o if side == 'OVER' else ev_u
    side_prob = po if side == 'OVER' else pu
    side_fair = fair_o if side == 'OVER' else 1 - fair_o
    dec = dec_o if side == 'OVER' else dec_u
    gap = abs(projection - line) / max(line, 1.0)
    edge_pts = side_prob - side_fair
    warnings = []
    if gap > .35:
        warnings.append('Projection is far from the posted line - check injuries and role changes before trusting it.')
    if conf == 'LOW':
        warnings.append('Low data confidence.')
    if ev >= .05 and edge_pts >= .04 and conf in ('HIGH', 'MEDIUM') and gap <= .35:
        rating = 'PLAY'
    elif ev >= .02:
        rating = 'LEAN'
    else:
        rating = 'PASS'
    b = dec - 1
    kelly = (side_prob * b - (1 - side_prob - (pp if side == 'OVER' else pp))) / b if b > 0 else 0
    stake = clamp(.25 * kelly, 0, .015) if rating != 'PASS' else 0
    return {'prop': prop, 'line': line, 'projection': round(projection, 2),
            'blended_projection': round(blended, 2), 'sd': round(sd, 2),
            'prob_over': round(po, 4), 'prob_under': round(pu, 4), 'prob_push': round(pp, 4),
            'model_only_prob_over': round(mo, 4), 'market_fair_over': round(fair_o, 4),
            'side': side, 'side_prob': round(side_prob, 4), 'side_fair': round(side_fair, 4),
            'side_price': (over_price if side == 'OVER' else under_price),
            'edge_pts': round(edge_pts, 4), 'ev': round(ev, 4), 'rating': rating,
            'stake_pct': round(stake * 100, 2), 'confidence': conf, 'games': len(values),
            'warnings': warnings, 'notes': notes}


# ----------------------------------------------------------------------------
# Game lines: spread / total / moneyline context and lean (no credits unless fetching)
# ----------------------------------------------------------------------------
GAME_SD_MARGIN = 13.5      # typical NFL spread of final margins around the line
GAME_SD_TOTAL = 10.5       # typical spread of total points around the line
GAME_MARKET_WEIGHT = .75   # game lines are the sharpest NFL market, so lean on the posted number
HOME_FIELD = 1.5           # points of margin
GAME_CACHE_MINUTES = int(os.getenv('GAME_LINES_CACHE_MINUTES', '15'))
TEAM_BY_NAME = {v: k for k, v in NFL_TEAMS.items()}


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def game_history():
    """One row per team per completed game. nflverse stores spread_line from the HOME team's side:
    positive means the home team was favoured, and home covers when (home - away) > spread_line."""
    sched = schedule_df()
    item = cache.get('game_history')
    if item and item[0] is sched:
        return item[1]
    hs, a_s = column(sched, ['home_score']), column(sched, ['away_score'])
    sl, tl = column(sched, ['spread_line']), column(sched, ['total_line'])
    hr, ar = column(sched, ['home_rest']), column(sched, ['away_rest'])
    rows = []
    if hs and a_s:
        for _, g in sched.iterrows():
            h, a = _num(g[hs]), _num(g[a_s])
            if h is None or a is None or pd.isna(g['_week']):
                continue
            line, total_line = (_num(g[sl]) if sl else None), (_num(g[tl]) if tl else None)
            result = h - a
            total = h + a
            ou = None if total_line is None or total == total_line else int(total > total_line)
            for team, opp, is_home in ((g['_home'], g['_away'], 1), (g['_away'], g['_home'], 0)):
                margin = result if is_home else -result
                cover = None
                if line is not None:
                    cover = (result - line) if is_home else (line - result)
                rows.append({
                    'team': team, 'opp': opp, 'home': is_home, 'week': int(g['_week']),
                    'pf': h if is_home else a, 'pa': a if is_home else h, 'margin': margin,
                    'ats': None if cover is None or cover == 0 else int(cover > 0),
                    'cover_margin': cover, 'over': ou, 'total': total,
                    'rest': _num(g[hr if is_home else ar]) if (hr and ar) else None})
    df = pd.DataFrame(rows, columns=['team', 'opp', 'home', 'week', 'pf', 'pa', 'margin', 'ats',
                                     'cover_margin', 'over', 'total', 'rest'])
    cache['game_history'] = (sched, df)
    return df


def _record(series):
    s = series.dropna()
    w = int((s == 1).sum())
    return {'w': w, 'l': int((s == 0).sum()), 'rate': (w / len(s)) if len(s) else None}


def team_profile(hist, team, before_week, role=None):
    d = hist[(hist['team'] == team) & (hist['week'] < before_week)].sort_values('week')
    last = d.tail(5)
    n = len(d)
    role_games = d[d['home'] == (1 if role == 'home' else 0)] if role else d.iloc[0:0]
    return {
        'games': n,
        'ppg': float(d['pf'].mean()) if n else None, 'papg': float(d['pa'].mean()) if n else None,
        'last5_ppg': float(last['pf'].mean()) if len(last) else None,
        'last5_papg': float(last['pa'].mean()) if len(last) else None,
        'last5_margin': float(last['margin'].mean()) if len(last) else None,
        'ats': _record(d['ats']), 'ats_last5': _record(last['ats']),
        'ats_role': _record(role_games['ats']) if role else None,
        'ou': {'over': int((d['over'] == 1).sum()), 'under': int((d['over'] == 0).sum())},
        'ou_last5': {'over': int((last['over'] == 1).sum()), 'under': int((last['over'] == 0).sum())},
        'last5_total_avg': float(last['total'].mean()) if len(last) else None}


def _shrunk(value, n, league):
    if value is None:
        return league
    w = n / (n + 4)
    return w * value + (1 - w) * league


def game_projection(hist, home, away, week):
    before = hist[hist['week'] < week]
    league = float(before['pf'].mean()) if len(before) else 22.5
    hp = team_profile(hist, home, week, 'home')
    ap = team_profile(hist, away, week, 'away')

    def rating(p, key, last_key):
        if p[key] is None:
            return league
        blend = p[key] * .65 + (p[last_key] if p[last_key] is not None else p[key]) * .35
        return _shrunk(blend, p['games'], league)
    h_off, h_def = rating(hp, 'ppg', 'last5_ppg'), rating(hp, 'papg', 'last5_papg')
    a_off, a_def = rating(ap, 'ppg', 'last5_ppg'), rating(ap, 'papg', 'last5_papg')
    home_pts = league + (h_off - league) + (a_def - league) + HOME_FIELD / 2
    away_pts = league + (a_off - league) + (h_def - league) - HOME_FIELD / 2
    return {'home_pts': round(home_pts, 1), 'away_pts': round(away_pts, 1),
            'margin': round(home_pts - away_pts, 1), 'total': round(home_pts + away_pts, 1),
            'league_ppg': round(league, 1)}, hp, ap


def two_way(mu, sd, line):
    """P(over/home side), P(under/away side), P(push) for a whole-number outcome (points) vs a line."""
    over_min = math.floor(line) + 1
    under_max = math.ceil(line) - 1
    p_over = 1 - _ncdf((over_min - .5 - mu) / sd)
    p_under = _ncdf((under_max + .5 - mu) / sd)
    return p_over, p_under, max(0.0, 1 - p_over - p_under)


def side_ev(p_win, p_push, price):
    dec = american_to_decimal(price)
    return p_win * (dec - 1) - (1 - p_win - p_push)


def grade_side(prices, probs, names):
    """Pick the better side by EV. LEAN needs EV >= 2% and a 2-point edge; there is no PLAY for games."""
    (p1, p2, push), (a, b) = probs, prices
    ev1, ev2 = side_ev(p1, push, a), side_ev(p2, push, b)
    if ev1 >= ev2:
        side, ev, p, price = names[0], ev1, p1, a
    else:
        side, ev, p, price = names[1], ev2, p2, b
    fair = implied_prob(price)
    edge = (p - fair) * 100
    return {'side': side, 'ev': round(ev, 4), 'win_prob': round(p, 4), 'edge_pts': round(edge, 1),
            'rating': 'LEAN' if ev >= .02 and edge >= 2 else 'PASS',
            'ev_by_side': {names[0]: round(ev1, 4), names[1]: round(ev2, 4)}}


def agreement(signals, names):
    votes = [s['direction'] for s in signals if s['direction'] in names]
    count = len(votes)
    best_side = max(names, key=votes.count) if votes else None
    best = votes.count(best_side) if best_side else 0
    tie = votes.count(names[0]) == votes.count(names[1])
    label = 'MIXED'
    if count >= 2 and best >= 2 and not tie:
        label = f'{best}/{count} {best_side}'
    return {'label': label, 'count': count, 'best': best, 'direction': None if tie else best_side}


def game_context(sched_row):
    """Rest, division, roof, temperature and wind where the schedule carries them."""
    def pick(names):
        c = column(sched_row.to_frame().T, names)
        return sched_row[c] if c else None
    roof = pick(['roof'])
    roof = None if roof is None or pd.isna(roof) else str(roof).lower()
    div = pick(['div_game'])
    return {'home_rest': _num(pick(['home_rest'])), 'away_rest': _num(pick(['away_rest'])),
            'divisional': None if div is None or pd.isna(div) else bool(int(div)),
            'roof': roof, 'temp': _num(pick(['temp'])), 'wind': _num(pick(['wind'])),
            'surface': (None if pick(['surface']) is None or pd.isna(pick(['surface'])) else str(pick(['surface'])))}


def find_game(home, away, week):
    sched = schedule_df()
    g = sched[(sched['_home'] == home) & (sched['_away'] == away)]
    if week is not None:
        g = g[g['_week'] == week]
    if g.empty:
        raise HTTPException(404, 'No scheduled game found for those teams.')
    now = pd.Timestamp.now(tz='UTC')
    upcoming = g[g['_kickoff'] > now]
    return (upcoming if not upcoming.empty else g).sort_values('_kickoff').iloc[0]


def ml_win_prob(home_ml, away_ml, model_margin):
    """Moneyline win chance for the home team.
    Start from the market's own no-vig probability, turn it into the margin it implies, blend that margin
    with the model's (market-heavy), and convert back with the same curve. With zero disagreement the
    answer is exactly the market's probability, so a favorite or underdog never looks like value just
    because the margin-to-win-chance curve is a little off."""
    ih, ia = implied_prob(home_ml), implied_prob(away_ml)
    market_home = ih / (ih + ia)
    market_margin = GAME_SD_MARGIN * NormalDist().inv_cdf(min(max(market_home, .001), .999))
    mu = GAME_MARKET_WEIGHT * market_margin + (1 - GAME_MARKET_WEIGHT) * model_margin
    return _ncdf(mu / GAME_SD_MARGIN), market_home


@app.get('/api/game-analysis')
def game_analysis(home: str, away: str, week: int | None = None,
                  spread: float | None = None, total: float | None = None,
                  spread_home_price: float | None = None, spread_away_price: float | None = None,
                  over_price: float | None = None, under_price: float | None = None,
                  home_ml: float | None = None, away_ml: float | None = None):
    """Free (no Odds API credits). `spread` is the HOME team's number as a sportsbook shows it
    (home -3.5 is entered as -3.5; home +2.5 as 2.5)."""
    home, away = validate(home, away)
    for v, n in ((spread_home_price, 'Spread price'), (spread_away_price, 'Spread price'), (over_price, 'Total price'),
                 (under_price, 'Total price'), (home_ml, 'Moneyline'), (away_ml, 'Moneyline')):
        check_price(v, n)
    if spread is not None and not (-40 <= spread <= 40):
        raise HTTPException(400, 'Spread must be between -40 and 40.')
    if total is not None and not (20 <= total <= 90):
        raise HTTPException(400, 'Total must be between 20 and 90.')
    row = find_game(home, away, week)
    wk = int(row['_week'])
    hist = game_history()
    proj, hp, ap = game_projection(hist, home, away, wk)
    ctx = game_context(row)

    hs_col, as_col = column(row.to_frame().T, ['home_score']), column(row.to_frame().T, ['away_score'])
    final = None
    if hs_col and as_col and _num(row[hs_col]) is not None and _num(row[as_col]) is not None:
        final = {'home_score': int(row[hs_col]), 'away_score': int(row[as_col])}

    frame = row.to_frame().T
    sc_, tc_ = column(frame, ['spread_line']), column(frame, ['total_line'])
    sched_lines = {}
    if sc_ and _num(row[sc_]) is not None:
        sched_lines['spread'] = 0.0 - _num(row[sc_])          # back to the sportsbook view of the home number
    if tc_ and _num(row[tc_]) is not None:
        sched_lines['total'] = _num(row[tc_])
    small = min(hp['games'], ap['games']) < 4
    out = {'season': season(), 'week': wk, 'home': home, 'away': away, 'final': final, 'schedule_lines': sched_lines,
           'projection': proj, 'context': ctx, 'profiles': {'home': hp, 'away': ap},
           'small_sample': small, 'spread': None, 'total': None, 'moneyline': None,
           'notes': ['Game lines are the sharpest NFL market. Treat leans as a second opinion, not a signal to bet.']}
    if small:
        out['notes'].append('Fewer than 4 games of data for one team, so projections lean heavily on league averages.')

    w = GAME_MARKET_WEIGHT
    if spread is not None:
        market_margin = -spread
        mu = w * market_margin + (1 - w) * proj['margin']
        p_home, p_away, push = two_way(mu, GAME_SD_MARGIN, -spread)
        a, b = spread_home_price if spread_home_price is not None else -110, spread_away_price if spread_away_price is not None else -110
        g = grade_side((a, b), (p_home, p_away, push), (home, away))
        if small:
            g['rating'] = 'PASS'
        sig = []
        diff = proj['margin'] - market_margin
        sig.append({'name': 'Model', 'direction': home if diff >= 1 else away if diff <= -1 else None,
                    'detail': f"Model margin {home} {proj['margin']:+.1f} vs line {market_margin:+.1f}"})
        h5, a5 = hp['ats_last5']['rate'], ap['ats_last5']['rate']
        nh, na = hp['ats_last5']['w'] + hp['ats_last5']['l'], ap['ats_last5']['w'] + ap['ats_last5']['l']
        d = None
        if h5 is not None and a5 is not None and nh >= 3 and na >= 3:
            d = home if h5 - a5 >= .2 else away if a5 - h5 >= .2 else None
        sig.append({'name': 'Cover form (last 5)', 'direction': d,
                    'detail': f"{home} {hp['ats_last5']['w']}-{hp['ats_last5']['l']} ATS, {away} {ap['ats_last5']['w']}-{ap['ats_last5']['l']} ATS"})
        m5 = None
        if hp['last5_margin'] is not None and ap['last5_margin'] is not None:
            m5 = (hp['last5_margin'] - ap['last5_margin']) / 2 + HOME_FIELD
        sig.append({'name': 'Recent margins', 'direction': None if m5 is None else (home if m5 - market_margin >= 1.5 else away if market_margin - m5 >= 1.5 else None),
                    'detail': '' if m5 is None else f"Last-5 form implies {home} {m5:+.1f} vs line {market_margin:+.1f}"})
        hr_, ar_ = ctx['home_rest'], ctx['away_rest']
        rest_dir = None
        if hr_ is not None and ar_ is not None:
            rest_dir = home if hr_ - ar_ >= 2 else away if ar_ - hr_ >= 2 else None
        sig.append({'name': 'Rest', 'direction': rest_dir,
                    'detail': '' if hr_ is None or ar_ is None else f"{home} {int(hr_)} days, {away} {int(ar_)} days"})
        out['spread'] = {'line': spread, 'home_price': a, 'away_price': b, 'market_margin': market_margin,
                         'blended_margin': round(mu, 1), 'home_cover': round(p_home, 4), 'away_cover': round(p_away, 4),
                         'push': round(push, 4), **g, 'signals': sig, 'agreement': agreement(sig, (home, away))}
        if final:
            res = (final['home_score'] - final['away_score']) + spread
            out['spread']['result'] = 'PUSH' if res == 0 else home if res > 0 else away

    if total is not None:
        mu = w * total + (1 - w) * proj['total']
        p_over, p_under, push = two_way(mu, GAME_SD_TOTAL, total)
        a, b = over_price if over_price is not None else -110, under_price if under_price is not None else -110
        g = grade_side((a, b), (p_over, p_under, push), ('OVER', 'UNDER'))
        if small:
            g['rating'] = 'PASS'
        sig = []
        diff = proj['total'] - total
        sig.append({'name': 'Model', 'direction': 'OVER' if diff >= 1.5 else 'UNDER' if diff <= -1.5 else None,
                    'detail': f"Model total {proj['total']} vs line {total}"})
        ho, ao = hp['ou_last5'], ap['ou_last5']
        games = ho['over'] + ho['under'] + ao['over'] + ao['under']
        over_rate = (ho['over'] + ao['over']) / games if games else None
        sig.append({'name': 'Over/Under form (last 5)',
                    'direction': None if over_rate is None or games < 6 else 'OVER' if over_rate >= .6 else 'UNDER' if over_rate <= .4 else None,
                    'detail': f"{home} {ho['over']}-{ho['under']} O/U, {away} {ao['over']}-{ao['under']} O/U"})
        t5 = [x for x in (hp['last5_total_avg'], ap['last5_total_avg']) if x is not None]
        avg5 = sum(t5) / len(t5) if t5 else None
        sig.append({'name': 'Recent scoring', 'direction': None if avg5 is None else 'OVER' if avg5 - total >= 2 else 'UNDER' if total - avg5 >= 2 else None,
                    'detail': '' if avg5 is None else f"Both teams' last-5 games averaged {avg5:.1f} total points"})
        outdoors = ctx['roof'] in (None, 'outdoors', 'open')
        wind, temp = ctx['wind'], ctx['temp']
        bad = outdoors and ((wind is not None and wind >= 15) or (temp is not None and temp <= 25))
        sig.append({'name': 'Weather', 'direction': 'UNDER' if bad else None,
                    'detail': ('Dome / closed roof' if ctx['roof'] in ('dome', 'closed') else
                               'No weather data yet' if wind is None and temp is None else
                               f"{'' if temp is None else str(int(temp)) + 'F'} {'' if wind is None else str(int(wind)) + ' mph wind'}".strip())})
        out['total'] = {'line': total, 'over_price': a, 'under_price': b, 'blended_total': round(mu, 1),
                        'over': round(p_over, 4), 'under': round(p_under, 4), 'push': round(push, 4),
                        **g, 'signals': sig, 'agreement': agreement(sig, ('OVER', 'UNDER'))}
        if final:
            t = final['home_score'] + final['away_score']
            out['total']['result'] = 'PUSH' if t == total else 'OVER' if t > total else 'UNDER'

    if home_ml is not None and away_ml is not None:
        p_home, market_home = ml_win_prob(home_ml, away_ml, proj['margin'])
        p_away = 1 - p_home
        g = grade_side((home_ml, away_ml), (p_home, p_away, 0.0), (home, away))
        if small:
            g['rating'] = 'PASS'
        out['moneyline'] = {'home_price': home_ml, 'away_price': away_ml, 'home_win': round(p_home, 4),
                            'away_win': round(p_away, 4), 'market_home': round(market_home, 4),
                            'market_away': round(1 - market_home, 4), **g}
        if final:
            out['moneyline']['result'] = home if final['home_score'] > final['away_score'] else away if final['away_score'] > final['home_score'] else 'TIE'
    return out


def parse_game_odds(events):
    games = []
    for e in events:
        home, away = TEAM_BY_NAME.get(e.get('home_team')), TEAM_BY_NAME.get(e.get('away_team'))
        if not home or not away:
            continue
        book = next((b for b in e.get('bookmakers', []) if str(b.get('key', '')).lower() == 'draftkings'), None)
        if not book:
            continue
        g = {'home': home, 'away': away, 'kickoff': e.get('commence_time'), 'updated': book.get('last_update')}
        for m in book.get('markets', []):
            outcomes = m.get('outcomes', [])
            if m.get('key') == 'spreads':
                for o in outcomes:
                    if o.get('name') == e.get('home_team'):
                        g['spread'], g['spread_home_price'] = o.get('point'), o.get('price')
                    elif o.get('name') == e.get('away_team'):
                        g['spread_away_price'] = o.get('price')
            elif m.get('key') == 'totals':
                for o in outcomes:
                    if str(o.get('name')).lower() == 'over':
                        g['total'], g['over_price'] = o.get('point'), o.get('price')
                    elif str(o.get('name')).lower() == 'under':
                        g['under_price'] = o.get('price')
            elif m.get('key') == 'h2h':
                for o in outcomes:
                    if o.get('name') == e.get('home_team'):
                        g['home_ml'] = o.get('price')
                    elif o.get('name') == e.get('away_team'):
                        g['away_ml'] = o.get('price')
        games.append(g)
    return games


@app.get('/api/game-lines')
def game_lines(request: Request):
    """Fast fetch: DraftKings spread, total and moneyline for EVERY upcoming game in one call
    (3 Odds API credits). Locked with the access code and cached to protect credits."""
    check_access(request)
    rate_limit(request, 'gamelines', 10)
    now = datetime.now(timezone.utc)
    stale = False
    with odds_lock:
        saved = cache.get('game_lines')
        if saved and now < saved[0]:
            games, fetched = saved[1], saved[2]
        else:
            try:
                events = odds_request('sports/americanfootball_nfl/odds',
                                      {'bookmakers': 'draftkings', 'markets': 'spreads,totals,h2h', 'oddsFormat': 'american'})
                games, fetched = parse_game_odds(events), now
                cache['game_lines'] = (now + timedelta(minutes=GAME_CACHE_MINUTES), games, now)
            except HTTPException:
                if not saved or now - saved[2] > timedelta(hours=24):
                    raise
                games, fetched, stale = saved[1], saved[2], True
    return {'games': games, 'updated_at': fetched.isoformat(), 'stale': stale, 'cache_minutes': GAME_CACHE_MINUTES,
            'credits_note': 'One call covers every game (about 3 credits).'}


@app.get('/api/injury-status')
def injury_status(team: str, player_name: str):
    team = norm_team(team)
    if team not in NFL_TEAMS:
        raise HTTPException(400, 'Invalid team.')
    week, _ = upcoming_week()
    status, text, verified = injury_status_for(team, player_name, week)
    return {'player': player_name, 'team': team, 'week': week, 'status': status, 'injury': text,
            'verified': verified}


# ----------------------------------------------------------------------------
# Historical hit rates
# ----------------------------------------------------------------------------
HISTORY_STAT_COLUMNS = {
    'receiving_yards': ['receiving_yards', 'rec_yards'], 'receptions': ['receptions'],
    'rushing_yards': ['rushing_yards', 'rush_yards'], 'rushing_attempts': ['carries', 'rushing_attempts', 'rush_attempts'],
    'passing_yards': ['passing_yards', 'pass_yards'], 'passing_attempts': ['attempts', 'passing_attempts'],
    'passing_completions': ['completions', 'passing_completions'],
    'passing_touchdowns': ['passing_tds', 'passing_touchdowns', 'pass_tds'],
    'interceptions': ['interceptions', 'passing_interceptions']}


def hit_rate_window(values, line):
    over = sum(x > line for x in values)
    under = sum(x < line for x in values)
    pushes = len(values) - over - under
    decided = over + under
    return {'games': len(values), 'over': over, 'under': under, 'pushes': pushes, 'decided': decided,
            'over_pct': round(100 * over / decided, 1) if decided else None,
            'under_pct': round(100 * under / decided, 1) if decided else None}


@app.get('/api/historical-hit-rates')
def historical_hit_rates(team: str, player_name: str, prop: str, line: float):
    team = norm_team(team)
    if team not in NFL_TEAMS:
        raise HTTPException(400, 'Invalid team.')
    if prop not in HISTORY_STAT_COLUMNS:
        raise HTTPException(400, 'Unsupported prop.')
    if not math.isfinite(line) or line < 0:
        raise HTTPException(400, 'Invalid line.')
    p = player(stats(), player_name, team)
    if p.empty:
        raise HTTPException(404, 'No completed games for this player.')
    sc = column(p, HISTORY_STAT_COLUMNS[prop])
    if not sc:
        raise HTTPException(503, 'Historical stat field unavailable.')
    p = p.assign(_value=pd.to_numeric(p[sc], errors='coerce')).dropna(subset=['_week', '_value'])
    p = p[p['_value'] >= 0]
    weekly = p.groupby('_week', as_index=False)['_value'].sum().sort_values('_week')
    values = weekly['_value'].astype(float).tolist()
    return {'player': player_name, 'team': team, 'prop': prop, 'line': line,
            'last_3': hit_rate_window(values[-3:], line), 'last_5': hit_rate_window(values[-5:], line),
            'season': hit_rate_window(values, line),
            'note': 'Over/under percentages exclude pushes from the denominator. Only recorded games are counted.'}


# ----------------------------------------------------------------------------
# Top Picks (statistical opportunities; no odds credits)
# ----------------------------------------------------------------------------
WEEKLY_LOCK = Lock()
WEEKLY_CACHE = {}


def weekly_window():
    now = datetime.now(ZoneInfo('Pacific/Honolulu'))
    days = (now.weekday() - 5) % 7
    last_sat = (now - timedelta(days=days)).replace(hour=8, minute=0, second=0, microsecond=0)
    if last_sat > now:
        last_sat -= timedelta(days=7)
    return last_sat, now


def weekly_schedule():
    """Exactly one NFL schedule week: the week of the nearest unplayed game."""
    schedule = schedule_df()
    now = pd.Timestamp.now(tz='UTC')
    upcoming = schedule[(schedule['_kickoff'] > now) & schedule['_week'].notna()].copy()
    if upcoming.empty:
        return [], 0, None
    nearest = upcoming.sort_values('_kickoff').iloc[0]
    target_week = int(nearest['_week'])
    selected = upcoming[upcoming['_week'] == target_week].sort_values('_kickoff')
    games, seen = [], set()
    for _, g in selected.iterrows():
        home, away = g['_home'], g['_away']       # FIX: normalised codes (old replace('LA','LAR') broke LAC/LAR)
        if home not in NFL_TEAMS or away not in NFL_TEAMS or home in seen or away in seen:
            continue
        seen.update([home, away])
        games.extend([(home, away), (away, home)])
    return games, target_week, nearest['_kickoff']


MIN_VOLUME = {'receptions': 2.0, 'receiving_yards': 20.0, 'rushing_yards': 25.0, 'rushing_attempts': 6.0,
              'passing_yards': 140.0, 'passing_attempts': 20.0, 'passing_completions': 12.0}


def weekly_candidates():
    df = stats()
    pos_col = column(df, ['position', 'player_position'])
    week_col = column(df, ['week'])
    opp_col = column(df, ['opponent_team', 'defteam', 'opponent'])
    name_col = column(df, ['player_display_name', 'player_name', 'full_name', 'display_name', 'player'])
    if not all([pos_col, week_col]):
        raise HTTPException(503, 'Weekly player statistics fields unavailable.')
    games, slate_week, next_kickoff = weekly_schedule()
    if not games:
        return [], 0, None
    latest_week = float(df['_week'].max())          # computed once (was recomputed inside the player loop)
    inj = injury_data()
    inj_week_ok = int(slate_week) in inj['weeks']
    inj_df = inj['df']
    candidates = []
    for team, opponent in games:
        roster = df[(df['_team'] == team) & (df['_week'] < slate_week)]
        if roster.empty:
            continue
        injured = {}
        if inj_week_ok:
            x = inj_df[(inj_df['_team'] == team) & (inj_df['_week'] == slate_week)]
            injured = dict(zip(x['_name'], x['_status']))
        for nm, rows in roster.groupby('_name', sort=False):
            rows = rows.sort_values('_week')
            pos = str(rows['_pos'].iloc[-1])
            if pos not in ('QB', 'RB', 'WR', 'TE') or len(rows) < 5:
                continue
            if rows['_week'].max() < latest_week - 1:
                continue
            status = injured.get(nm, 'HEALTHY' if inj_week_ok else 'UNKNOWN')
            if status in ('OUT', 'DOUBTFUL'):
                continue
            if pos == 'QB':
                prop_map = [('passing_yards', ['passing_yards', 'pass_yards']), ('passing_attempts', ['attempts', 'passing_attempts']),
                            ('passing_completions', ['completions', 'passing_completions']),
                            ('rushing_yards', ['rushing_yards', 'rush_yards']),
                            ('rushing_attempts', ['carries', 'rushing_attempts', 'rush_attempts'])]
                usage_col = column(rows, ['attempts', 'passing_attempts'])
            elif pos == 'RB':
                prop_map = [('rushing_yards', ['rushing_yards', 'rush_yards']),
                            ('rushing_attempts', ['carries', 'rushing_attempts', 'rush_attempts']),
                            ('receiving_yards', ['receiving_yards', 'rec_yards']), ('receptions', ['receptions'])]
                usage_col = column(rows, ['carries', 'rushing_attempts', 'rush_attempts'])
            else:
                prop_map = [('receiving_yards', ['receiving_yards', 'rec_yards']), ('receptions', ['receptions'])]
                usage_col = column(rows, ['targets'])
            usage = numeric(rows, usage_col).tolist()
            if sum(usage[-3:]) / min(3, len(usage)) < (8 if pos == 'QB' else 3 if pos == 'RB' else 2):
                continue
            display = str(rows[name_col].iloc[-1]) if name_col else nm
            for prop, cols in prop_map:
                col = column(rows, cols)
                if not col:
                    continue
                values = numeric(rows, col).tolist()
                t = trend(values)
                baseline, projected = t['season'], t['weighted']
                if baseline <= 0 or projected <= 0 or projected < MIN_VOLUME.get(prop, 0):
                    continue
                defense_factor, defense_games = 1.0, 0
                if prop in ('receiving_yards', 'receptions'):
                    metric_cols = ['receiving_yards', 'rec_yards'] if prop == 'receiving_yards' else ['receptions']
                    defense_factor, defense_games, _, _ = position_defense(df, opponent, pos, metric_cols, week_col, opp_col)
                elif pos == 'RB' and prop == 'rushing_yards':
                    defense_factor, defense_games, _, _ = position_defense(df, opponent, 'RB', ['rushing_yards', 'rush_yards'], week_col, opp_col)
                elif pos == 'QB':
                    rates = qb_defensive_rates(df, opponent, week_col, opp_col)
                    k = 'qb_' + prop if prop.startswith('rushing') else prop
                    allowed, league = rates['allowed'].get(k), rates['league'].get(k)
                    defense_games = rates['games']
                    if allowed is not None and league and league > 0 and defense_games >= 3:
                        defense_factor = 1 + .45 * (max(.8, min(1.2, allowed / league)) - 1)
                projected = max(0, projected * (1 + .5 * (defense_factor - 1)))
                relative = projected / baseline - 1
                quality = confidence(values, defense_games)
                if quality == 'LOW':
                    continue
                direction = 'OVER' if relative > 0 else 'UNDER'
                score = round(min(30, abs(relative) * 100) + (5 if quality == 'HIGH' else 2), 1)
                if abs(relative) < .08 or abs(projected - baseline) < .5:
                    continue
                reasons = []
                if direction == 'OVER':
                    if t['last_3'] > baseline * 1.08:
                        reasons.append('Recent production above season average')
                    if defense_factor > 1.025:
                        reasons.append('Opponent allows above-average production to this position')
                else:
                    if t['last_3'] < baseline * .92:
                        reasons.append('Recent production below season average')
                    if defense_factor < .975:
                        reasons.append('Opponent allows below-average production to this position')
                if status == 'QUESTIONABLE':
                    reasons.append('Listed QUESTIONABLE - verify status')
                if not reasons:
                    reasons.append('Recent form and matchup suggest ' + direction + ' the season baseline')
                candidates.append({'player': display, 'position': pos, 'team': team, 'opponent': opponent,
                                   'prop': prop, 'projection': round(projected, 2), 'season_average': round(baseline, 2),
                                   'score': score, 'confidence': quality, 'reasons': reasons, 'direction': direction,
                                   'injury_status': status})
    selected, counts = [], {}
    for direction in ('OVER', 'UNDER'):
        ranked = sorted((x for x in candidates if x['direction'] == direction), key=lambda x: x['score'], reverse=True)
        picked = 0
        for item in ranked:
            key = (item['team'], item['player'])
            if counts.get(key, 0) >= 2:
                continue
            counts[key] = counts.get(key, 0) + 1
            selected.append(item)
            picked += 1
            if picked == 10:
                break
    return selected, len(games) // 2, (slate_week, next_kickoff, inj_week_ok)


@app.get('/api/weekly-top20')
def weekly_top20():
    window, now = weekly_window()
    with WEEKLY_LOCK:
        cached = WEEKLY_CACHE.get('result')
        # FIX: previously cached until the next Saturday, so started games stayed on the list.
        # Now it expires at the next kickoff and rebuilds for the remaining slate.
        if cached and time.time() < cached['expires']:
            return cached['data']
        items, games, meta = weekly_candidates()
        ttl = 900
        injuries_verified = False
        if meta:
            slate_week, kickoff, injuries_verified = meta
            # keep the list until the next kickoff (then it rebuilds for the remaining slate)
            ttl = min(6 * 3600, max(60, kickoff.timestamp() - time.time()))
        expires = time.time() + ttl
        result = {'as_of_hst': now.strftime('%Y-%m-%d %I:%M %p HST'),
                  'weekly_window_hst': window.strftime('%Y-%m-%d %I:%M %p HST'),
                  'games': games, 'items': items, 'uses_odds_credits': False,
                  'injuries_verified': bool(injuries_verified),
                  'note': ('Statistical opportunities, not sportsbook edges. '
                           + ('Players listed OUT/DOUBTFUL on the official report are excluded; verify game-day status.'
                              if injuries_verified else
                              'Official injury report for this week is not published yet - injuries NOT verified.'))}
        WEEKLY_CACHE['result'] = {'expires': expires, 'data': result}
        return result


# ----------------------------------------------------------------------------
# Position-specific touchdowns allowed
# ----------------------------------------------------------------------------
TD_CACHE_MINUTES = 60


def defensive_touchdown_data():
    now = datetime.now(timezone.utc)
    current_season = season()
    saved = cache.get('defensive_touchdowns')
    if saved and saved[0] == current_season and now < saved[1]:
        return saved[2]
    df = stats()
    touchdown_columns = {
        'receiving': column(df, ['receiving_tds', 'receiving_touchdowns']),
        'rushing': column(df, ['rushing_tds', 'rushing_touchdowns']),
        'passing': column(df, ['passing_tds', 'passing_touchdowns'])}
    if not any(touchdown_columns.values()):
        raise HTTPException(503, 'Touchdown columns are unavailable in NFL statistics.')
    d = df[df['_opp'].isin(NFL_TEAMS) & df['_week'].notna()]
    if d.empty:
        raise HTTPException(503, 'No completed NFL games available for TD analysis.')
    all_games = d[['_opp', '_week']].drop_duplicates()
    definitions = {'WR': [('receiving', 'Receiving TDs Allowed')], 'TE': [('receiving', 'Receiving TDs Allowed')],
                   'RB': [('rushing', 'RB Rushing TDs Allowed'), ('receiving', 'RB Receiving TDs Allowed')],
                   'QB': [('passing', 'Passing TDs Allowed'), ('rushing', 'QB Rushing TDs Allowed')]}
    result = {}
    for position, builds in definitions.items():
        rows = d[d['_pos'] == position]
        position_result = {}
        for td_type, label in builds:
            src = touchdown_columns[td_type]
            if src is None:
                position_result[td_type] = {'label': label, 'available': False, 'reason': 'Touchdown statistic unavailable'}
                continue
            totals = rows.assign(_td=numeric(rows, src)).groupby(['_opp', '_week'], as_index=False)['_td'].sum()
            weekly = all_games.merge(totals, on=['_opp', '_week'], how='left').fillna({'_td': 0}).sort_values('_week')
            by_defense = {}
            for defense, games in weekly.groupby('_opp'):
                values = games['_td'].tolist()
                if not values:
                    continue
                by_defense[defense] = {'label': label, 'available': True, 'season_total': int(round(sum(values))),
                                       'per_game': round(sum(values) / len(values), 2),
                                       'last_5_total': int(round(sum(values[-5:]))),
                                       'last_5_per_game': round(sum(values[-5:]) / len(values[-5:]), 2),
                                       'games': len(values), 'last_5_games': min(5, len(values))}
            ranked = sorted(by_defense, key=lambda t: (-by_defense[t]['per_game'], t))
            prev_rate = prev_rank = None
            for index, defense in enumerate(ranked):
                rate = by_defense[defense]['per_game']
                rank = prev_rank if (prev_rate is not None and rate == prev_rate) else index + 1
                by_defense[defense]['nfl_rank'] = rank
                by_defense[defense]['ranked_defenses'] = len(ranked)
                prev_rate, prev_rank = rate, rank
            position_result[td_type] = by_defense
        result[position] = position_result
    cache['defensive_touchdowns'] = (current_season, now + timedelta(minutes=TD_CACHE_MINUTES), result)
    return result


@app.get('/api/defense-touchdowns')
def defense_touchdowns(defense: str, position: str):
    defense = norm_team(defense)
    position = position.upper().strip()
    if defense not in NFL_TEAMS:
        raise HTTPException(400, 'Invalid NFL defense.')
    if position not in ('WR', 'TE', 'RB', 'QB'):
        raise HTTPException(400, 'Position must be WR, TE, RB, or QB.')
    position_data = defensive_touchdown_data().get(position, {})
    result = {}
    for td_type, defenses in position_data.items():
        if not isinstance(defenses, dict):
            continue
        if defenses.get('available') is False:
            result[td_type] = defenses
            continue
        result[td_type] = defenses.get(defense, {'available': False, 'reason': 'No completed games for this defense'})
    return {'season': season(), 'defense': defense, 'defense_name': NFL_TEAMS[defense], 'position': position,
            'touchdowns': result, 'uses_odds_credits': False,
            'ranking_note': 'Rank 1 means the most touchdowns allowed per game to this position.'}


if (BASE_DIR / 'static').is_dir():
    app.mount('/static', StaticFiles(directory=str(BASE_DIR / 'static')), name='static')
