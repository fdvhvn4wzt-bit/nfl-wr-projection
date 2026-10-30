from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd

app = FastAPI(title='NFL WR Receiving Yard Projection')
BASE = Path(__file__).parent
app.mount('/static', StaticFiles(directory=BASE / 'static'), name='static')

@app.get('/', response_class=HTMLResponse)
def home():
    return (BASE / 'static' / 'index.html').read_text()

@app.get('/api/health')
def health():
    return {'status': 'ok', 'message': 'App is running. Stats load from nflverse when available.'}

@app.get('/api/players')
def players():
    try:
        import nflreadpy as nfl
        # Try current and previous completed seasons so offseason/early-season use works.
        from datetime import datetime
        year = datetime.now().year
        seasons = list(range(year - 2, year + 1))
        data = nfl.load_player_stats(seasons=seasons, summary_level='week').to_pandas()
        cols = [c for c in ['player_display_name', 'player_name', 'recent_team', 'position'] if c in data.columns]
        if not cols:
            raise ValueError('Unexpected player statistics columns returned by nflverse.')
        wr = data[data.get('position', pd.Series(index=data.index, dtype=str)).eq('WR')] if 'position' in data else data
        name_col = 'player_display_name' if 'player_display_name' in wr else ('player_name' if 'player_name' in wr else cols[0])
        names = sorted(wr[name_col].dropna().astype(str).unique().tolist())
        return {'players': names, 'teams': sorted(wr['recent_team'].dropna().astype(str).unique().tolist()) if 'recent_team' in wr else []}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f'Could not load nflverse stats right now: {e}')

@app.get('/api/project')
def project(qb: str, defense: str, receiver: str, line: float | None = None):
    try:
        import nflreadpy as nfl
        from datetime import datetime
        year = datetime.now().year
        seasons = list(range(year - 2, year + 1))
        df = nfl.load_player_stats(seasons=seasons, summary_level='week').to_pandas()
        required = {'player_display_name','position','recent_team','opponent_team','attempts','targets','receptions','receiving_yards','completions'}
        if not required.issubset(df.columns):
    missing = sorted(required - set(df.columns))
    available = sorted(df.columns.tolist())
    raise ValueError(
        f"Missing columns: {missing}. Available columns: {available}"
    )
    missing = sorted(required - set(df.columns))
    available = sorted(df.columns.tolist())
    raise ValueError(
        f"Missing columns: {missing}. Available columns: {available}"
    )
        df = df[df['season_type'].eq('REG')] if 'season_type' in df else df
        def weighted(series):
            series = series.dropna().astype(float)
            if series.empty: return None
            season = series.mean()
            last5 = series.tail(5).mean()
            last3 = series.tail(3).mean()
            return .5*season + .3*last5 + .2*last3
        wr = df[(df.player_display_name.str.lower() == receiver.lower()) & (df.position == 'WR')].sort_values(['season','week'])
        q = df[(df.player_display_name.str.lower() == qb.lower()) & (df.position == 'QB')].sort_values(['season','week'])
        if wr.empty or q.empty: raise ValueError('Could not find that WR or QB in the loaded stats. Check spelling and try again.')
        # Aggregate season/recent game metrics for the selected receiver and QB.
        wr_stats = {'targets': weighted(wr.targets), 'receptions': weighted(wr.receptions), 'yards': weighted(wr.receiving_yards)}
        q_attempts = weighted(q.attempts)
        q_comp = weighted(q.completions)
        catch_rate = (wr_stats['receptions'] / wr_stats['targets']) if wr_stats['targets'] else 0
        target_share = (wr_stats['targets'] / q_attempts) if q_attempts else 0
        defense_rows = df[df.opponent_team.astype(str).str.upper() == defense.upper()]
        # Defensive passing yards allowed per completion, derived from opponent passing rows.
        opp = defense_rows.groupby(['season','week'], as_index=False).agg(yards=('passing_yards','sum'), comps=('completions','sum')) if 'passing_yards' in defense_rows else pd.DataFrame()
        ypc_allowed = (opp.yards.sum()/opp.comps.sum()) if not opp.empty and opp.comps.sum() else 10.5
        projected_receptions = q_comp * target_share * catch_rate
        projection = projected_receptions * ypc_allowed
        result = {'projection_yards': round(projection,1), 'projected_receptions': round(projected_receptions,1), 'target_share': round(target_share,3), 'catch_rate': round(catch_rate,3), 'yards_per_completion_allowed': round(ypc_allowed,2), 'qb_attempts_weighted': round(q_attempts or 0,1), 'qb_completions_weighted': round(q_comp or 0,1)}
        if line is not None:
            result['line'] = line
            result['edge'] = round(projection-line,1)
            result['lean'] = 'OVER' if projection > line else ('UNDER' if projection < line else 'NO EDGE')
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
