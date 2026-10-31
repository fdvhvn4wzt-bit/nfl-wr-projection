from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json, math, os
from threading import Lock
import pandas as pd
import nflreadpy as nfl
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title='NFL Player Prop Projections')
NAMES = '''ARI|Arizona Cardinals;ATL|Atlanta Falcons;BAL|Baltimore Ravens;BUF|Buffalo Bills;CAR|Carolina Panthers;CHI|Chicago Bears;CIN|Cincinnati Bengals;CLE|Cleveland Browns;DAL|Dallas Cowboys;DEN|Denver Broncos;DET|Detroit Lions;GB|Green Bay Packers;HOU|Houston Texans;IND|Indianapolis Colts;JAX|Jacksonville Jaguars;KC|Kansas City Chiefs;LAC|Los Angeles Chargers;LAR|Los Angeles Rams;LV|Las Vegas Raiders;MIA|Miami Dolphins;MIN|Minnesota Vikings;NE|New England Patriots;NO|New Orleans Saints;NYG|New York Giants;NYJ|New York Jets;PHI|Philadelphia Eagles;PIT|Pittsburgh Steelers;SEA|Seattle Seahawks;SF|San Francisco 49ers;TB|Tampa Bay Buccaneers;TEN|Tennessee Titans;WAS|Washington Commanders'''
NFL_TEAMS = dict(x.split('|') for x in NAMES.split(';'))
SPORTSBOOKS = ['DraftKings','Other']
MARKETS = {'receiving_yards':'player_reception_yds','receptions':'player_receptions','rushing_yards':'player_rush_yds','rushing_attempts':'player_rush_attempts'}
cache = {}
odds_lock = Lock()
ODDS_CACHE_MINUTES = 360
def season():
    today=date.today(); return today.year if today.month>=8 else today.year-1
def column(df, choices):
    found={str(c).lower():c for c in df.columns}
    return next((found[x.lower()] for x in choices if x.lower() in found),None)
def numeric(df,c):return pd.to_numeric(df[c],errors='coerce').fillna(0) if c else pd.Series(0.0,index=df.index)
def avg(values):
    v=[float(x) for x in values if pd.notna(x) and math.isfinite(float(x)) and float(x)>=0]
    return (sum(v)/len(v)*.5+sum(v[-5:])/len(v[-5:])*.3+sum(v[-3:])/len(v[-3:])*.2) if v else 0.0
def trend(values):
    """Automatic last-3/last-5/season baseline; handles short samples."""
    v=[float(x) for x in values if pd.notna(x) and math.isfinite(float(x))]
    if not v:return {'last_3':0.0,'last_5':0.0,'season':0.0,'weighted':0.0,'games':0}
    season_mean=sum(v)/len(v)
    last3=sum(v[-3:])/len(v[-3:])
    last5=sum(v[-5:])/len(v[-5:])
    return {'last_3':round(last3,2),'last_5':round(last5,2),'season':round(season_mean,2),
            'weighted':round(.40*last3+.35*last5+.25*season_mean,2),'games':len(v)}
def confidence(values, defense_games=0):
    """Heuristic data-quality label, NOT a probability or betting recommendation."""
    v=[float(x) for x in values if pd.notna(x) and math.isfinite(float(x))]
    n=len(v)
    if n<3:return 'LOW'
    mean=sum(v)/n
    if mean<=0:return 'LOW'
    sd=(sum((x-mean)**2 for x in v)/n)**0.5
    cv=sd/max(abs(mean),1.0)
    # Require a useful player and defense sample and reasonable stability.
    if n>=8 and defense_games>=5 and cv<=0.35:return 'HIGH'
    if n>=5 and defense_games>=3 and cv<=0.65:return 'MEDIUM'
    return 'LOW'

def matchup_factor(defense_rate, league_rate, minimum=.80, maximum=1.20):
    """Constrain small-sample defensive effects so projections remain stable."""
    if league_rate<=0:return 1.0
    return max(minimum,min(maximum,defense_rate/league_rate))
def team_codes(code):
    return ['LAR', 'LA'] if code in ('LAR', 'LA') else [code]

def position_defense(df, opponent, position, metric, week_col, opponent_col):
    """Per-game yards allowed to a position, relative to league average.
    Uses completed weekly player totals, shrinks small samples and caps impact.
    Returns factor, opponent games, opponent average, league average.
    """
    pc=column(df,['position','player_position'])
    yc=column(df,metric)
    if not pc or not yc or not opponent_col or not week_col:
        return 1.0,0,None,None
    z=df[df[pc].astype(str).str.upper()==position].copy()
    if z.empty:return 1.0,0,None,None
    z['_yards']=numeric(z,yc)
    z['_opp']=z[opponent_col].astype(str).str.upper().str.strip().replace({'LA':'LAR'})
    z['_week']=pd.to_numeric(z[week_col],errors='coerce')
    grouped=z.groupby(['_opp','_week'],as_index=False)['_yards'].sum()
    # A defense with no players of this position in the data may have zero
    # allowed; group all game weeks, not just weeks with recorded yardage.
    if grouped.empty:return 1.0,0,None,None
    league=float(grouped['_yards'].mean())
    selected=grouped[grouped['_opp']==('LAR' if opponent=='LA' else opponent)]
    games=len(selected)
    if not games or league<=0:return 1.0,games,None,round(league,2) if league>0 else None
    allowed=float(selected['_yards'].mean())
    raw=matchup_factor(allowed,league,.80,1.20)
    # Small-sample shrinkage; max impact approaches 12% with more games.
    factor=1.0+(raw-1.0)*.60*games/(games+4)
    return factor,games,round(allowed,2),round(league,2)

def usage_adjustment(values):
    """Recent opportunities vs season average, with a conservative cap."""
    v=[float(x) for x in values if pd.notna(x) and math.isfinite(float(x))]
    if len(v)<4:return 1.0,'Limited workload history'
    baseline=sum(v)/len(v)
    if baseline<=0:return 1.0,'Limited workload history'
    recent=sum(v[-3:])/min(3,len(v))
    change=recent/baseline-1.0
    factor=1.0+max(-.10,min(.10,change*.30))
    return factor,('Usage rising' if factor>1.025 else 'Usage falling' if factor<.975 else 'Usage steady')

def lean(e):return 'NO LINE' if e is None else ('OVER' if e>0 else 'UNDER' if e<0 else 'EVEN')
def validate(offense,defense):
    offense=offense.upper().strip(); defense=defense.upper().strip()
    if offense not in NFL_TEAMS or defense not in NFL_TEAMS or offense==defense:raise HTTPException(400,'Select two different valid NFL teams.')
    return offense,defense
def stats():
    now=datetime.now(); s=season(); item=cache.get('stats')
    if item and item[0]==s and now<item[1]:return item[2]
    try:d=nfl.load_player_stats(seasons=[s],summary_level='week').to_pandas()
    except Exception as exc:raise HTTPException(503,f'Unable to load NFL statistics: {exc}')
    if d is None or d.empty:raise HTTPException(503,f'No current-season stats for {s}.')
    sc=column(d,['season']); st=column(d,['season_type'])
    if sc:d=d[pd.to_numeric(d[sc],errors='coerce')==s]
    if st:d=d[d[st].astype(str).str.upper().isin(['REG','REGULAR'])]
    if d.empty:raise HTTPException(503,'No regular-season stats available yet.')
    cache['stats']=(s,now+timedelta(minutes=60),d);return d
def rosters():
    now=datetime.now();s=season();item=cache.get('rosters')
    if item and item[0]==s and now<item[1]:return item[2]
    try:d=nfl.load_rosters(seasons=[s]).to_pandas()
    except Exception as exc:raise HTTPException(503,f'Unable to load NFL rosters: {exc}')
    if d is None or d.empty:raise HTTPException(503,'NFL roster data unavailable.')
    cache['rosters']=(s,now+timedelta(hours=12),d);return d
def player(df,name,team):
    nc=column(df,['player_display_name','player_name','full_name','display_name','player']);tc=column(df,['recent_team','team','team_abbr','posteam']);wc=column(df,['week'])
    if not nc:raise HTTPException(500,'Player-name column missing.')
    r=df[df[nc].astype(str).str.casefold()==name.casefold()].copy()
    if tc:r=r[r[tc].astype(str).str.upper().str.strip().isin(team_codes(team))]
    if wc:r['_sort']=pd.to_numeric(r[wc],errors='coerce');r=r.sort_values('_sort')
    return r
@app.get('/',response_class=HTMLResponse)
def home():return Path('static/index.html').read_text(encoding='utf-8')
@app.get('/api/teams')
def teams():return [{'code':k,'name':v} for k,v in NFL_TEAMS.items()]
@app.get('/api/sportsbooks')
def books():return SPORTSBOOKS
@app.get('/api/health')
def health():return {'status':'ok','season':season(),'live_odds_configured':bool(os.getenv('ODDS_API_KEY'))}
@app.get('/api/roster')
def roster(team:str):
    team=team.upper().strip()
    if team not in NFL_TEAMS:raise HTTPException(400,'Invalid NFL team.')
    d=rosters();tc=column(d,['team','team_abbr','recent_team']);nc=column(d,['player_display_name','player_name','full_name','display_name','player']);pc=column(d,['position','player_position'])
    if not all([tc,nc,pc]):raise HTTPException(500,'Roster columns missing.')
    valid_codes=['LA','LAR'] if team=='LAR' else [team]
    r=d[d[tc].astype(str).str.upper().str.strip().isin(valid_codes)].copy();sc=column(r,['status','roster_status'])
    if sc:r=r[r[sc].astype(str).str.upper().isin(['ACT','ACTIVE','RES','PUP','IR'])]
    r[pc]=r[pc].astype(str).str.upper()
    def listing(positions):return sorted([{'name':str(x[nc]),'position':str(x[pc]),'status':str(x[sc]) if sc else ''} for _,x in r[r[pc].isin(positions)].iterrows()],key=lambda x:x['name'])
    return {'team':team,'quarterbacks':listing(['QB']),'receivers':listing(['WR','TE']),'running_backs':listing(['RB'])}
@app.get('/api/project')
def project(offense_team:str,qb:str,receiver:str,defense:str,sportsbook:str='Other',line:float|None=None,receptions_line:float|None=None):
    offense_team,defense=validate(offense_team,defense);d=stats();w=column(d,['week']);oc=column(d,['opponent_team','defteam','opponent'])
    if not w:raise HTTPException(500,'Week column missing.')
    q=player(d,qb,offense_team);r=player(d,receiver,offense_team)
    if q.empty or r.empty:raise HTTPException(404,'Selected QB or receiver has no current-season statistics.')
    ac=column(q,['attempts','passing_attempts']);cc=column(q,['completions','passing_completions']);tc=column(r,['targets']);rc=column(r,['receptions']);yc=column(r,['receiving_yards','rec_yards'])
    if not all([ac,cc,tc,rc,yc]):raise HTTPException(500,'Receiving/passing stat columns missing.')
    q['_a']=numeric(q,ac);q['_c']=numeric(q,cc);r['_t']=numeric(r,tc);r['_r']=numeric(r,rc);r['_y']=numeric(r,yc)
    qw=q.groupby(w).agg(attempts=('_a','sum'),completions=('_c','sum')).reset_index();rw=r.groupby(w).agg(targets=('_t','sum'),receptions=('_r','sum'),yards=('_y','sum')).reset_index()
    matched=qw.merge(rw,on=w);matched=matched[matched.attempts>0]
    share=avg((matched.targets/matched.attempts).tolist()) if not matched.empty else 0
    catch=r['_r'].sum()/r['_t'].sum() if r['_t'].sum()>0 else 0
    qa=avg(qw.attempts.tolist());qc=avg(qw.completions.tolist());recent=avg(rw.yards.tolist());dypc=10.5;dg=0
    if oc:
        z=d[d[oc].astype(str).str.upper().str.strip().isin(team_codes(defense))].copy();pc=column(z,['position','player_position'])
        if pc:z=z[z[pc].astype(str).str.upper()=='QB']
        py=column(z,['passing_yards','pass_yards']);pcomp=column(z,['completions','passing_completions'])
        if py and pcomp:
            z['_y']=numeric(z,py);z['_c']=numeric(z,pcomp);zw=z.groupby(w).agg(y=('_y','sum'),c=('_c','sum')).reset_index();zw=zw[zw.c>0];dg=len(zw)
            if dg:dypc=avg((zw.y/zw.c).tolist())
    targets=qa*share;usage_receptions=targets*catch;original=targets*dypc
    receptions_trend=trend(rw.receptions.tolist());yards_trend=trend(rw.yards.tolist())
    # A stable automatic blend: recent usage plus observed recent production.
    receptions=.55*usage_receptions+.45*receptions_trend['weighted']
    base_yards=.55*(receptions*dypc)+.45*yards_trend['weighted']
    # Automatic, conservative usage + position-specific matchup adjustment.
    position_col=column(r,['position','player_position'])
    position=str(r.iloc[-1][position_col]).upper() if position_col else 'WR'
    position=position if position in ('WR','TE') else 'WR'
    usage_factor,usage_note=usage_adjustment(rw.targets.tolist())
    defense_factor,pos_games,pos_allowed,league_allowed=position_defense(
        d,defense,position,['receiving_yards','rec_yards'],w,oc)
    # Retain the old improved baseline as the starting point; avoid
    # double-counting full opponent impact (base_yards already uses dypc).
    improved=max(0,base_yards*usage_factor*(1+.5*(defense_factor-1)))
    reasons=[usage_note]
    if pos_games:
        reasons.append(('Favorable' if defense_factor>1.015 else 'Tough' if defense_factor<.985 else 'Neutral')+' matchup vs '+position)
    else:reasons.append('Position matchup data limited')
    ye=improved-line if line is not None else None;re=receptions-receptions_line if receptions_line is not None else None
    return {'offense_team':offense_team,'quarterback':qb,'receiver':receiver,'defense':defense,'opponent_name':NFL_TEAMS[defense],'season':season(),'improved_projection':round(improved,2),'original_projection':round(original,2),'qb_attempts_weighted':round(qa,2),'qb_completions_weighted':round(qc,2),'target_share':round(share*100,2),'catch_rate':round(catch*100,2),'receiver_recent_yards_average':round(recent,2),'projected_targets':round(targets,2),'projected_receptions':round(receptions,2),'yards_per_completion_allowed':round(dypc,2),'matched_games':len(matched),'defense_games_sampled':dg,'quick_factors':reasons,'usage_adjustment_pct':round((usage_factor-1)*100,1),'position_matchup_adjustment_pct':round((defense_factor-1)*50*100,1),'position_defense_yards_allowed':pos_allowed,'league_position_yards_allowed':league_allowed,'position_defense_games':pos_games,'recent_form':{'receiving_yards':yards_trend,'receptions':receptions_trend},'confidence':{'receiving_yards':confidence(rw.yards.tolist(),dg),'receptions':confidence(rw.receptions.tolist(),dg)},'sportsbook':sportsbook if sportsbook in SPORTSBOOKS else 'Other','line':line,'improved_edge':round(ye,2) if ye is not None else None,'lean':lean(ye),'receptions_line':receptions_line,'receptions_edge':round(re,2) if re is not None else None,'receptions_lean':lean(re)}
@app.get('/api/project/rushing')
def rushing(offense_team:str,running_back:str,defense:str,sportsbook:str='Other',yards_line:float|None=None,attempts_line:float|None=None):
    offense_team,defense=validate(offense_team,defense);d=stats();w=column(d,['week']);tc=column(d,['recent_team','team','team_abbr','posteam']);oc=column(d,['opponent_team','defteam','opponent']);cc=column(d,['carries','rushing_attempts','rush_attempts']);yc=column(d,['rushing_yards','rushing_yards_gained','rush_yards'])
    if not all([w,tc,cc,yc]):raise HTTPException(500,'Rushing data columns missing.')
    r=player(d,running_back,offense_team)
    if r.empty:raise HTTPException(404,'No current-season stats for selected running back.')
    r['_c']=numeric(r,cc);r['_y']=numeric(r,yc);rw=r.groupby(w).agg(c=('_c','sum'),y=('_y','sum')).reset_index();rw=rw[rw.c>0]
    if rw.empty:raise HTTPException(404,'No rushing attempts for this player this season.')
    rw['ypc']=rw.y/rw.c;team=d[d[tc].astype(str).str.upper().str.strip().isin(team_codes(offense_team))].copy();team['_c']=numeric(team,cc);tw=team.groupby(w).agg(team_c=('_c','sum')).reset_index();tw=tw[tw.team_c>0]
    joined=rw[[w,'c']].merge(tw,on=w);share=avg((joined.c/joined.team_c).tolist()) if not joined.empty else 0
    teamavg=avg(tw.team_c.tolist());attempts=teamavg*share;player_ypc=avg(rw.ypc.tolist());def_ypc=4.3;dg=0
    if oc:
        z=d[d[oc].astype(str).str.upper().str.strip().isin(team_codes(defense))].copy();z['_c']=numeric(z,cc);z['_y']=numeric(z,yc);zw=z.groupby(w).agg(c=('_c','sum'),y=('_y','sum')).reset_index();zw=zw[zw.c>0];dg=len(zw)
        if dg:def_ypc=avg((zw.y/zw.c).tolist())
    attempts_trend=trend(rw.c.tolist());yards_trend=trend(rw.y.tolist())
    attempts=.55*attempts+.45*attempts_trend['weighted']
    blended=(player_ypc+def_ypc)/2;original=attempts*blended
    usage_factor,usage_note=usage_adjustment(rw.c.tolist())
    defense_factor,pos_games,pos_allowed,league_allowed=position_defense(
        d,defense,'RB',['rushing_yards','rushing_yards_gained','rush_yards'],w,oc)
    # Retain the original baseline; modest adjustments only.
    improved=max(0,attempts*def_ypc*usage_factor*(1+.5*(defense_factor-1)))
    reasons=[usage_note]
    if pos_games:reasons.append(('Favorable' if defense_factor>1.015 else 'Tough' if defense_factor<.985 else 'Neutral')+' matchup vs RB')
    else:reasons.append('Position matchup data limited')
    ye=improved-yards_line if yards_line is not None else None;ae=attempts-attempts_line if attempts_line is not None else None
    return {'offense_team':offense_team,'running_back':running_back,'defense':defense,'opponent_name':NFL_TEAMS[defense],'season':season(),'projected_attempts':round(attempts,2),'projected_rushing_yards':round(improved,2),'improved_projected_rushing_yards':round(improved,2),'original_projected_rushing_yards':round(original,2),'player_attempts_average':round(avg(rw.c.tolist()),2),'player_rushing_yards_average':round(avg(rw.y.tolist()),2),'player_yards_per_carry':round(player_ypc,2),'team_attempts_average':round(teamavg,2),'carry_share':round(share*100,2),'defense_yards_per_attempt_allowed':round(def_ypc,2),'adjusted_yards_per_carry':round(blended,2),'player_games_sampled':len(rw),'defense_games_sampled':dg,'quick_factors':reasons,'usage_adjustment_pct':round((usage_factor-1)*100,1),'position_matchup_adjustment_pct':round((defense_factor-1)*50*100,1),'position_defense_yards_allowed':pos_allowed,'league_position_yards_allowed':league_allowed,'position_defense_games':pos_games,'recent_form':{'rushing_yards':yards_trend,'rushing_attempts':attempts_trend},'confidence':{'rushing_yards':confidence(rw.y.tolist(),dg),'rushing_attempts':confidence(rw.c.tolist(),dg)},'sportsbook':sportsbook if sportsbook in SPORTSBOOKS else 'Other','yards_line':yards_line,'yards_edge':round(ye,2) if ye is not None else None,'yards_lean':lean(ye),'attempts_line':attempts_line,'attempts_edge':round(ae,2) if ae is not None else None,'attempts_lean':lean(ae)}

def odds_request(path,params):
    key=os.getenv('ODDS_API_KEY')
    if not key:raise HTTPException(503,'Live odds not configured. Add ODDS_API_KEY to Render environment variables.')
    url='https://api.the-odds-api.com/v4/'+path+'?'+urlencode({**params,'apiKey':key})
    try:
        with urlopen(Request(url,headers={'User-Agent':'NFL-Prop-Dashboard/1.0'}),timeout=20) as response:return json.loads(response.read())
    except Exception:raise HTTPException(502,'Odds provider request failed. Check market availability and your Odds API plan.')
@app.get('/api/live-odds')
def live_odds(offense_team:str,defense:str,player:str):
    offense_team,defense=validate(offense_team,defense);now=datetime.now(timezone.utc);item=cache.get('events')
    if item and now<item[0]:events=item[1]
    else:
        events=odds_request('sports/americanfootball_nfl/events',{});cache['events']=(now+timedelta(minutes=3),events)
    event=next((e for e in events if {e.get('home_team'),e.get('away_team')}=={NFL_TEAMS[offense_team],NFL_TEAMS[defense]}),None)
    if not event:return {'available':False,'message':'No upcoming matching NFL game found.','books':{'DraftKings':{}},'updated_at':now.isoformat()}
    # Cache the full event response, not a single player's props. All players
    # share the same DraftKings response for six hours.
    event_key='event_odds:'+str(event['id'])
    with odds_lock:
        saved=cache.get(event_key)
        if saved and now<saved[0]:
            data=saved[1]
            stale=False
        else:
            try:
                data=odds_request(f"sports/americanfootball_nfl/events/{event['id']}/odds",{'regions':'us','markets':','.join(MARKETS.values()),'oddsFormat':'american','bookmakers':'draftkings'})
                cache[event_key]=(now+timedelta(minutes=ODDS_CACHE_MINUTES),data,now)
                stale=False
            except HTTPException:
                if not saved or now-saved[2]>timedelta(hours=24):raise
                data=saved[1]
                stale=True
    result={'DraftKings':{}}
    for book in data.get('bookmakers',[]):
        name={'draftkings':'DraftKings'}.get(book.get('key','').lower())
        if not name:continue
        for market in book.get('markets',[]):
            prop=next((k for k,v in MARKETS.items() if v==market.get('key')),None)
            if not prop:continue
            for outcome in market.get('outcomes',[]):
                if str(outcome.get('description','')).casefold()==player.casefold() and str(outcome.get('name','')).lower()=='over' and outcome.get('point') is not None:
                    result[name][prop]={'line':outcome['point'],'over_price':outcome.get('price'),'last_update':market.get('last_update') or book.get('last_update')}
    available=any(result[b] for b in result)
    return {'available':available,'books':result,'event':event.get('commence_time'),'updated_at':(cache[event_key][2] if cache.get(event_key) else now).isoformat(),'stale':stale,'cache_minutes':ODDS_CACHE_MINUTES,'message':('â ï¸ Showing cached odds; live refresh failed.' if stale else ('' if available else 'No matching player prop markets currently posted by DraftKings.'))}

app.mount('/static',StaticFiles(directory='static'),name='static')
