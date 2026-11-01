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
MARKETS = {'receiving_yards':'player_reception_yds','receptions':'player_receptions','rushing_yards':'player_rush_yds','rushing_attempts':'player_rush_attempts','passing_yards':'player_pass_yds','passing_attempts':'player_pass_attempts','passing_completions':'player_pass_completions','passing_touchdowns':'player_pass_tds','interceptions':'player_pass_interceptions'}
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

def rb_receiving_defense(df, opponent, week_col, opponent_col):
    """Per-game RB targets, catches and receiving yards allowed; zeros included."""
    pc=column(df,['position','player_position'])
    tc=column(df,['targets']); rc=column(df,['receptions'])
    yc=column(df,['receiving_yards','rec_yards'])
    if not all([pc,tc,rc,yc,week_col,opponent_col]):
        return {'games':0,'targets':None,'receptions':None,'yards':None,'league_receptions':None,'league_yards':None}
    z=df.copy()
    z['_opp']=z[opponent_col].astype(str).str.upper().str.strip().replace({'LA':'LAR'})
    z['_week']=pd.to_numeric(z[week_col],errors='coerce')
    # All opponent/game combinations, including games where RBs recorded zero.
    games=z[['_opp','_week']].dropna().drop_duplicates()
    rb=z[z[pc].astype(str).str.upper().str.strip()=='RB'].copy()
    rb['_t']=numeric(rb,tc);rb['_r']=numeric(rb,rc);rb['_y']=numeric(rb,yc)
    totals=rb.groupby(['_opp','_week'],as_index=False).agg(targets=('_t','sum'),receptions=('_r','sum'),yards=('_y','sum'))
    all_games=games.merge(totals,on=['_opp','_week'],how='left').fillna({'targets':0,'receptions':0,'yards':0})
    selected=all_games[all_games['_opp']==('LAR' if opponent=='LA' else opponent)]
    def rounded_mean(frame,col):return round(float(frame[col].mean()),2) if not frame.empty else None
    return {'games':len(selected),'targets':rounded_mean(selected,'targets'),
            'receptions':rounded_mean(selected,'receptions'),'yards':rounded_mean(selected,'yards'),
            'league_receptions':rounded_mean(all_games,'receptions'),
            'league_yards':rounded_mean(all_games,'yards')}

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
def rushing(offense_team:str,running_back:str,defense:str,sportsbook:str='Other',yards_line:float|None=None,attempts_line:float|None=None,receiving_yards_line:float|None=None,receptions_line:float|None=None):
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
    # RB receiving props use ALL player weeks, including games with zero catches.
    target_col=column(r,['targets']); reception_col=column(r,['receptions'])
    receiving_col=column(r,['receiving_yards','rec_yards'])
    if not all([target_col,reception_col,receiving_col]):
        raise HTTPException(500,'RB receiving stat columns missing.')
    receiving_weekly=r.copy()
    receiving_weekly['_rt']=numeric(receiving_weekly,target_col)
    receiving_weekly['_rr']=numeric(receiving_weekly,reception_col)
    receiving_weekly['_ry']=numeric(receiving_weekly,receiving_col)
    receiving_weekly=receiving_weekly.groupby(w).agg(targets=('_rt','sum'),receptions=('_rr','sum'),yards=('_ry','sum')).reset_index()
    receiving_weekly=receiving_weekly.sort_values(w)
    defense_rec=rb_receiving_defense(d,defense,w,oc)
    rb_targets=receiving_weekly.targets.tolist()
    rb_receptions=receiving_weekly.receptions.tolist()
    rb_yards=receiving_weekly.yards.tolist()
    target_trend=trend(rb_targets)
    reception_trend=trend(rb_receptions)
    receiving_yards_trend=trend(rb_yards)
    target_factor,target_note=usage_adjustment(rb_targets)
    projected_targets=max(0,target_trend['weighted']*target_factor)
    player_catch=sum(rb_receptions)/sum(rb_targets) if sum(rb_targets)>0 else 0
    # The player's catch rate is blended with the defense's RB catch rate.
    defense_catch=(defense_rec['receptions']/defense_rec['targets']
                   if defense_rec['targets'] and defense_rec['receptions'] is not None else None)
    expected_catch=(.8*player_catch+.2*defense_catch) if defense_catch is not None else player_catch
    expected_catch=max(0,min(1,expected_catch))
    base_receptions=.6*(projected_targets*expected_catch)+.4*reception_trend['weighted']
    def rec_matchup(metric,league):
        if metric is None or league is None or league<=0:return 1.0
        raw=matchup_factor(metric,league,.8,1.2)
        return 1+(.35*(raw-1)*defense_rec['games']/(defense_rec['games']+4))
    reception_factor=rec_matchup(defense_rec['receptions'],defense_rec['league_receptions'])
    projected_rb_receptions=max(0,base_receptions*reception_factor)
    player_ypr=sum(rb_yards)/sum(rb_receptions) if sum(rb_receptions)>0 else 0
    defense_ypr=(defense_rec['yards']/defense_rec['receptions']
                 if defense_rec['receptions'] and defense_rec['yards'] is not None else None)
    expected_ypr=(.8*player_ypr+.2*defense_ypr) if defense_ypr is not None else player_ypr
    # No second matchup multiplier on receiving yards: defensive efficiency
    # has already entered expected yards/reception and projected catches.
    projected_rb_receiving_yards=max(0,.6*(projected_rb_receptions*expected_ypr)+.4*receiving_yards_trend['weighted'])
    rb_receiving_reasons=[target_note,
        'Opponent RB receptions allowed: '+str(defense_rec['receptions'])+' per game' if defense_rec['receptions'] is not None else 'RB receiving defense data limited']
    rb_receiving_confidence={
        'receiving_yards':confidence(rb_yards,defense_rec['games']),
        'receptions':confidence(rb_receptions,defense_rec['games'])}
    ye=improved-yards_line if yards_line is not None else None;ae=attempts-attempts_line if attempts_line is not None else None
    return {'offense_team':offense_team,'running_back':running_back,'defense':defense,'opponent_name':NFL_TEAMS[defense],'season':season(),'projected_attempts':round(attempts,2),'projected_rushing_yards':round(improved,2),'improved_projected_rushing_yards':round(improved,2),'original_projected_rushing_yards':round(original,2),'player_attempts_average':round(avg(rw.c.tolist()),2),'player_rushing_yards_average':round(avg(rw.y.tolist()),2),'player_yards_per_carry':round(player_ypc,2),'team_attempts_average':round(teamavg,2),'carry_share':round(share*100,2),'defense_yards_per_attempt_allowed':round(def_ypc,2),'adjusted_yards_per_carry':round(blended,2),'player_games_sampled':len(rw),'defense_games_sampled':dg,'quick_factors':reasons,'usage_adjustment_pct':round((usage_factor-1)*100,1),'position_matchup_adjustment_pct':round((defense_factor-1)*50*100,1),'position_defense_yards_allowed':pos_allowed,'league_position_yards_allowed':league_allowed,'position_defense_games':pos_games,'recent_form':{'rushing_yards':yards_trend,'rushing_attempts':attempts_trend},'confidence':{'rushing_yards':confidence(rw.y.tolist(),dg),'rushing_attempts':confidence(rw.c.tolist(),dg)},'rb_projected_targets':round(projected_targets,2),'rb_receiving_yards_projection':round(projected_rb_receiving_yards,2),'rb_receptions_projection':round(projected_rb_receptions,2),'rb_receiving_yards_average':round(avg(rb_yards),2),'rb_receptions_average':round(avg(rb_receptions),2),'rb_targets_average':round(avg(rb_targets),2),'rb_catch_rate':round(player_catch*100,2),'rb_yards_per_reception':round(player_ypr,2),'rb_receiving_defense':defense_rec,'rb_receiving_reasons':rb_receiving_reasons,'rb_receiving_confidence':rb_receiving_confidence,'rb_receiving_recent_form':{'targets':target_trend,'receptions':reception_trend,'receiving_yards':receiving_yards_trend},'sportsbook':sportsbook if sportsbook in SPORTSBOOKS else 'Other','yards_line':yards_line,'yards_edge':round(ye,2) if ye is not None else None,'yards_lean':lean(ye),'attempts_line':attempts_line,'attempts_edge':round(ae,2) if ae is not None else None,'attempts_lean':lean(ae)}


# Additional QB projections. Existing WR/TE and RB endpoints are unchanged.
def qb_defensive_rates(df, defense, week_col, opponent_col):
    cols = {
        'passing_yards': column(df,['passing_yards','pass_yards']),
        'passing_attempts': column(df,['attempts','passing_attempts']),
        'passing_completions': column(df,['completions','passing_completions']),
        'passing_touchdowns': column(df,['passing_tds','passing_touchdowns']),
        'interceptions': column(df,['interceptions','passing_interceptions']),
        'qb_rushing_yards': column(df,['rushing_yards','rush_yards']),
        'qb_rushing_attempts': column(df,['carries','rushing_attempts','rush_attempts'])
    }
    pc=column(df,['position','player_position'])
    if not all([week_col,opponent_col,pc]):
        return {'games':0,'league_games':0,'allowed':{},'league':{}}
    z=df[df[pc].astype(str).str.upper().str.strip()=='QB'].copy()
    if z.empty:return {'games':0,'league_games':0,'allowed':{},'league':{}}
    z['_opponent']=z[opponent_col].astype(str).str.upper().str.strip().replace({'LA':'LAR'})
    z['_week']=pd.to_numeric(z[week_col],errors='coerce')
    for k,c in cols.items():z[k]=numeric(z,c)
    weekly=z.groupby(['_opponent','_week'],as_index=False)[list(cols)].sum()
    selected=weekly[weekly['_opponent']==('LAR' if defense=='LA' else defense)]
    def means(frame):return {k:round(float(frame[k].mean()),2) if not frame.empty else None for k in cols}
    return {'games':len(selected),'league_games':len(weekly),'allowed':means(selected),'league':means(weekly)}

@app.get('/api/project/qb')
def project_qb(offense_team:str,quarterback:str,defense:str,
               passing_yards_line:float|None=None,passing_attempts_line:float|None=None,
               passing_completions_line:float|None=None,passing_touchdowns_line:float|None=None,
               interceptions_line:float|None=None,rushing_yards_line:float|None=None,
               rushing_attempts_line:float|None=None):
    offense_team,defense=validate(offense_team,defense)
    df=stats();wc=column(df,['week']);oc=column(df,['opponent_team','defteam','opponent'])
    if not wc:raise HTTPException(500,'Week column missing.')
    r=player(df,quarterback,offense_team)
    if r.empty:raise HTTPException(404,'No current-season stats for selected quarterback.')
    mapping={
        'passing_yards':['passing_yards','pass_yards'],
        'passing_attempts':['attempts','passing_attempts'],
        'passing_completions':['completions','passing_completions'],
        'passing_touchdowns':['passing_tds','passing_touchdowns'],
        'interceptions':['interceptions','passing_interceptions'],
        'rushing_yards':['rushing_yards','rush_yards'],
        'rushing_attempts':['carries','rushing_attempts','rush_attempts']
    }
    missing=[k for k,v in mapping.items() if not column(r,v)]
    if missing:raise HTTPException(503,'QB statistics unavailable for: '+', '.join(missing))
    for k,v in mapping.items():r[k+'_model']=numeric(r,column(r,v))
    weekly=r.groupby(wc,as_index=False)[[k+'_model' for k in mapping]].sum().sort_values(wc)
    weekly=weekly[weekly['passing_attempts_model']>0]
    if weekly.empty:raise HTTPException(404,'No passing games recorded for selected quarterback.')
    defense_rates=qb_defensive_rates(df,defense,wc,oc)
    allowed=defense_rates['allowed'];league=defense_rates['league'];dg=defense_rates['games']
    trends={k:trend(weekly[k+'_model'].tolist()) for k in mapping}
    def adjustment(k,limit=.12):
        a=allowed.get(k);l=league.get(k)
        if a is None or l is None or l<=0 or dg<2:return 1.0
        raw=max(1-limit,min(1+limit,a/l))
        return 1+(raw-1)*dg/(dg+5)*.65
    def ratio(a,b,default):return a/b if b>0 else default
    attempts_base=trends['passing_attempts']['weighted']
    attempts=max(0,attempts_base*adjustment('passing_attempts'))
    total_att=float(weekly['passing_attempts_model'].sum())
    total_comp=float(weekly['passing_completions_model'].sum())
    total_yds=float(weekly['passing_yards_model'].sum())
    completion_rate=ratio(total_comp,total_att,.65)
    ypa=ratio(total_yds,total_att,7.0)
    opponent_completion_rate=ratio(allowed.get('passing_completions') or 0,allowed.get('passing_attempts') or 0,completion_rate)
    league_completion_rate=ratio(league.get('passing_completions') or 0,league.get('passing_attempts') or 0,completion_rate)
    completion_shift=max(-.045,min(.045,(opponent_completion_rate-league_completion_rate)*.35)) if dg>=2 else 0
    expected_completion_rate=max(.40,min(.80,completion_rate+completion_shift))
    completions=max(0,attempts*expected_completion_rate)
    opponent_ypa=ratio(allowed.get('passing_yards') or 0,allowed.get('passing_attempts') or 0,ypa)
    league_ypa=ratio(league.get('passing_yards') or 0,league.get('passing_attempts') or 0,ypa)
    efficiency_factor=max(.90,min(1.10,1+.35*(opponent_ypa/max(league_ypa,.1)-1))) if dg>=2 else 1
    passing_yards=max(0,.72*(attempts*ypa*efficiency_factor)+.28*trends['passing_yards']['weighted'])
    td_rate=ratio(float(weekly['passing_touchdowns_model'].sum()),total_att,.04)
    int_rate=ratio(float(weekly['interceptions_model'].sum()),total_att,.025)
    # Regression toward league-average rates reduces volatility for rare outcomes.
    league_td_rate=ratio(league.get('passing_touchdowns') or 0,league.get('passing_attempts') or 0,.045)
    league_int_rate=ratio(league.get('interceptions') or 0,league.get('passing_attempts') or 0,.025)
    td_rate=(td_rate*total_att+league_td_rate*70)/(total_att+70)
    int_rate=(int_rate*total_att+league_int_rate*90)/(total_att+90)
    touchdowns=max(0,attempts*td_rate*adjustment('passing_touchdowns',.15))
    interceptions=max(0,attempts*int_rate*adjustment('interceptions',.15))
    rush_attempts=max(0,trends['rushing_attempts']['weighted']*adjustment('qb_rushing_attempts'))
    rush_ypa=ratio(float(weekly['rushing_yards_model'].sum()),float(weekly['rushing_attempts_model'].sum()),4.0)
    rush_yards=max(0,.65*rush_attempts*rush_ypa*adjustment('qb_rushing_yards')+.35*trends['rushing_yards']['weighted'])
    projections={'passing_yards':passing_yards,'passing_attempts':attempts,'passing_completions':completions,
                 'passing_touchdowns':touchdowns,'interceptions':interceptions,
                 'rushing_yards':rush_yards,'rushing_attempts':rush_attempts}
    lines={'passing_yards':passing_yards_line,'passing_attempts':passing_attempts_line,
           'passing_completions':passing_completions_line,'passing_touchdowns':passing_touchdowns_line,
           'interceptions':interceptions_line,'rushing_yards':rushing_yards_line,'rushing_attempts':rushing_attempts_line}
    for k,v in lines.items():
        if v is not None and (not math.isfinite(v) or v<0):raise HTTPException(400,'Prop lines must be nonnegative finite numbers.')
    edges={k:round(projections[k]-v,2) if v is not None else None for k,v in lines.items()}
    return {'offense_team':offense_team,'quarterback':quarterback,'defense':defense,
            'opponent_name':NFL_TEAMS[defense],'season':season(),
            'projections':{k:round(v,2) for k,v in projections.items()},
            'recent_form':trends,'defense_matchup':defense_rates,
            'confidence':{k:confidence(weekly[k+'_model'].tolist(),dg) for k in mapping},
            'lines':lines,'edges':edges,'leans':{k:lean(edges[k]) for k in mapping},
            'player_games_sampled':len(weekly),'defense_games_sampled':dg,
            'notes':['Passing volume drives completions and yards projections.',
                     'Opponent QB statistics use current-season completed games.',
                     'Touchdowns and interceptions are high-variance estimates, not probabilities.',
                     'QB rushing includes official rushing totals, including kneel-down effects.']}

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


# Weekly statistical opportunities: no sportsbook API requests.
from zoneinfo import ZoneInfo
WEEKLY_LOCK = Lock()
WEEKLY_CACHE = {}

def weekly_window():
    now = datetime.now(ZoneInfo('Pacific/Honolulu'))
    # A new list becomes eligible at 8:00 AM HST each Saturday.
    days = (now.weekday() - 5) % 7
    last_sat = (now - timedelta(days=days)).replace(hour=8, minute=0, second=0, microsecond=0)
    if last_sat > now: last_sat -= timedelta(days=7)
    return last_sat, now

def weekly_schedule():
    """Select exactly one NFL schedule week, never a rolling nine-day window."""
    try:
        schedule = nfl.load_schedules(seasons=[season()]).to_pandas()
    except Exception as exc:
        raise HTTPException(503, f'Unable to load NFL schedule: {exc}')
    hc=column(schedule,['home_team']); ac=column(schedule,['away_team'])
    dc=column(schedule,['gameday','game_date']); tc=column(schedule,['gametime','game_time'])
    wc=column(schedule,['week']); st=column(schedule,['game_type','season_type'])
    if not all([hc,ac,dc,wc]):
        raise HTTPException(503,'NFL schedule is missing team, date, or week fields.')
    schedule=schedule.copy()
    if st:
        schedule=schedule[schedule[st].astype(str).str.upper().isin(['REG','REGULAR'])].copy()
    # nflverse gametime is Eastern local time. Date-only records cannot be
    # trusted for an accurate upcoming slate, so fail closed instead.
    kickoff=column(schedule,['start_time','game_datetime','kickoff'])
    if kickoff:
        times=pd.to_datetime(schedule[kickoff],errors='coerce',utc=True)
    elif tc:
        local=pd.to_datetime(schedule[dc].astype(str)+' '+schedule[tc].astype(str),errors='coerce')
        times=local.dt.tz_localize('America/New_York',ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    else:
        raise HTTPException(503,'NFL schedule has no kickoff times; refusing to guess matchups.')
    schedule['_kickoff']=times
    schedule['_week']=pd.to_numeric(schedule[wc],errors='coerce')
    now=pd.Timestamp.now(tz='UTC')
    upcoming=schedule[(schedule['_kickoff']>now)&schedule['_week'].notna()].copy()
    if upcoming.empty:return [],0
    # The nearest unplayed game defines the slate week; include only
    # games from that exact week that have not started.
    nearest=upcoming.sort_values('_kickoff').iloc[0]
    target_week=int(nearest['_week'])
    selected=upcoming[upcoming['_week']==target_week].sort_values('_kickoff')
    games=[]; seen=set()
    for _,g in selected.iterrows():
        home=str(g[hc]).upper().strip().replace('LA','LAR')
        away=str(g[ac]).upper().strip().replace('LA','LAR')
        if home not in NFL_TEAMS or away not in NFL_TEAMS:continue
        if home in seen or away in seen:continue
        seen.update([home,away])
        games.extend([(home,away),(away,home)])
    return games,target_week

def weekly_candidates():
    df=stats(); team_col=column(df,['recent_team','team','team_abbr','posteam'])
    name_col=column(df,['player_display_name','player_name','full_name','display_name','player'])
    pos_col=column(df,['position','player_position']); week_col=column(df,['week'])
    if not all([team_col,name_col,pos_col,week_col]):raise HTTPException(503,'Weekly player statistics fields unavailable.')
    games,slate_week=weekly_schedule()
    if not games: return [],0
    candidates=[]
    for team,opponent in games:
        roster=df[df[team_col].astype(str).str.upper().str.strip().isin(team_codes(team))].copy()
        if roster.empty:continue
        roster['_w']=pd.to_numeric(roster[week_col],errors='coerce')
        roster=roster[roster['_w']<slate_week].sort_values('_w')
        if roster.empty:continue
        for name,rows in roster.groupby(name_col,sort=False):
            pos=str(rows[pos_col].iloc[-1]).upper().strip()
            if pos not in ('QB','RB','WR','TE') or len(rows)<5:continue
            # Require current team membership in the most recent recorded week.
            latest_week=float(df[week_col].pipe(pd.to_numeric,errors='coerce').max())
            if rows['_w'].max()<latest_week-1:continue
            if pos=='QB':
                prop_map=[('passing_yards',['passing_yards','pass_yards']),('passing_attempts',['attempts','passing_attempts']),('passing_completions',['completions','passing_completions']),('rushing_yards',['rushing_yards','rush_yards']),('rushing_attempts',['carries','rushing_attempts','rush_attempts'])]
                usage_col=column(rows,['attempts','passing_attempts'])
            elif pos=='RB':
                prop_map=[('rushing_yards',['rushing_yards','rush_yards']),('rushing_attempts',['carries','rushing_attempts','rush_attempts']),('receiving_yards',['receiving_yards','rec_yards']),('receptions',['receptions'])]
                usage_col=column(rows,['carries','rushing_attempts','rush_attempts'])
            else:
                prop_map=[('receiving_yards',['receiving_yards','rec_yards']),('receptions',['receptions'])]
                usage_col=column(rows,['targets'])
            usage=numeric(rows,usage_col).tolist()
            if sum(usage[-3:])/min(3,len(usage))< (8 if pos=='QB' else 3 if pos=='RB' else 2):continue
            for prop,cols in prop_map:
                col=column(rows,cols)
                if not col:continue
                values=numeric(rows,col).tolist()
                t=trend(values)
                baseline=t['season']; projected=t['weighted']
                if baseline<=0 or projected<=0:continue
                # Minimum actionable volumes: don't rank 0.23 receptions.
                minimums={'receptions':2.0,'receiving_yards':20.0,'rushing_yards':25.0,
                          'rushing_attempts':6.0,'passing_yards':140.0,
                          'passing_attempts':20.0,'passing_completions':12.0}
                if projected<minimums.get(prop,0):continue
                # Ranking reflects recent change and opponent position-specific allowance.
                defense_factor=1.0; defense_games=0
                if prop in ('receiving_yards','receptions'):
                    metric_cols=['receiving_yards','rec_yards'] if prop=='receiving_yards' else ['receptions']
                    defense_factor,defense_games,_,_=position_defense(df,opponent,pos,metric_cols,week_col,column(df,['opponent_team','defteam','opponent']))
                elif pos=='RB' and prop=='rushing_yards':
                    defense_factor,defense_games,_,_=position_defense(df,opponent,'RB',['rushing_yards','rush_yards'],week_col,column(df,['opponent_team','defteam','opponent']))
                elif pos=='QB':
                    rates=qb_defensive_rates(df,opponent,week_col,column(df,['opponent_team','defteam','opponent']))
                    k='qb_'+prop if prop.startswith('rushing') else prop
                    allowed=rates['allowed'].get(k); league=rates['league'].get(k)
                    defense_games=rates['games']
                    if allowed is not None and league and league>0 and defense_games>=3:
                        defense_factor=1+.45*(max(.8,min(1.2,allowed/league))-1)
                # Shrink the matchup influence; don't pretend this is a sportsbook edge.
                projected=max(0,projected*(1+.5*(defense_factor-1)))
                relative=(projected/baseline-1)
                # Penalize unstable and small samples; avoid comparing raw yard and catch counts.
                quality=confidence(values,defense_games)
                score=round(max(-30,min(30,relative*100))+(5 if quality=='HIGH' else 2 if quality=='MEDIUM' else -3),1)
                if score<=5 or abs(projected-baseline)<max(.5,baseline*.08):continue
                reasons=[]
                if t['last_3']>baseline*1.08:reasons.append('Recent production above season average')
                if defense_factor>1.025:reasons.append('Opponent allows above-average production to this position')
                if quality=='LOW':reasons.append('Limited or variable data â use caution')
                if not reasons:reasons.append('Recent form and season baseline favor this prop')
                candidates.append({'player':str(name),'position':pos,'team':team,'opponent':opponent,
                    'prop':prop,'projection':round(projected,2),'season_average':round(baseline,2),
                    'score':score,'confidence':quality,'reasons':reasons,'direction':'ABOVE BASELINE'})
    # Prevent one player dominating the entire leaderboard; at most two props each.
    candidates.sort(key=lambda x:x['score'],reverse=True)
    selected=[]; counts={}
    for item in candidates:
        key=(item['team'],item['player'])
        if counts.get(key,0)>=2:continue
        counts[key]=counts.get(key,0)+1
        selected.append(item)
        if len(selected)==20:break
    return selected,len(games)//2

@app.get('/api/weekly-top20')
def weekly_top20():
    window,now=weekly_window()
    key=window.isoformat()
    with WEEKLY_LOCK:
        cached=WEEKLY_CACHE.get(key)
        if cached:return cached
        items,games=weekly_candidates()
        result={'as_of_hst':now.strftime('%Y-%m-%d %I:%M %p HST'),
            'weekly_window_hst':window.strftime('%Y-%m-%d %I:%M %p HST'),
            'games':games,'items':items,'uses_odds_credits':False,
            'injuries_verified':False,
            'note':'Statistical opportunities, not sportsbook edges. Injury and inactive status are NOT verified. Check official reports before using.'}
        # Cache in memory; on Render free restarts, the list will rebuild on next view.
        WEEKLY_CACHE[key]=result
        for old in list(WEEKLY_CACHE):
            if old!=key:del WEEKLY_CACHE[old]
        return result
