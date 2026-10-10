"""Strict protocol-v1 validation before any robot-side persistence."""
from datetime import datetime
import json
import math
import re
from zoneinfo import ZoneInfo
ID = re.compile(r'[A-Za-z0-9_-]{1,128}')
META = {'schemaVersion','createdAt','updatedAt','deleted','deletedAt','origin','operationId'}
KINDS = {
 'todos': ('todoId', {'text','done','due','doneAt','deviceIds'}),
 'schedules': ('scheduleId', {'kind','label','at','timeOfDay','timeZone','repeat','enabled','deviceIds','lastFired','snoozes','durationSeconds','deadline'}),
 'memoryFacts': ('factId', {'category','content','confidence','source','learnedAt'}),
 'memoryPeople': ('personId', {'name','relationship','notes','faceEncodingId','firstSeen','lastSeen'}),
}

def text(value, maximum, empty=False):
    if not isinstance(value,str) or not (0 if empty else 1)<=len(value)<=maximum:
        raise ValueError('Invalid bounded text.')

def utc(value):
    if not isinstance(value,str): raise ValueError('UTC metadata required.')
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None: raise ValueError('UTC offset required.')
    return dt

def validate(kind, ident, row):
    if kind not in KINDS or not isinstance(row,dict) or not ID.fullmatch(ident):
        raise ValueError('Invalid canonical identity.')
    key,fields=KINDS[kind]
    if len(json.dumps(row,allow_nan=False).encode())>16384 or set(row)-(META|fields|{key}):
        raise ValueError('Unsupported or oversized record.')
    if row.get(key)!=ident or (type(row.get('schemaVersion')) is not int or row.get('schemaVersion')!=1) or type(row.get('deleted')) is not bool:
        raise ValueError('Invalid record version or identity.')
    if utc(row['createdAt'])>utc(row['updatedAt']): raise ValueError('Invalid modification time.')
    text(row.get('origin'),128)
    if 'operationId' in row:text(row['operationId'],128)
    if row.get('deletedAt')!=(row['updatedAt'] if row['deleted'] else None):raise ValueError('Invalid tombstone.')
    targets=row.get('deviceIds',[])
    if not isinstance(targets,list) or len(targets)>8 or any(not isinstance(d,str) or not ID.fullmatch(d) for d in targets) or len(set(targets))!=len(targets):raise ValueError('Invalid device targets.')
    if row['deleted']:return
    if kind=='todos':
        text(row.get('text'),2000)
        if type(row.get('done')) is not bool:raise ValueError('Invalid to-do state.')
        if row.get('due') is not None:text(row['due'],32,True)
        if row.get('doneAt') is not None:utc(row['doneAt'])
    elif kind=='schedules':
        text(row.get('label'),80,True);text(row.get('timeZone'),80);ZoneInfo(row['timeZone'])
        if row.get('kind') not in ('alarm','timer','reminder') or type(row.get('enabled')) is not bool:raise ValueError('Invalid schedule.')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?',row.get('at','')):raise ValueError('Full local wall time required.')
        datetime.fromisoformat(row['at'])
        repeat=row.get('repeat')
        if repeat is not None and (not isinstance(repeat,list) or len(repeat)>7 or any(d not in ('mon','tue','wed','thu','fri','sat','sun') for d in repeat) or len(set(repeat))!=len(repeat)):raise ValueError('Invalid recurrence.')
        if row.get('lastFired') is not None or row.get('snoozes',0)!=0:raise ValueError('Execution state is robot-local.')
    elif kind=='memoryFacts':
        text(row.get('category'),80);text(row.get('content'),2000);utc(row['learnedAt'])
        confidence=row.get('confidence')
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1 or row.get('source') not in ('conversation','vision','manual'):raise ValueError('Invalid memory metadata.')
    else:
        text(row.get('name'),80);text(row.get('relationship',''),2000,True);text(row.get('notes',''),2000,True)
        utc(row['firstSeen']);utc(row['lastSeen'])
        if row.get('faceEncodingId') is not None:text(row['faceEncodingId'],128)
