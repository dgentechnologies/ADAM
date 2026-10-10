"""Full stored-plan replication, separate from the scheduler's display snapshot.

The configured robot timezone must match the scheduler's local timezone. Old
naive modification dates are normalized with that explicit zone, never guessed
from the desktop. Apply acknowledgements follow an atomic durable write.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
from zoneinfo import ZoneInfo

import desktop_pairing
import scheduler
from config import SCHEDULE_FILE
from memory_store import save_json

DAYS = ['mon','tue','wed','thu','fri','sat','sun']


def revision(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def stamp(value,zone):
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None: dt=dt.replace(tzinfo=ZoneInfo(zone))
    return dt.astimezone(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')


def snapshot():
    identity=desktop_pairing.identity()
    if not identity or not identity.get('timeZone'):
        raise ValueError('Configure the robot timezone before enabling the data bridge.')
    uid,device,zone=identity['uid'],identity['deviceId'],identity['timeZone']
    documents={}
    for kind in ('todos','schedules'):
        for row in scheduler._store[kind]:
            key='todoId' if kind=='todos' else 'scheduleId'
            path=f'users/{uid}/{kind}/{row["id"]}'
            updated=stamp(row.get('updated_at') or row['created'],zone)
            item={key:row['id'],'createdAt':stamp(row['created'],zone),'updatedAt':updated,
                  'deleted':False,'deletedAt':None,'origin':'pi:'+device,'schemaVersion':1,
                  'deviceIds':row.get('sync_device_ids',[device])}
            if kind=='todos':
                item.update(text=row['text'],done=bool(row.get('done')),due=row.get('due'),
                            doneAt=stamp(row['done_at'],zone) if row.get('done_at') else None)
            else:
                repeat=row.get('repeat')
                item.update(kind=row['kind'],label=row['label'],at=row['at'],timeZone=zone,
                    repeat=[DAYS[d] for d in repeat['weekdays']] if repeat else None,
                    timeOfDay=row.get('time_of_day'),enabled=bool(row.get('enabled',True)),
                    lastFired=None,snoozes=0)
                if row['kind']=='timer':item.update(durationSeconds=row.get('duration_s'),deadline=stamp(row['at'],zone))
            # Preserve accepted canonical desired state and operation identity
            # until the Pi actually changes the stored user-editable fields.
            previous=row.get('sync_canonical')
            if previous and (row.get('updated_at') == previous.get('updatedAt') or row.get('sync_source_revision')==revision({k:v for k,v in row.items() if not k.startswith('sync_') and k not in ('last_fired','snoozes')})):
                item=deepcopy(previous)
            documents[path]=item
    for tomb in scheduler._store.get('tombstones',[]):
        kind='todos' if tomb['kind']=='todo' else 'schedules';key='todoId' if kind=='todos' else 'scheduleId'
        path=f'users/{uid}/{kind}/{tomb["id"]}'
        documents[path]=tomb.get('sync_canonical') or {key:tomb['id'],'deleted':True,'deletedAt':stamp(tomb['deleted_at'],zone),
            'updatedAt':stamp(tomb['deleted_at'],zone),'createdAt':stamp(tomb['deleted_at'],zone),
            'deviceIds':[device],'origin':'pi:'+device,'schemaVersion':1}
    import memory_sync
    documents.update(memory_sync.snapshot())
    return {'ok':True,'schemaVersion':1,'deviceId':device,'uid':uid,'documents':documents,
            'revisions':{path:revision(row) for path,row in documents.items()},
            'execution':{row['id']:{'lastFiredAt':row.get('last_fired'),'snoozes':row.get('snoozes',0)} for row in scheduler._store['schedules']}}


def apply(body):
    current=snapshot();path=body.get('path','');record=body.get('record');identity=desktop_pairing.identity()
    if path.startswith('devices/'):
        import memory_sync
        return memory_sync.apply(body)
    prefix=f'users/{identity["uid"]}/'
    if not path.startswith(prefix) or not isinstance(record,dict) or len(json.dumps(record))>16384:
        raise ValueError('Invalid sync record.')
    parts=path.split('/')
    if len(parts)!=4 or parts[2] not in ('todos','schedules') or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',parts[3]):
        raise ValueError('Invalid sync identity.')
    kind,ident=parts[2:]
    from canonical_contract import validate
    validate(kind,ident,record)
    key='todoId' if kind=='todos' else 'scheduleId'
    if record.get(key)!=ident or record.get('schemaVersion',1)!=1 or type(record.get('deleted')) is not bool:
        raise ValueError('Invalid schema or identity.')
    targets=record.get('deviceIds',[])
    if not isinstance(targets,list) or len(targets)>100 or (targets and identity['deviceId'] not in targets):
        raise ValueError('This record does not target this ADAM.')
    wanted=revision(record);actual=current['revisions'].get(path)
    if actual==wanted: return {'ok':True,'appliedRevision':wanted,'deviceId':identity['deviceId'],'path':path}
    if actual!=body.get('baseRevision'): raise ValueError('The robot changed this record. Sync again before applying.')
    # An ISO offset is mandatory for replication metadata.
    for field in ('updatedAt','createdAt'):
        if datetime.fromisoformat(record[field].replace('Z','+00:00')).tzinfo is None:
            raise ValueError('UTC metadata is required.')
    next_store=deepcopy(scheduler._store)
    existing=next((r for r in next_store[kind] if r['id']==ident),{})
    next_store[kind]=[r for r in next_store[kind] if r['id']!=ident]
    next_store['tombstones']=[r for r in next_store.get('tombstones',[]) if not (r['id']==ident and r['kind']==('todo' if kind=='todos' else 'schedule'))]
    if record['deleted']:
        next_store['tombstones'].append({'id':ident,'kind':'todo' if kind=='todos' else 'schedule',
            'label':'','deleted_at':record['updatedAt'],'sync_canonical':record})
    else:
        row={**existing,'id':ident,'created':record['createdAt'],'updated_at':record['updatedAt'],
             'sync_device_ids':targets,'sync_canonical':record}
        if kind=='todos':
            if not isinstance(record.get('text'),str) or not 1<=len(record['text'].strip())<=2000 or type(record.get('done')) is not bool:
                raise ValueError('Invalid to-do.')
            row.update(text=record['text'],done=record['done'],due=record.get('due'),done_at=record.get('doneAt'))
        else:
            if record.get('timeZone')!=identity['timeZone']:
                raise ValueError('Choose this ADAM’s configured timezone before applying the schedule.')
            # The legacy scheduler executes in local wall time. Refuse ambiguous
            # DST occurrences rather than silently executing twice or at a gap.
            at=datetime.fromisoformat(record['at']);zone=ZoneInfo(identity['timeZone'])
            if at.astimezone().utcoffset() != at.replace(tzinfo=zone).utcoffset():
                raise ValueError('Robot OS timezone differs from its provisioned timezone.')
            if at.tzinfo is not None or at.replace(tzinfo=zone,fold=0).utcoffset()!=at.replace(tzinfo=zone,fold=1).utcoffset():
                raise ValueError('Ambiguous or nonexistent local alarm time.')
            if record.get('kind') not in ('alarm','reminder','timer') or type(record.get('enabled')) is not bool:
                raise ValueError('Invalid schedule.')
            repeat=record.get('repeat')
            if repeat is not None and (not isinstance(repeat,list) or len(repeat)>7 or any(d not in DAYS for d in repeat)):
                raise ValueError('Invalid recurrence.')
            row.update(kind=record['kind'],label=str(record.get('label',''))[:80],at=record['at'],enabled=record['enabled'],
                       repeat={'weekdays':[DAYS.index(d) for d in repeat]} if repeat else None,
                       time_of_day=record.get('timeOfDay') or record['at'][11:16],
                       last_fired=existing.get('last_fired'),snoozes=existing.get('snoozes',0))
            if row['kind']=='timer':
                seconds=record.get('durationSeconds')
                if not isinstance(seconds,(int,float)) or not 0<seconds<=604800 or not record.get('deadline'):
                    raise ValueError('A timer needs its original duration and deadline.')
                deadline=datetime.fromisoformat(record['deadline'].replace('Z','+00:00'))
                if deadline.tzinfo is None or abs((deadline-at.replace(tzinfo=zone)).total_seconds())>=1:
                    raise ValueError('Timer deadline does not match its scheduled time.')
                row['duration_s']=seconds
        row['sync_source_revision']=revision({k:v for k,v in row.items() if not k.startswith('sync_') and k not in ('last_fired','snoozes')})
        next_store[kind].append(row)
    save_json(SCHEDULE_FILE,next_store,strict=True)
    scheduler._store.clear();scheduler._store.update(next_store)
    return {'ok':True,'appliedRevision':wanted,'deviceId':identity['deviceId'],'path':path}
