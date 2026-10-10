"""Versioned metadata around legacy in-place memory maps; no biometric export."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re
import uuid
import desktop_pairing
import memory_store
from config import MEMORY_FILE, FACE_MEMORY_FILE


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def now():return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')


def source(kind,key):
    raw=(memory_store.memory if kind=='memoryFacts' else memory_store.faces).get(key)
    if raw is None:return None
    if kind=='memoryFacts':return {'category':key,'content':str(raw)}
    return {k:str(raw.get(k,'')) for k in ('name','relationship','notes')}


def _materialize(kind,key,row):
    target=memory_store.memory if kind=='memoryFacts' else memory_store.faces
    updated=deepcopy(target)
    if row['deleted']:updated.pop(key,None)
    elif kind=='memoryFacts':updated[key]=row['content']
    else:updated[key]={**updated.get(key,{}),'name':row['name'],'relationship':row.get('relationship',''),
                      'notes':row.get('notes',''),'last_seen':row['updatedAt']}
    memory_store.save_json(MEMORY_FILE if kind=='memoryFacts' else FACE_MEMORY_FILE,updated,strict=True)
    target.clear();target.update(updated)


def recover():
    state=desktop_pairing.read('memory-sync.json') or {'records':{}}
    pending=state.get('pending')
    if pending:
        _materialize(pending['kind'],pending['key'],pending['record'])
        state['records'][pending['path']]={'kind':pending['kind'],'key':pending['key'],
            'record':pending['record'],'sourceHash':digest(source(pending['kind'],pending['key']))}
        state.pop('pending',None);desktop_pairing.write('memory-sync.json',state)
    return state


def snapshot():
    identity=desktop_pairing.identity()
    if not identity:return {}
    state=recover();records=state['records'];device=identity['deviceId'];changed=False
    for kind,values in [('memoryFacts',memory_store.memory),('memoryPeople',memory_store.faces)]:
        for key in list(values):
            known=next((path for path,entry in records.items() if entry['kind']==kind and entry['key']==key),None)
            if known is None:
                ident=str(uuid.uuid5(uuid.NAMESPACE_URL,identity['hardwareId']+'/'+kind+'/'+key))
                known=f'devices/{device}/{kind}/{ident}'
                records[known]={'key':key,'kind':kind,'sourceHash':'','record':{}}
    if len(records)>2000:raise ValueError('Robot memory exceeds the sync limit.')
    for path,entry in records.items():
        raw=source(entry['kind'],entry['key']);sha=digest(raw)
        if sha==entry['sourceHash']:continue
        person=entry['kind']=='memoryPeople';old=entry['record'];stamp=now()
        row={'personId' if person else 'factId':path.rsplit('/',1)[-1],
             'createdAt':old.get('createdAt',stamp),'updatedAt':stamp,'deleted':raw is None,
             'deletedAt':stamp if raw is None else None,'origin':'pi:'+device,'schemaVersion':1}
        if raw is not None:
            row.update(raw)
            if person:row.update(firstSeen=old.get('firstSeen',stamp),lastSeen=stamp,faceEncodingId=None)
            else:row.update(learnedAt=old.get('learnedAt',stamp),confidence=1,source='conversation')
        entry.update(record=row,sourceHash=sha);changed=True
    if changed:desktop_pairing.write('memory-sync.json',state)
    return {path:deepcopy(entry['record']) for path,entry in records.items()}


def apply(body):
    identity=desktop_pairing.identity();path=body.get('path','');row=body.get('record',{})
    parts=path.split('/')
    if (len(parts)!=4 or parts[:2]!=['devices',identity['deviceId']] or parts[2] not in ('memoryFacts','memoryPeople')
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',parts[3])):raise ValueError('Invalid memory scope.')
    kind,ident=parts[2:];person=kind=='memoryPeople';keyfield='personId' if person else 'factId'
    if row.get(keyfield)!=ident or type(row.get('deleted')) is not bool or len(json.dumps(row))>16384:
        raise ValueError('Invalid memory record.')
    allowed={keyfield,'createdAt','updatedAt','deleted','deletedAt','origin','schemaVersion','operationId'}
    allowed|={'name','relationship','notes','firstSeen','lastSeen','faceEncodingId'} if person else {'category','content','learnedAt','confidence','source'}
    if set(row)-allowed:raise ValueError('Unsupported memory fields.')
    if not row['deleted']:
        for key in (['name'] if person else ['category','content']):
            if not isinstance(row.get(key),str) or not 1<=len(row[key])<=2000:raise ValueError('Invalid memory content.')
    previous=snapshot().get(path)
    wanted=digest(row);actual=digest(previous) if previous else None
    if wanted==actual:return {'ok':True,'path':path,'deviceId':identity['deviceId'],'appliedRevision':wanted}
    if actual!=body.get('baseRevision'):raise ValueError('Memory changed on ADAM; sync again.')
    state=recover();entry=state['records'].get(path)
    key=entry['key'] if entry else ('cloud_'+ident if person else str(row.get('category','fact'))+' ['+ident[:8]+']')
    state['pending']={'path':path,'key':key,'kind':kind,'record':row}
    desktop_pairing.write('memory-sync.json',state);recover()
    return {'ok':True,'path':path,'deviceId':identity['deviceId'],'appliedRevision':wanted}
