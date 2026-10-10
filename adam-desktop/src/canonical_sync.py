"""UID-isolated canonical Firestore cache, durable outbox and conflict journal.

Legacy envelopes are never written here. Unknown fields are retained on reads;
mutations are validated and use document preconditions. Ambiguous simultaneous
edits stay in the journal for explicit resolution instead of disappearing.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import quote
import uuid

import requests
from account import FIREBASE_PROJECT_ID
from secure_store import data_directory

BASE = f'https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents'
ID = re.compile(r'[A-Za-z0-9_-]{1,128}')
DATES = {'createdAt','updatedAt','deletedAt','doneAt','learnedAt','firstSeen','lastSeen'}
LIMIT = 1000


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')


def encode_value(value):
    if value is None: return {'nullValue': None}
    if isinstance(value,bool): return {'booleanValue':value}
    if isinstance(value,int): return {'integerValue':str(value)}
    if isinstance(value,float):
        import math
        if not math.isfinite(value): raise ValueError('Invalid numeric value.')
        return {'doubleValue':value}
    if isinstance(value,str): return {'stringValue':value}
    if isinstance(value,list): return {'arrayValue':{'values':[encode_value(x) for x in value]}}
    if isinstance(value,dict): return {'mapValue':{'fields':{k:encode_value(v) for k,v in value.items()}}}
    raise ValueError('Unsupported cloud value.')


def decode_value(value):
    if not isinstance(value,dict): raise ValueError('Invalid cloud value.')
    if 'nullValue' in value:return None
    for key in ('stringValue','booleanValue','doubleValue','timestampValue'):
        if key in value:return value[key]
    if 'integerValue' in value:return int(value['integerValue'])
    if 'arrayValue' in value:return [decode_value(x) for x in value['arrayValue'].get('values',[])]
    if 'mapValue' in value:return {k:decode_value(v) for k,v in value['mapValue'].get('fields',{}).items()}
    raise ValueError('Unsupported cloud value.')


def fields(data):
    result = {}
    for key, value in data.items():
        result[key] = {'timestampValue': value} if key in DATES and value is not None else encode_value(value)
    return result


def decode(document):
    result = {}
    for key, value in document.get('fields', {}).items():
        result[key] = value['timestampValue'] if 'timestampValue' in value else decode_value(value)
    return result


def validate(path, record, uid, owned):
    parts = path.split('/')
    if len(parts) != 4 or not all(ID.fullmatch(p) for p in parts):
        raise ValueError('Invalid canonical record path.')
    root, owner, kind, ident = parts
    if root == 'users' and owner == uid and kind in ('todos','schedules','notes'):
        pass
    elif root == 'devices' and owner in owned and kind in ('memoryFacts','memoryPeople'):
        pass
    else:
        raise ValueError('This record is outside the current account or device scope.')
    if not isinstance(record, dict) or len(json.dumps(record).encode()) > 16384:
        raise ValueError('Record is too large or invalid.')
    identity_key = {'todos':'todoId','schedules':'scheduleId','notes':'noteId','memoryFacts':'factId','memoryPeople':'personId'}[kind]
    if record.get(identity_key) != ident or type(record.get('deleted')) is not bool:
        raise ValueError('Record identity or deletion state is invalid.')
    for key in ('createdAt','updatedAt'):
        stamp = record.get(key)
        if not isinstance(stamp,str): raise ValueError('UTC record timestamps are required.')
        dt = datetime.fromisoformat(stamp.replace('Z','+00:00'))
        if dt.tzinfo is None: raise ValueError('Bookkeeping timestamps must include UTC offset.')
    if record.get('schemaVersion',1) != 1:
        raise ValueError('Update ADAM before editing this data version.')
    allowed = {'schemaVersion','createdAt','updatedAt','deleted','deletedAt','origin','operationId',identity_key}
    allowed |= {'todos':{'text','done','due','doneAt','deviceIds'},
        'schedules':{'kind','label','at','timeOfDay','timeZone','repeat','enabled','deviceIds','lastFired','snoozes','durationSeconds','deadline'},
        'notes':{'content','title'},'memoryFacts':{'category','content','confidence','source','learnedAt'},
        'memoryPeople':{'name','relationship','notes','faceEncodingId','firstSeen','lastSeen'}}[kind]
    if set(record)-allowed: raise ValueError('Unsupported fields must be reviewed before editing this record.')
    targets = record.get('deviceIds',[])
    if not isinstance(targets,list) or len(targets)>100 or any(d not in owned for d in targets):
        raise ValueError('Plans may target only your owned devices.')
    if record['deleted']: return
    required = {'todos':('text',),'schedules':('at',),'notes':('content',),'memoryFacts':('category','content'),'memoryPeople':('name',)}[kind]
    for key in required:
        if not isinstance(record.get(key),str) or not 1<=len(record[key].strip())<=2000:
            raise ValueError('Record text is missing or too long.')
    if kind=='todos' and type(record.get('done')) is not bool: raise ValueError('Invalid to-do state.')
    if kind=='schedules':
        if record.get('kind') not in ('alarm','reminder','timer') or type(record.get('enabled')) is not bool:
            raise ValueError('Invalid schedule type or state.')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?',record['at']):
            raise ValueError('Use a complete local date and time for this schedule.')
        datetime.fromisoformat(record['at'])
        if record.get('timeZone'):
            from zoneinfo import ZoneInfo
            ZoneInfo(record['timeZone'])
        repeat=record.get('repeat')
        if repeat is not None and (not isinstance(repeat,list) or len(repeat)>7 or any(x not in ('mon','tue','wed','thu','fri','sat','sun') for x in repeat)):
            raise ValueError('Invalid repeat days.')


class CanonicalSync:
    def __init__(self, account, catalog, directory=None, session=None):
        self.account, self.catalog = account, catalog
        self.directory = Path(directory) if directory else data_directory()/'canonical'
        self.http = session or requests.Session()
        self.lock = threading.RLock()
        self.sync_lock = threading.Lock()
        self.epoch = 0
        self.current_uid = ''
        self.auth_epoch = getattr(account, '_epoch', 0)
        self.error = ''

    def uid(self):
        uid=(self.account.user or {}).get('uid','')
        if uid != self.current_uid or self.auth_epoch != getattr(self.account, '_epoch', 0):
            self.epoch+=1; self.current_uid=uid; self.error=''; self.auth_epoch=getattr(self.account, '_epoch', 0)
        if not uid: raise ValueError('Sign in to access your account data.')
        return uid

    def fence(self, uid, epoch):
        if self.uid()!=uid or self.epoch!=epoch: raise ValueError('The account changed during sync.')

    def path(self,uid): return self.directory/(hashlib.sha256(uid.encode()).hexdigest()+'.json')

    def read(self,uid):
        path=self.path(uid)
        if not path.exists(): return {'version':1,'documents':{},'outbox':{},'conflicts':{},'lastSynced':None,'bridge':{}}
        try:
            if path.stat().st_size>16_000_000: raise ValueError()
            state=json.loads(path.read_text())
            if state['version']!=1 or not all(isinstance(state[k],dict) for k in ('documents','outbox','conflicts','bridge')): raise ValueError()
            return state
        except (ValueError,KeyError,TypeError) as error:
            raise ValueError('Your saved account data needs recovery. The original file has been preserved.') from error

    def write(self,uid,state):
        self.directory.mkdir(parents=True,exist_ok=True)
        temp=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=self.directory,delete=False,encoding='utf-8') as f:
                temp=Path(f.name); json.dump(state,f,ensure_ascii=False);f.flush();os.fsync(f.fileno())
            os.replace(temp,self.path(uid))
        finally:
            if temp: temp.unlink(missing_ok=True)

    def status(self):
        if not self.account.user: return {'pending':0,'conflicts':0,'lastSynced':None,'error':''}
        with self.lock:
            state=self.read(self.uid())
            return {'pending':len(state['outbox']),'conflicts':len(state['conflicts']),
                    'lastSynced':state['lastSynced'],'error':self.error,'syncing':self.sync_lock.locked(),
                    'backend':'canonical-firestore'}

    def records(self):
        with self.lock:
            state=self.read(self.uid())
            # Application evidence is valid only for the current exact record.
            # Remote edits can advance the cache without going through save().
            for rows in state['bridge'].values():
                for path, evidence in rows.items():
                    current = state['documents'].get(path)
                    digest = hashlib.sha256(json.dumps(current, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    if evidence.get('status') == 'robot_applied' and evidence.get('cloudRevision') != digest:
                        evidence['status'] = 'waiting_for_robot'
            return {'documents':state['documents'],'outbox':list(state['outbox']),
                    'conflicts':state['conflicts'],'bridge':state['bridge'],'sync':self.status()}

    def save(self,path,data):
        uid=self.uid();epoch=self.epoch;owned={d['deviceId'] for d in self.catalog(self.account)}
        with self.lock:
            self.fence(uid,epoch);state=self.read(uid);previous=state['documents'].get(path)
            stamp=now();record={**deepcopy(data),'updatedAt':stamp,'createdAt':data.get('createdAt') or stamp,
                'schemaVersion':1,'origin':'desktop','operationId':str(uuid.uuid4())}
            record.setdefault('deleted',False);record['deletedAt']=stamp if record['deleted'] else None
            validate(path,record,uid,owned)
            if len(state['documents'])>=3100 and path not in state['documents']: raise ValueError('Account data is full.')
            baseline=state['outbox'].get(path,{}).get('base',previous)
            state['documents'][path]=record
            state['outbox'][path]={'record':record,'base':baseline,'operationId':record['operationId']}
            state['conflicts'].pop(path,None)
            for device_bridge in state['bridge'].values():
                if path in device_bridge:
                    device_bridge[path]['status'] = 'waiting_for_cloud'
            self.write(uid,state)
        return self.records()

    def request(self,method,path,token,**kwargs):
        response=self.http.request(method,BASE+'/'+quote(path,safe='/'),headers={'Authorization':'Bearer '+token},
            timeout=(5,15),allow_redirects=False,**kwargs)
        if response.status_code in (409,412) or (method == 'GET' and response.status_code == 404): return response,None
        if not response.ok: raise ValueError('Cloud sync is unavailable or not authorized. Your changes remain saved locally.')
        payload=response.json()
        if not isinstance(payload,dict): raise ValueError('Cloud returned an invalid response.')
        return response,payload

    def sync(self):
        if not self.sync_lock.acquire(False): return self.status()
        try:
            uid=self.uid();epoch=self.epoch;token=self.account.id_token();self.fence(uid,epoch)
            owned={d['deviceId'] for d in self.catalog(self.account)};self.fence(uid,epoch)
            # Proposed notes/settings paths are not read until deployed rules
            # support them. Current mobile canonical collections are stable.
            collections=[f'users/{uid}/todos',f'users/{uid}/schedules']
            for device in sorted(owned):
                if not device.startswith('ADAM-SIM-'):
                    collections.extend([f'devices/{device}/memoryFacts',f'devices/{device}/memoryPeople'])
            remote={}
            for collection in collections:
                cursor=None;count=0
                while True:
                    self.fence(uid,epoch)
                    _,payload=self.request('GET',collection,token,params={'pageSize':100,**({'pageToken':cursor} if cursor else {})})
                    if payload is None: break
                    for doc in payload.get('documents',[]):
                        count+=1
                        if count>LIMIT: raise ValueError('Cloud collection exceeds the supported limit.')
                        ident=doc['name'].rsplit('/',1)[-1]
                        if not ID.fullmatch(ident): raise ValueError('Invalid cloud identity.')
                        row=decode(doc)
                        if len(json.dumps(row).encode())>16384: raise ValueError('Cloud record exceeds the supported size.')
                        # Legacy canonical rows may predate common bookkeeping.
                        check=deepcopy(row)
                        check.setdefault('deleted',False)
                        updated=check.get('updatedAt') or check.get('learnedAt') or check.get('lastSeen') or check.get('firstSeen') or check.get('createdAt')
                        check.setdefault('updatedAt',updated)
                        check.setdefault('createdAt',check.get('learnedAt') or check.get('firstSeen') or updated)
                        validate(collection+'/'+ident,check,uid,owned)
                        remote[collection+'/'+ident]=row
                        if len(remote)>3100 or sum(len(json.dumps(v)) for v in remote.values())>8_000_000:
                            raise ValueError('Cloud account exceeds the supported size.')
                    cursor=payload.get('nextPageToken')
                    if not cursor: break
            with self.lock:
                self.fence(uid,epoch);state=self.read(uid);operations=deepcopy(state['outbox'])
            for path,op in operations.items():
                self.fence(uid,epoch);validate(path,op['record'],uid,owned)
                if '/notes/' in path: raise ValueError('Personal-note cloud rules are not deployed. Your note is saved locally.')
                for attempt in range(4):
                    self.fence(uid,epoch);response,doc=self.request('GET',path,token)
                    current=decode(doc) if doc else None
                    if current and current.get('operationId')==op['operationId']:
                        committed=current;break
                    if current != op['base'] and current != op['record']:
                        with self.lock:
                            self.fence(uid,epoch);latest=self.read(uid)
                            latest['conflicts'][path]={'local':op['record'],'remote':current}
                            self.write(uid,latest)
                        committed=None;break
                    params={'currentDocument.updateTime':doc['updateTime']} if doc else {'currentDocument.exists':'false'}
                    self.fence(uid,epoch)
                    response,_=self.request('PATCH',path,token,params=params,json={'fields':fields(op['record'])})
                    if response.status_code not in (409,412): committed=op['record'];break
                else: raise ValueError('Another device is changing this record. Retry sync shortly.')
                if committed is not None:
                    remote[path]=committed
                    with self.lock:
                        self.fence(uid,epoch);latest=self.read(uid)
                        if latest['outbox'].get(path,{}).get('operationId')==op['operationId']:
                            latest['outbox'].pop(path,None);latest['conflicts'].pop(path,None)
                        else:
                            latest['outbox'][path]['base']=committed
                        self.write(uid,latest)
            with self.lock:
                self.fence(uid,epoch);state=self.read(uid)
                for path,data in remote.items():
                    if path not in state['outbox']: state['documents'][path]=data
                # Revoked ownership removes visibility without deleting the
                # old UID's file/outbox. Never apply another owner's data.
                for path in list(state['documents']):
                    if path.startswith('devices/') and path.split('/')[1] not in owned:
                        state['documents'].pop(path)
                state['lastSynced']=now();self.write(uid,state);self.error=''
            return self.status()
        except Exception as error:
            self.error=str(error);raise
        finally: self.sync_lock.release()

    def resolve(self,path,choice):
        with self.lock:
            uid=self.uid();state=self.read(uid);conflict=state['conflicts'].get(path)
            if not conflict or choice not in ('local','remote'): raise ValueError('Choose a valid conflict resolution.')
            if choice=='remote':
                if conflict['remote'] is None: state['documents'].pop(path,None)
                else: state['documents'][path]=conflict['remote']
                state['outbox'].pop(path,None)
            else:
                state['outbox'][path]['base']=conflict['remote']
            state['conflicts'].pop(path);self.write(uid,state)
        return self.records()
