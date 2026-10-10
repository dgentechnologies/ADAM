"""Exercise the shipped Pi persistence modules without microphone/hardware imports."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from test_onboarding import robot

ROOT=Path(__file__).parents[2]/'MP-MC codes/pi/adam'
STAMP='2026-10-09T12:00:00.000Z'

def load(name,monkeypatch):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,name,module);spec.loader.exec_module(module)
    return module

@pytest.fixture
def pi(robot,tmp_path,monkeypatch):
    robot.write('identity.json',{**robot.identity(),'timeZone':'Etc/UTC'})
    config=SimpleNamespace(MEMORY_FILE=tmp_path/'facts.json',FACE_MEMORY_FILE=tmp_path/'faces.json',
        CONV_MEMORY_FILE=tmp_path/'conversation.json',CONV_MAX_TURNS=20,SCHEDULE_FILE=tmp_path/'schedules.json')
    monkeypatch.setitem(sys.modules,'config',config);monkeypatch.setitem(sys.modules,'desktop_pairing',robot)
    load('canonical_contract',monkeypatch)
    memory=load('memory_store',monkeypatch)
    scheduler=SimpleNamespace(_store={'schedules':[],'todos':[],'tombstones':[]})
    monkeypatch.setitem(sys.modules,'scheduler',scheduler)
    memories=load('memory_sync',monkeypatch);records=load('record_sync',monkeypatch)
    return records,memories,memory,scheduler,config

def plan():
    return {'scheduleId':'alarm-1','kind':'alarm','label':'Wake up','at':'2030-10-10T07:00','timeZone':'Etc/UTC',
            'enabled':True,'repeat':None,'deviceIds':['ADAM-TEST'],'createdAt':STAMP,'updatedAt':STAMP,
            'deleted':False,'deletedAt':None,'origin':'desktop','schemaVersion':1}

def test_plan_apply_is_durable_idempotent_and_does_not_reset_firing(pi):
    records,_,_,scheduler,config=pi;row=plan();path='users/alice/schedules/alarm-1'
    ack=records.apply({'path':path,'record':row,'baseRevision':None})
    assert ack['appliedRevision']==records.revision(row)
    assert json.loads(config.SCHEDULE_FILE.read_text())['schedules'][0]['id']=='alarm-1'
    records.apply({'path':path,'record':row,'baseRevision':None})
    assert len(scheduler._store['schedules'])==1
    scheduler._store['schedules'][0].update(last_fired='2030-10-10T07:00',enabled=False)
    snapshot=records.snapshot()
    assert snapshot['documents'][path]['enabled'] is True
    assert snapshot['execution']['alarm-1']['lastFiredAt']=='2030-10-10T07:00'

def test_bad_scope_wrong_zone_and_changed_revision_do_not_write(pi):
    records,_,_,scheduler,config=pi
    for row,path in [(plan(),'users/bob/schedules/alarm-1'),({**plan(),'timeZone':'Asia/Kolkata'},'users/alice/schedules/alarm-1'),({**plan(),'deviceIds':['ADAM-OTHER']},'users/alice/schedules/alarm-1')]:
        with pytest.raises(ValueError):records.apply({'path':path,'record':row,'baseRevision':None})
    assert scheduler._store['schedules']==[]
    assert not config.SCHEDULE_FILE.exists()

def test_failed_persistence_never_acknowledges_or_mutates_scheduler(pi,monkeypatch):
    records,_,_,scheduler,_=pi
    def fail(*args,**kwargs):raise OSError('disk full')
    monkeypatch.setattr(records,'save_json',fail)
    with pytest.raises(OSError):records.apply({'path':'users/alice/schedules/alarm-1','record':plan(),'baseRevision':None})
    assert scheduler._store['schedules']==[]

def test_voice_memories_versioned_deleted_and_no_biometrics_exported(pi):
    _,memories,memory,_,_=pi
    memory.memory['coffee']='No sugar'
    memory.faces['person-1']={'name':'Sam','notes':'Friend','embedding':[0.1,0.2],'photo':'private-file'}
    first=memories.snapshot();assert len(first)==2
    assert 'embedding' not in json.dumps(first) and 'private-file' not in json.dumps(first)
    assert memories.snapshot()==first
    path=next(k for k in first if '/memoryFacts/' in k)
    del memory.memory['coffee']
    assert memories.snapshot()[path]['deleted'] is True

def test_cloud_memory_apply_recovers_and_keeps_existing_face_payload_local(pi):
    _,memories,memory,_,config=pi
    memory.faces['sam']={'name':'Sam','notes':'Old','embedding':[1,2,3]}
    first=memories.snapshot();path=next(iter(first));updated={**first[path],'notes':'Updated','updatedAt':first[path]['updatedAt']}
    ack=memories.apply({'path':path,'record':updated,'baseRevision':memories.digest(first[path])})
    assert ack['appliedRevision']==memories.digest(updated)
    assert memory.faces['sam']['embedding']==[1,2,3]
    assert json.loads(config.FACE_MEMORY_FILE.read_text())['sam']['notes']=='Updated'
    assert memories.snapshot()[path]==updated

def test_receipts_are_signed_and_report_only_durable_firing(pi,monkeypatch):
    import base64
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    records,_,_,scheduler,config=pi
    records.apply({'path':'users/alice/schedules/alarm-1','record':plan(),'baseRevision':None})
    evidence=load('cloud_evidence',monkeypatch)
    scheduler._store['schedules'][0]['last_fired']='2026-10-09T07:00#alarm-1'
    receipt=evidence.execution_receipts()['receipts'][0]
    payload=json.loads(base64.b64decode(receipt['payload']))
    assert payload['lastFiredAt'] is None # unsaved in-memory change is not evidence
    config.SCHEDULE_FILE.write_text(json.dumps(scheduler._store))
    receipt=evidence.execution_receipts()['receipts'][0]
    payload=json.loads(base64.b64decode(receipt['payload']))
    assert payload['lastOccurrenceId']=='2026-10-09T07:00#alarm-1'
    assert payload['lastFiredAt']=='2026-10-09T07:00:00.000Z'
    key=serialization.load_pem_public_key(evidence.public_bundle()['publicKeyPem'].encode())
    key.verify(base64.b64decode(receipt['signature']),base64.b64decode(receipt['payload']),ec.ECDSA(hashes.SHA256()))
    assert 'private' not in json.dumps(evidence.public_bundle()).lower()


def test_python_claim_proof_verifies_in_cloud_javascript(pi,monkeypatch):
    import subprocess
    evidence=load('cloud_evidence',monkeypatch)
    payload={'proof':evidence.sign('claim'),'hardware':{**evidence.public_bundle(),'enabled':True}}
    script="import {verifyProof} from './firebase/src/protocol.js'; let s=''; for await(const c of process.stdin)s+=c; const d=JSON.parse(s); if(verifyProof(d.proof,d.hardware,'claim','alice').uid!=='alice')process.exit(1);"
    subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(payload),text=True,cwd=ROOT.parents[2],check=True,capture_output=True)


def test_pi_rejects_credentials_biometrics_and_wrong_schema_before_persistence(pi):
    records,_,_,scheduler,_=pi
    for change in [{'apiKey':'secret'},{'schemaVersion':2},{'lastFired':'forged'}, {'deviceIds':['ADAM-TEST']*9}]:
        with pytest.raises(ValueError):records.apply({'path':'users/alice/schedules/alarm-1','record':{**plan(),**change},'baseRevision':None})
    assert not scheduler._store['schedules']


def test_real_scheduler_preserves_second_precision_timers(pi,monkeypatch):
    _,_,_,_,config=pi
    config.MISSED_GRACE_S=300;config.CLOCK_JUMP_S=20;config.TICK_INTERVAL_S=1
    scheduler=load('scheduler',monkeypatch)
    before=scheduler._now()
    scheduler.set_timer(seconds=42,label='Seconds matter')
    row=scheduler._store['schedules'][0]
    due=scheduler._parse(row['at'])
    assert due is not None and 40 < (due-before).total_seconds() <= 42
    scheduler._check(due+scheduler.dt.timedelta(seconds=1),43,43)
    assert scheduler.drain_pending_fires()[0]['id']==row['id']
