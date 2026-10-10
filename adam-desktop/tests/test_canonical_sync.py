from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import pytest
from canonical_sync import CanonicalSync, fields, decode, validate
from pi_bridge import PiBridge, revision

UID='alice'; PATH='users/alice/todos/todo-1'
STAMP='2026-10-09T12:00:00.000Z'

def todo(text='Review'):
    return {'todoId':'todo-1','text':text,'done':False,'due':None,'doneAt':None,'deviceIds':[],
        'createdAt':STAMP,'updatedAt':STAMP,'deleted':False,'deletedAt':None,'origin':'mobile'}

class Server:
    def __init__(self):self.docs={};self.calls=[];self.serial=0
    def request(self,method,url,**kw):
        path=url.split('/documents/')[1];self.calls.append((method,path,kw));status=200
        if method=='GET' and len(path.split('/'))==3:
            payload={'documents':[{'name':'documents/'+k,'fields':fields(v),'updateTime':str(self.serial)} for k,v in self.docs.items() if k.rsplit('/',1)[0]==path]}
        elif method=='GET':
            if path in self.docs:payload={'fields':fields(self.docs[path]),'updateTime':str(self.serial)}
            else:status=404;payload={}
        else:
            self.docs[path]=decode(kw['json']);self.serial+=1;payload={'updateTime':str(self.serial)}
        return SimpleNamespace(status_code=status,ok=status==200,json=lambda:payload)

@pytest.fixture
def fixture(tmp_path):
    account=SimpleNamespace(user={'uid':UID},id_token=lambda:'fixture',_epoch=0)
    server=Server();catalog=Mock(return_value=[{'deviceId':'ADAM-TEST'}])
    service=CanonicalSync(account,catalog,directory=tmp_path,session=server)
    return service,server,account,catalog

def test_durable_save_restart_upload_uses_only_canonical_document(fixture,tmp_path):
    service,server,account,catalog=fixture
    result=service.save(PATH,todo());assert result['outbox']==[PATH]
    restarted=CanonicalSync(account,catalog,directory=tmp_path,session=server)
    restarted.sync()
    assert restarted.status()['pending']==0
    assert server.docs[PATH]['text']=='Review'
    assert not any(path=='users/alice' for _,path,_ in server.calls)
    assert all('companion' not in json.dumps(kw) for _,_,kw in server.calls)
    assert (tmp_path/next(tmp_path.iterdir()).name).exists()

def test_two_clients_conflicts_and_tombstones_preserve_both_versions(fixture,tmp_path):
    service,server,account,catalog=fixture;server.docs[PATH]=todo();service.sync()
    service.save(PATH,todo('Desktop'))
    server.docs[PATH]=todo('Phone');service.sync()
    state=service.records();assert state['conflicts'][PATH]['local']['text']=='Desktop'
    assert state['conflicts'][PATH]['remote']['text']=='Phone'
    service.resolve(PATH,'local');service.sync();assert server.docs[PATH]['text']=='Desktop'
    service.save(PATH,{**server.docs[PATH],'deleted':True});service.sync()
    assert server.docs[PATH]['deleted'] is True
    assert all(method!='DELETE' for method,_,_ in server.calls)

def test_account_fence_includes_same_uid_new_auth_session(fixture):
    service,server,account,catalog=fixture;service.uid();epoch=service.epoch
    account._epoch+=1
    with pytest.raises(ValueError):service.fence('alice',epoch)
    account.user={'uid':'bob'}
    with pytest.raises(ValueError):service.save(PATH,todo())

def test_scope_secrets_unsupported_schema_and_foreign_targets_rejected(fixture):
    service,_,_,_=fixture
    for path,row in [('users/bob/todos/todo-1',todo()),(PATH,{**todo(),'syncToken':'secret'}),
                     (PATH,{**todo(),'deviceIds':['ADAM-FOREIGN']})]:
        with pytest.raises(ValueError):service.save(path,row)
    with pytest.raises(ValueError):validate(PATH,{**todo(),'schemaVersion':9},UID,{'ADAM-TEST'})

def test_bridge_requires_exact_ack_and_persists_real_evidence(fixture):
    service,server,_,_=fixture;server.docs[PATH]=todo();service.sync()
    gate=SimpleNamespace(status=lambda:{'ready':True,'selected':'ADAM-TEST'})
    connection=Mock()
    connection.call.side_effect=[({'uid':UID,'deviceId':'ADAM-TEST','schemaVersion':1,'documents':{}},None),
        ({'deviceId':'ADAM-TEST','path':PATH,'appliedRevision':revision(todo())},None)]
    result=PiBridge(service,gate,connection).run()
    assert result['records'][PATH]['status']=='robot_applied'
    assert service.records()['bridge']['ADAM-TEST'][PATH]['piRevision']==revision(todo())
    assert not any('/executionState/' in path for _,path,_ in server.calls)

def test_bridge_rejects_wrong_robot_or_fabricated_ack(fixture):
    service,server,_,_=fixture;server.docs[PATH]=todo();service.sync()
    gate=SimpleNamespace(status=lambda:{'ready':True,'selected':'ADAM-TEST'})
    connection=Mock();connection.call.side_effect=[({'uid':UID,'deviceId':'ADAM-TEST','schemaVersion':1,'documents':{}},None),
        ({'deviceId':'ADAM-OTHER','path':PATH,'appliedRevision':'wrong'},None)]
    with pytest.raises(ValueError):PiBridge(service,gate,connection).run()
    assert not service.records()['bridge']

def test_missing_patch_endpoint_keeps_outbox(fixture):
    service,server,_,_=fixture
    original=server.request
    def request(method,url,**kw):
        if method=='PATCH':
            return SimpleNamespace(status_code=404,ok=False,json=lambda:{})
        return original(method,url,**kw)
    server.request=request
    service.save(PATH,todo())
    with pytest.raises(ValueError, match='remain saved locally'):
        service.sync()
    assert service.records()['outbox']==[PATH]


def test_new_edit_invalidates_old_application_badge_but_keeps_baseline(fixture):
    service,server,_,_=fixture
    server.docs[PATH]=todo();service.sync()
    state=service.read(UID)
    state['bridge']['ADAM-TEST']={PATH:{'status':'robot_applied','cloudRevision':'old','piRevision':'old'}}
    service.write(UID,state)
    service.save(PATH,todo('Changed'))
    evidence=service.records()['bridge']['ADAM-TEST'][PATH]
    assert evidence['status']=='waiting_for_cloud'
    assert evidence['piRevision']=='old'


def test_execution_relay_advances_only_after_entire_page_accepted(fixture):
    service,server,_,_=fixture
    gate=SimpleNamespace(status=lambda:{'ready':True,'selected':'ADAM-TEST'})
    connection=Mock()
    connection.call.return_value=({'deviceId':'ADAM-TEST','receipts':[{'payload':'signed'}],'nextCursor':'next'},None)
    server.post=Mock(return_value=SimpleNamespace(ok=False,json=lambda:{}))
    bridge=PiBridge(service,gate,connection)
    with pytest.raises(ValueError,match='verification is pending'):bridge.report_execution()
    assert not service.read(UID).get('receiptCursors')
    server.post.return_value=SimpleNamespace(ok=True,json=lambda:{'result':{'ok':True}})
    assert bridge.report_execution()=={'verified':1,'more':True}
    assert service.read(UID)['receiptCursors']['ADAM-TEST']=='next'
    assert server.post.call_args.kwargs['allow_redirects'] is False


def test_execution_relay_account_change_cannot_commit_cursor(fixture):
    service,server,account,_=fixture
    gate=SimpleNamespace(status=lambda:{'ready':True,'selected':'ADAM-TEST'})
    connection=Mock()
    connection.call.return_value=({'deviceId':'ADAM-TEST','receipts':[{'payload':'signed'}],'nextCursor':'next'},None)
    def post(*args,**kwargs):
        account._epoch+=1
        return SimpleNamespace(ok=True,json=lambda:{'result':{'ok':True}})
    server.post=post
    with pytest.raises(ValueError):PiBridge(service,gate,connection).report_execution()
    assert not service.read(UID).get('receiptCursors')
