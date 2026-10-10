"""Selected-robot bridge with durable, per-record apply evidence and conflicts."""
from copy import deepcopy
import hashlib
import json


def revision(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PiBridge:
    def __init__(self, canonical, onboarding, connection):
        self.canonical,self.onboarding,self.connection=canonical,onboarding,connection

    def run(self):
        gate=self.onboarding.status()
        if not gate['ready']: raise ValueError('Verify your ADAM connection before syncing robot data.')
        device=gate['selected'];uid=self.canonical.uid();epoch=self.canonical.epoch
        def fence():
            self.canonical.fence(uid,epoch)
            current=self.onboarding.status()
            if not current['ready'] or current['selected']!=device:
                raise ValueError('The selected ADAM changed during delivery.')
        fence();snapshot,error=self.connection.call('GET','/api/sync/records')
        if error:raise ValueError(error)
        if snapshot.get('uid')!=uid or snapshot.get('deviceId')!=device or snapshot.get('schemaVersion')!=1:
            raise ValueError('Robot sync identity did not match this account.')
        documents=snapshot.get('documents',{})
        if not isinstance(documents,dict) or len(documents)>3000:raise ValueError('Robot sync exceeds the supported limit.')
        with self.canonical.lock:
            local=self.canonical.read(uid);cloud=deepcopy(local['documents']);outbox=set(local['outbox'])
            baselines=deepcopy(local['bridge'].get(device,{}))
        results={}
        for path in sorted(set(documents)|set(cloud)):
            fence()
            plan = path.startswith(f'users/{uid}/') and path.split('/')[2] in ('schedules','todos')
            memory = path.startswith(f'devices/{device}/') and path.split('/')[2] in ('memoryFacts','memoryPeople')
            if not plan and not memory:
                continue
            source=cloud.get(path);robot=documents.get(path);old=baselines.get(path,{})
            if source and source.get('deviceIds') and device not in source['deviceIds']:continue
            if path in outbox:
                results[path]={'status':'waiting_for_cloud'};continue
            cr=revision(source) if source is not None else None;pr=revision(robot) if robot is not None else None
            if source==robot:
                results[path]={'status':'robot_applied','cloudRevision':cr,'piRevision':pr};continue
            cloud_changed=cr!=old.get('cloudRevision');pi_changed=pr!=old.get('piRevision')
            if (source is None and robot is not None) or (pi_changed and not cloud_changed):
                # Save robot evidence to the current UID's durable outbox. It is
                # not cloud committed until a later successful canonical sync.
                from canonical_sync import validate
                owned={d['deviceId'] for d in self.canonical.catalog(self.canonical.account)}
                validate(path,robot,uid,owned)
                self.canonical.save(path,robot)
                results[path]={'status':'received_from_robot','cloudRevision':cr,'piRevision':pr}
            elif robot is None or (cloud_changed and not pi_changed):
                fence();ack,error=self.connection.call('POST','/api/sync/apply',
                    {'path':path,'record':source,'baseRevision':pr})
                if error: results[path]={'status':'delivery_failed','error':error};continue
                if ack.get('deviceId')!=device or ack.get('path')!=path or ack.get('appliedRevision')!=cr:
                    raise ValueError('ADAM did not acknowledge this exact record revision.')
                results[path]={'status':'robot_applied','cloudRevision':cr,'piRevision':cr}
            else:
                results[path]={'status':'conflict','cloudRevision':old.get('cloudRevision'),
                    'piRevision':old.get('piRevision'),'cloud':source,'robot':robot}
        with self.canonical.lock:
            fence();latest=self.canonical.read(uid)
            latest['bridge'][device]={**baselines,**results};self.canonical.write(uid,latest)
        return {'deviceId':device,'records':results,'execution':snapshot.get('execution',{})}

    def report_execution(self):
        """Relay bounded, signed Pi evidence; the cloud verifies every claim.

        Cursor advances only when the whole page is accepted. A failed relay
        retries from the same cursor with fresh short-lived robot proofs.
        """
        from account import FIREBASE_PROJECT_ID
        from urllib.parse import quote
        gate=self.onboarding.status()
        if not gate['ready']:raise ValueError('Connect to your verified ADAM first.')
        uid=self.canonical.uid();epoch=self.canonical.epoch;device=gate['selected']
        def fence():
            self.canonical.fence(uid,epoch)
            current=self.onboarding.status()
            if not current['ready'] or current['selected']!=device:raise ValueError('ADAM changed during execution reporting.')
        with self.canonical.lock:
            cursor=self.canonical.read(uid).get('receiptCursors',{}).get(device,'')
        fence()
        payload,error=self.connection.call('GET','/api/sync/receipts?after='+quote(cursor,safe=''))
        if error or not isinstance(payload,dict) or payload.get('deviceId')!=device:raise ValueError('Robot execution evidence is unavailable. Update ADAM and retry.')
        receipts=payload.get('receipts',[])
        if not isinstance(receipts,list) or len(receipts)>8:raise ValueError('Invalid execution evidence batch.')
        token=self.canonical.account.id_token()
        for receipt in receipts:
            fence()
            response=self.canonical.http.post(f'https://us-central1-{FIREBASE_PROJECT_ID}.cloudfunctions.net/reportExecution',
                headers={'Authorization':'Bearer '+token},json={'data':receipt},timeout=(5,10),allow_redirects=False)
            if not response.ok or response.json().get('result',{}).get('ok') is not True:
                raise ValueError('Cloud execution verification is pending. Deploy the trusted Firebase function, register this ADAM, then retry.')
        with self.canonical.lock:
            fence();state=self.canonical.read(uid)
            state.setdefault('receiptCursors',{})[device]=payload.get('nextCursor','')
            self.canonical.write(uid,state)
        return {'verified':len(receipts),'more':bool(payload.get('nextCursor'))}
