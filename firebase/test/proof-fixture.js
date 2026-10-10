import {generateKeyPairSync,sign} from 'node:crypto';
export function fixture(purpose='claim', extra={}) {
 const {publicKey,privateKey}=generateKeyPairSync('ec',{namedCurve:'prime256v1'});
 const hardware={enabled:true,hardwareId:'hardware-1',deviceId:'ADAM-TEST',ownershipEpoch:0,publicKeyPem:publicKey.export({type:'spki',format:'pem'})};
 const payload={version:1,purpose,uid:'alice',hardwareId:'hardware-1',deviceId:'ADAM-TEST',ownershipEpoch:0,nonce:'a'.repeat(32),issuedAt:Date.now(),expiresAt:Date.now()+60000,...extra};
 function envelope(p=payload){const bytes=Buffer.from(JSON.stringify(p));return {payload:bytes.toString('base64'),signature:sign('sha256',bytes,privateKey).toString('base64')};}
 return {hardware,payload,envelope};
}
