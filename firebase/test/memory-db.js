// Small transaction adapter for workflow unit tests. Emulator tests remain the
// authority for real Firestore transaction and Security Rules behavior.
export function memoryDB() {
 const rows=new Map();
 const ref=path=>({path,collection:kind=>({doc:id=>ref(`${path}/${kind}/${id}`),limit:()=>({query:`${path}/${kind}/`})})});
 const snapshot=path=>({exists:rows.has(path),data:()=>structuredClone(rows.get(path))});
 return {rows,doc:ref,runTransaction:async fn=>{
  const writes=[];const tx={get:async r=>r.query?{empty:![...rows.keys()].some(k=>k.startsWith(r.query))}:snapshot(r.path),
   set:(r,v)=>writes.push(()=>rows.set(r.path,v)),update:(r,v)=>writes.push(()=>rows.set(r.path,{...rows.get(r.path),...v})),
   create:(r,v)=>{if(rows.has(r.path))throw Error('Already exists');writes.push(()=>rows.set(r.path,v));}};
  const value=await fn(tx);writes.forEach(f=>f());return value;
 }};
}
