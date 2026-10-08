/**
 * ADAM assembly viewport and animated capsule-eye display.
 * DGEN Technologies Pvt. Ltd.
 * Uses the local assembly, monochrome materials and an OLED face canvas
 * with gentle gaze shifts, natural blinks and subtle live emotion changes.
 */

let scene, camera, renderer, controls;
let adamModel = null;
let headNode = null;
let faceMesh = null;
let faceCanvas = null;
let faceCtx = null;
let faceTexture = null;

// Live mirror state
let currentEmotion = 'idle';
let isSpeaking = false;
let isListening = false;
let isConnected = false;
let blinkProgress = 1.0; // 1 = open, 0 = closed
let animTime = 0;
let viewportVisible = true;
let viewportOnScreen = true;
let contextLost = false;
let animationFrame = 0;
let lastFrame = 0;
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
let reducedMotion = motionPreference.matches;

const EMOTIONS = [
  'happy', 'sad', 'surprised', 'angry', 'thinking',
  'excited', 'love', 'blush', 'confused', 'smug',
  'sleep', 'rizz', 'panic', 'shy', 'reconnecting'
];

function init3D() {
  const container = document.getElementById('viewportContainer');
  const canvas = document.getElementById('threeCanvas');
  if (!container || !canvas) return;

  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene & Camera
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 100);
  camera.position.set(1.3, 0.7, 4.5);

  // WebGL Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'low-power' });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  // Achromatic Studio Lighting
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
  keyLight.position.set(2, 4, 3);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x888888, 0.8);
  fillLight.position.set(-3, 1, -1);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 1.5);
  rimLight.position.set(0, 3, -3);
  scene.add(rimLight);

  const ambient = new THREE.AmbientLight(0x333336, 0.9);
  scene.add(ambient);

  // OrbitControls
  if (THREE.OrbitControls) {
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enabled = true;
    controls.enablePan = false;
    controls.enableZoom = false;
    controls.enableDamping = !reducedMotion;
    controls.minPolarAngle = Math.PI / 3;
    controls.maxPolarAngle = Math.PI * 0.58;
    controls.target.set(0, 0.02, 0);
    controls.update();
    controls.addEventListener('change', request3DFrame);
  }

  // Setup Dynamic OLED Face Canvas
  setupFaceCanvas();

  // Load Model
  loadAdamModel();
  createStudioFloor();
  document.getElementById('robotFallback').hidden = true;
  canvas.addEventListener('keydown', (event) => {
    if (!adamModel || !['ArrowLeft','ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    adamModel.rotation.y += event.key === 'ArrowLeft' ? -0.15 : 0.15;
    request3DFrame();
  });
  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    contextLost = true;
    sync3DAnimation();
    canvas.hidden = true;
    document.getElementById('robotFallback').hidden = false;
  });
  canvas.addEventListener('webglcontextrestored', () => {
    contextLost = false;
    canvas.hidden = false;
    document.getElementById('robotFallback').hidden = true;
    sync3DAnimation();
  });

  // Resize listener
  window.addEventListener('resize', onWindowResize);
  new ResizeObserver(onWindowResize).observe(container);
  new IntersectionObserver(entries => {
    viewportOnScreen = entries[0].isIntersecting;
    sync3DAnimation();
  }).observe(container);
  document.addEventListener('visibilitychange', sync3DAnimation);
  motionPreference.addEventListener('change', event => {
    reducedMotion = event.matches;
    if (controls) controls.enableDamping = !reducedMotion;
    blinkProgress = 1;
    request3DFrame();
  });

  // Start Animation
  request3DFrame();
}

function setupFaceCanvas() {
  faceCanvas = document.createElement('canvas');
  faceCanvas.width = 512;
  faceCanvas.height = 512;
  faceCtx = faceCanvas.getContext('2d');
  faceTexture = new THREE.CanvasTexture(faceCanvas);
  faceTexture.flipY = false;
  faceTexture.encoding = THREE.sRGBEncoding;
}

function loadAdamModel() {
  const loader=new THREE.GLTFLoader();
  loader.load('/static/models/adam-body.glb',gltf=>{
    // The local Blender export uses +X as its front. Present +Z to the camera.
    const content=gltf.scene;content.rotation.y=-Math.PI/2;content.updateMatrixWorld(true);
    const bounds=new THREE.Box3().setFromObject(content),size=bounds.getSize(new THREE.Vector3()),center=bounds.getCenter(new THREE.Vector3());
    const scale=1.95/size.y;content.scale.multiplyScalar(scale);content.position.copy(center.multiplyScalar(-scale));
    adamModel=new THREE.Group();adamModel.add(content);
    content.traverse(child=>{
      if(!child.isMesh)return;
      const name=(child.name||'').toLowerCase();
      if(name==='facescreen'){
        faceMesh=child;
        // Screen geometry faces +X in the CAD export. Project UVs across Z/Y.
        const geometry=child.geometry.clone();geometry.computeBoundingBox();
        const box=geometry.boundingBox,position=geometry.attributes.position;
        const uv=new Float32Array(position.count*2);
        for(let i=0;i<position.count;i++){
          uv[i*2]=1-(position.getZ(i)-box.min.z)/(box.max.z-box.min.z);
          uv[i*2+1]=(position.getY(i)-box.min.y)/(box.max.y-box.min.y);
        }
        geometry.setAttribute('uv',new THREE.BufferAttribute(uv,2));child.geometry=geometry;
        faceTexture.flipY=true;child.material=new THREE.MeshBasicMaterial({map:faceTexture});
      }else{
        const style=material=>{
          const white=material?.name==='Material.001',rim=material?.name==='ADAM_Silver';
          return new THREE.MeshStandardMaterial({color:white?0xd9dde4:rim?0x23262d:0x0c0e12,metalness:rim?.35:.18,roughness:white?.5:rim?.4:.5});
        };
        child.material=Array.isArray(child.material)?child.material.map(style):style(child.material);
      }
      if(name==='head')headNode=child;
    });
    scene.add(adamModel);adamModel.updateMatrixWorld(true);
    if(headNode){
      const head=new THREE.Box3().setFromObject(headNode),c=head.getCenter(new THREE.Vector3()),d=head.getSize(new THREE.Vector3());
      Object.assign(sensorLocations,{
        touch1:[head.min.x,c.y,head.max.z-d.z*.2],
        touch2:[head.max.x,c.y,head.max.z-d.z*.2],
        touch3:[c.x,head.max.y,c.z],
        touch4:[c.x,c.y,head.min.z]
      });
      // Anchor to the actual rear shell, not a point on the side of its box.
      const rearRay=new THREE.Raycaster(new THREE.Vector3(c.x,c.y,head.min.z-d.z),new THREE.Vector3(0,0,1));
      const rearHit=rearRay.intersectObject(headNode,true)[0];
      if(rearHit)sensorLocations.touch4=rearHit.point.toArray();
    }
    document.getElementById('threeCanvas').dataset.modelSource='adam-body.glb';
    updateSensorAnchors();request3DFrame();
  },undefined,()=>{
    createParametricAdam();
    document.getElementById('threeCanvas').dataset.modelSource='fallback';
    updateSensorAnchors();request3DFrame();
  });
}

function attachFallbackFacePlane() {
  // Create an OLED visor plate attached in front of the head
  const planeGeo = new THREE.PlaneGeometry(0.35, 0.25);
  const planeMat = new THREE.MeshBasicMaterial({
    map: faceTexture,
    transparent: true,
    opacity: 0.95,
  });
  faceMesh = new THREE.Mesh(planeGeo, planeMat);
  faceMesh.position.set(0, 0.15, 0.28);
  scene.add(faceMesh);
}

function createParametricAdam() {
  const group = new THREE.Group();
  // A sculpted body and soft oval head reproduce ADAM's physical silhouette.
  const headGeo = new THREE.SphereGeometry(0.62, 56, 36);
  const headMat = new THREE.MeshStandardMaterial({
    color: 0x070708,
    metalness: 0.18,
    roughness: 0.48,
  });
  const head = new THREE.Mesh(headGeo, headMat);
  head.scale.set(1, 0.79, 0.75);
  head.position.y = 0.55;
  headNode = head;
  group.add(head);
  // Rounded visor with normalized UVs for the dynamic OLED face.
  const screenShape = new THREE.Shape();
  const sw = 1.02, sh = 0.59, radius = 0.24;
  screenShape.moveTo(-sw/2+radius,-sh/2);
  screenShape.lineTo(sw/2-radius,-sh/2);
  screenShape.quadraticCurveTo(sw/2,-sh/2,sw/2,-sh/2+radius);
  screenShape.lineTo(sw/2,sh/2-radius);
  screenShape.quadraticCurveTo(sw/2,sh/2,sw/2-radius,sh/2);
  screenShape.lineTo(-sw/2+radius,sh/2);
  screenShape.quadraticCurveTo(-sw/2,sh/2,-sw/2,sh/2-radius);
  screenShape.lineTo(-sw/2,-sh/2+radius);
  screenShape.quadraticCurveTo(-sw/2,-sh/2,-sw/2+radius,-sh/2);
  const visorGeo = new THREE.ShapeGeometry(screenShape, 24);
  const pos = visorGeo.attributes.position;
  const uv = visorGeo.attributes.uv;
  for (let i=0; i<pos.count; i++) uv.setXY(i,pos.getX(i)/sw+0.5,pos.getY(i)/sh+0.5);
  faceTexture.flipY = true;
  const visorMat = new THREE.MeshBasicMaterial({ map: faceTexture, side: THREE.DoubleSide });
  faceMesh = new THREE.Mesh(visorGeo, visorMat);
  faceMesh.position.set(0, 0.55, 0.48);
  group.add(faceMesh);
  // Smooth shoulder, tapered waist and rounded base; no network model load.
  const bodyPoints = [
    [0,-.92],[.36,-.92],[.52,-.89],[.58,-.83],[.59,-.76],
    [.56,-.53],[.5,-.25],[.44,-.12],[.35,-.06],[.18,-.04],[0,-.04]
  ].map(([x,y])=>new THREE.Vector2(x,y));
  const bodyGeo = new THREE.LatheGeometry(bodyPoints, 64);
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0x070708,
    metalness: 0.18,
    roughness: 0.5,
  });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  group.add(body);
  const neck = new THREE.Mesh(new THREE.CylinderGeometry(.18,.21,.2,32),headMat);
  neck.position.y=.08;group.add(neck);
  const base = new THREE.Mesh(new THREE.CylinderGeometry(.565,.565,.075,64),new THREE.MeshStandardMaterial({color:0x09090b,roughness:.8}));
  base.position.y=-.87;group.add(base);
  const mark = document.createElement('canvas');mark.width=512;mark.height=128;
  const ctx=mark.getContext('2d');
  const markTexture=new THREE.CanvasTexture(mark);
  const brandImage=new Image();brandImage.onload=()=>{
    ctx.drawImage(brandImage,0,0,512,128);
    const pixels=ctx.getImageData(0,0,512,128);
    for(let i=0;i<pixels.data.length;i+=4)pixels.data[i+3]=Math.max(0,(Math.max(pixels.data[i],pixels.data[i+1],pixels.data[i+2])-25)*1.1);
    ctx.putImageData(pixels,0,0);markTexture.needsUpdate=true;
  };brandImage.src='/static/images/adam-wordmark.png';
  const logo=new THREE.Mesh(new THREE.PlaneGeometry(.54,.135),new THREE.MeshBasicMaterial({map:markTexture,transparent:true,depthWrite:false}));
  logo.position.set(0,-.46,.55);logo.rotation.x=-.16;group.add(logo);

  adamModel = group;
  scene.add(adamModel);
}

function createStudioFloor() {
  const shadowCanvas=document.createElement('canvas');shadowCanvas.width=256;shadowCanvas.height=256;
  const ctx=shadowCanvas.getContext('2d'),gradient=ctx.createRadialGradient(128,128,12,128,128,126);
  gradient.addColorStop(0,'rgba(0,0,0,.85)');gradient.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=gradient;ctx.fillRect(0,0,256,256);
  const shadow=new THREE.Mesh(new THREE.PlaneGeometry(3.6,3.6),new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(shadowCanvas),transparent:true,depthWrite:false}));
  shadow.rotation.x=-Math.PI/2;shadow.position.y=-.968;scene.add(shadow);
  for(const radius of [1.1,1.7,2.5]){
    const points=[];for(let i=0;i<128;i++){const angle=i/128*Math.PI*2;points.push(new THREE.Vector3(Math.cos(angle)*radius,-.96,Math.sin(angle)*radius));}
    const ring=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xa0a5b2,transparent:true,opacity:radius===1.1?.09:.045,depthWrite:false}));scene.add(ring);
  }
}

const sensorLocations={touch1:[-.59,.57,.12],touch2:[.59,.57,.12],touch3:[0,1.04,0],touch4:[0,.57,-.32]};
let sensorLayout=[];
function updateSensorAnchors(){
  const stage=document.getElementById('stage'),svg=document.getElementById('sensorLines');
  if(!stage||!svg)return;
  const box=stage.getBoundingClientRect();if(!box.width||!box.height)return;
  svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
  sensorLayout=[];svg.replaceChildren();
  for(const [sensor,coords] of Object.entries(sensorLocations)){
    const label=document.querySelector(`#pin-${sensor}>summary`);if(!label)continue;
    const rect=label.getBoundingClientRect(),left=rect.left+rect.width/2<box.left+box.width/2;
    const start={x:(left?rect.right:rect.left)-box.left,y:rect.bottom-box.top};
    const path=document.createElementNS(svg.namespaceURI,'path');
    const ring=document.createElementNS(svg.namespaceURI,'circle');ring.setAttribute('r','7');ring.setAttribute('class','anchor-ring');
    const dot=document.createElementNS(svg.namespaceURI,'circle');dot.setAttribute('r','3.5');
    svg.append(path,ring,dot);sensorLayout.push({sensor,vector:new THREE.Vector3(...coords),start,left,path,ring,dot,width:box.width,height:box.height});
  }
  projectSensorAnchors();
}
function projectSensorAnchors(){
  if(!camera||!adamModel)return;
  adamModel.updateMatrixWorld();camera.updateMatrixWorld();
  for(const item of sensorLayout){
    const world=item.vector.clone().applyMatrix4(adamModel.matrixWorld);
    const behind=item.sensor==='touch4'&&new THREE.Vector3(0,0,-1).transformDirection(adamModel.matrixWorld).dot(camera.position.clone().sub(world).normalize())<.08;
    for(const marker of [item.path,item.ring,item.dot])marker.style.display=behind?'none':'';
    const point=world.project(camera);
    const x=(point.x+1)*item.width/2,y=(1-point.y)*item.height/2;
    const elbow=item.start.x+(item.left?1:-1)*Math.min(36,Math.abs(x-item.start.x)*.3);
    item.path.setAttribute('d',`M ${item.start.x} ${item.start.y} L ${elbow} ${item.start.y} L ${x} ${y}`);
    for(const marker of [item.ring,item.dot]){marker.setAttribute('cx',x);marker.setAttribute('cy',y);}
  }
}
window.updateSensorAnchors=updateSensorAnchors;

// Both capsules move together like a gaze. Timers use visible animation time,
// so hiding the dashboard never queues blinks or causes a jump on return.
const eyeMotion = {x:0,y:0,targetX:0,targetY:0,nextLook:1.4,blinkStart:-1,
  nextBlink:2.8,doubleBlink:false,height:40,width:142,tilt:0};
const eyeEase = value => value * value * (3 - 2 * value);
function updateEyeMotion(delta) {
  const t=animTime,m=eyeMotion;
  const emotion=isConnected?currentEmotion:'idle';
  const mood={sad:[32,.12],angry:[32,-.14],surprised:[60,0],excited:[48,0],sleep:[4,0],thinking:[38,.04]}[emotion]||[40,0];
  if(reducedMotion){m.x=0;m.y=0;m.height=mood[0];m.tilt=mood[1];blinkProgress=1;return;}
  if(t>=m.nextLook){
    const centered=Math.random()<.35;
    m.targetX=centered?0:(Math.random()-.5)*27;
    m.targetY=centered?0:(Math.random()-.5)*12;
    m.nextLook=t+1.8+Math.random()*3.5;
  }
  const gazeBlend=1-Math.exp(-delta*16),moodBlend=1-Math.exp(-delta*8);
  m.x+=(m.targetX-m.x)*gazeBlend;m.y+=(m.targetY-m.y)*gazeBlend;
  m.height+=(mood[0]-m.height)*moodBlend;m.tilt+=(mood[1]-m.tilt)*moodBlend;
  if(m.blinkStart<0&&t>=m.nextBlink)m.blinkStart=t;
  blinkProgress=1;
  if(m.blinkStart>=0){
    const elapsed=t-m.blinkStart;
    // Quick closure, a short pause, then a softer reopening.
    if(elapsed<.085)blinkProgress=1-eyeEase(elapsed/.085);
    else if(elapsed<.12)blinkProgress=0;
    else if(elapsed<.29)blinkProgress=eyeEase((elapsed-.12)/.17);
    else {
      m.blinkStart=-1;
      const repeat=!m.doubleBlink&&Math.random()<.12;
      m.nextBlink=t+(repeat?.16:2.7+Math.random()*3.8);m.doubleBlink=repeat;
    }
  }
}
function updateFaceDisplay(delta=0) {
  if(!faceCtx)return;
  updateEyeMotion(delta);
  const ctx=faceCtx,m=eyeMotion;
  ctx.fillStyle='#050507';ctx.fillRect(0,0,faceCanvas.width,faceCanvas.height);
  ctx.save();ctx.fillStyle='#f1f3f7';ctx.shadowColor='#dde6ff';
  ctx.shadowBlur=6;
  // A pair of slim capsules is ADAM's resting face. No mouth or waveform.
  const eyeHeight=Math.max(2,m.height*blinkProgress);
  const cy=252+m.y;
  for(const [center,direction] of [[154,1],[358,-1]]){
    ctx.save();ctx.translate(center+m.x,cy);ctx.rotate(m.tilt*direction);
    drawRoundedRect(ctx,-m.width/2,-eyeHeight/2,m.width,eyeHeight,eyeHeight/2);
    ctx.restore();
  }
  ctx.restore();
  if(faceTexture)faceTexture.needsUpdate=true;
}

function drawRoundedRect(ctx, x, y, w, h, radius) {
  if (h <= 0) return;
  const r = Math.min(radius, h / 2, w / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fill();
}

function onWindowResize() {
  const container = document.getElementById('viewportContainer');
  if (!container || !camera || !renderer) return;
  const w = container.clientWidth;
  const h = container.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  updateSensorAnchors();
  request3DFrame();
}

function reset3DCamera() {
  if (camera && controls) {
    camera.position.set(1.3, 0.7, 4.5);
    controls.target.set(0, 0.02, 0);
    if (adamModel) adamModel.rotation.y = 0;
    controls.update();
    request3DFrame();
  }
}

function set3DStatus(robotState) {
  if (!robotState) return;
  if (typeof robotState.emotion === 'string') currentEmotion = robotState.emotion.toLowerCase();
  isSpeaking = Boolean(robotState.speaking);
  isListening = Boolean(robotState.listening);
  isConnected = Boolean(robotState.connected);
  request3DFrame();
}

function shouldRender3D() {
  return !!renderer && viewportVisible && viewportOnScreen && !document.hidden && !contextLost;
}
function request3DFrame() {
  if (!animationFrame && shouldRender3D()) animationFrame = requestAnimationFrame(animate);
}
function sync3DAnimation() {
  if (shouldRender3D()) request3DFrame();
  else { cancelAnimationFrame(animationFrame); animationFrame = 0; lastFrame = 0; }
}
function animate(timestamp = 0) {
  animationFrame = 0;
  if (!shouldRender3D()) return;
  if (!reducedMotion) request3DFrame();
  if (!reducedMotion && timestamp - lastFrame < 33) return;
  const delta = lastFrame ? Math.min((timestamp - lastFrame) / 1000, 0.1) : 0;
  lastFrame = timestamp;
  if (!reducedMotion) animTime += delta;

  // Update dynamic OLED face
  updateFaceDisplay(delta);

  if (controls) controls.update();
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
    projectSensorAnchors();
  }
}

window.init3D = init3D;
window.reset3DCamera = reset3DCamera;
window.viewADAMBack = () => {if(camera&&controls){camera.position.set(-1.3,.7,-4.5);controls.target.set(0,.02,0);controls.update();request3DFrame();}};
window.set3DStatus = set3DStatus;
window.set3DVisible = (visible) => { viewportVisible = visible; if (visible) onWindowResize(); sync3DAnimation(); };
