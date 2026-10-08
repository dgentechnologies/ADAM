/**
 * ADAM procedural 3D viewport and status mirror.
 * DGEN Technologies Pvt. Ltd.
 * Uses local geometry, monochrome materials and a dynamic OLED face canvas
 * with emotion mirroring (happy, sad, surprised, thinking, rizz, etc.), speaking
 * waveforms, and idle breathing animation.
 */

let scene, camera, renderer, controls;
let adamModel = null;
let headNode = null;
let faceMesh = null;
let faceCanvas = null;
let faceCtx = null;
let faceTexture = null;

// Live mirror state
let currentEmotion = 'happy';
let isSpeaking = false;
let isListening = false;
let isConnected = false;
let blinkProgress = 1.0; // 1 = open, 0 = closed
let lastBlinkTime = Date.now();
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
  camera.position.set(1.25, 0.85, 4.05);

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
  createParametricAdam();
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
  const loader = new THREE.GLTFLoader();
  const modelUrl = '/static/models/adam-body.glb';

  loader.load(
    modelUrl,
    (gltf) => {
      adamModel = gltf.scene;

      // Auto-fit & scale
      const box = new THREE.Box3().setFromObject(adamModel);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 1.3 / maxDim;

      adamModel.scale.setScalar(scale);
      adamModel.position.set(-center.x * scale, -center.y * scale, -center.z * scale);

      // Traverse meshes and setup materials
      adamModel.traverse((child) => {
        if (child.isMesh) {
          const name = (child.name || '').toLowerCase();

          // Check if OLED face screen (avoid matching outer faceplate bezel)
          if (name === 'facescreen' || name.includes('facescreen') || (name.includes('screen') && !name.includes('plate'))) {
            faceMesh = child;
            child.material = new THREE.MeshBasicMaterial({
              map: faceTexture,
              color: 0xffffff,
              transparent: false,
            });
          } else {
            // High-end Achromatic Charcoal Body / Bezel Material
            child.material = new THREE.MeshStandardMaterial({
              color: name.includes('faceplate') ? 0x242428 : 0x18181a,
              metalness: 0.35,
              roughness: 0.3,
            });
          }
        }
        if (child.name && child.name.toLowerCase().includes('head')) {
          headNode = child;
        }
      });

      // If no mesh matched 'facescreen' by name, find the front-most mesh
      if (!faceMesh) {
        attachFallbackFacePlane();
      }

      scene.add(adamModel);
      console.log('✅ ADAM 3D Model Loaded Successfully');
    },
    undefined,
    (error) => {
      console.warn('GLB Load failed, creating sleek parametric ADAM representation:', error);
      createParametricAdam();
    }
  );
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
    ctx.drawImage(brandImage,210,520,840,210,0,0,512,128);
    const pixels=ctx.getImageData(0,0,512,128);
    for(let i=0;i<pixels.data.length;i+=4)pixels.data[i+3]=Math.max(0,(Math.max(pixels.data[i],pixels.data[i+1],pixels.data[i+2])-25)*1.1);
    ctx.putImageData(pixels,0,0);markTexture.needsUpdate=true;
  };brandImage.src='/static/images/logo.png';
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

const sensorLocations={touch1:[-.59,.57,.12],touch2:[.59,.57,.12],touch3:[0,1.04,0],touch4:[.5,.78,-.16]};
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
    svg.append(path,ring,dot);sensorLayout.push({vector:new THREE.Vector3(...coords),start,left,path,ring,dot,width:box.width,height:box.height});
  }
  projectSensorAnchors();
}
function projectSensorAnchors(){
  if(!camera||!adamModel)return;
  adamModel.updateMatrixWorld();camera.updateMatrixWorld();
  for(const item of sensorLayout){
    const point=item.vector.clone().applyMatrix4(adamModel.matrixWorld).project(camera);
    const x=(point.x+1)*item.width/2,y=(1-point.y)*item.height/2;
    const elbow=item.start.x+(item.left?1:-1)*Math.min(36,Math.abs(x-item.start.x)*.3);
    item.path.setAttribute('d',`M ${item.start.x} ${item.start.y} L ${elbow} ${item.start.y} L ${x} ${y}`);
    for(const marker of [item.ring,item.dot]){marker.setAttribute('cx',x);marker.setAttribute('cy',y);}
  }
}
window.updateSensorAnchors=updateSensorAnchors;

function updateFaceDisplay() {
  if (!faceCtx) return;
  const ctx = faceCtx;
  const w = faceCanvas.width;
  const h = faceCanvas.height;

  // Background — Deep OLED Black with subtle scanlines
  ctx.fillStyle = '#060608';
  ctx.fillRect(0, 0, w, h);

  // Scanline grid
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.025)';
  ctx.lineWidth = 1;
  for (let y = 0; y < h; y += 8) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Handle blinking
  const now = Date.now();
  if (!reducedMotion && now - lastBlinkTime > 3500) {
    blinkProgress = Math.max(0.05, Math.sin((now - lastBlinkTime - 3500) * 0.015));
    if (now - lastBlinkTime > 3750) {
      blinkProgress = 1.0;
      lastBlinkTime = now + Math.random() * 2000;
    }
  }

  // Eye Style & Glow
  ctx.save();
  ctx.shadowColor = isConnected ? '#ffffff' : '#8e8e93';
  ctx.shadowBlur = isConnected ? 24 : 10;
  ctx.fillStyle = isConnected ? '#ffffff' : '#b0b0b5';

  const eyeCenterY = 220;
  const leftX = 160;
  const rightX = 352;

  // Draw Eyes based on current emotion
  drawEmotionEyes(ctx, leftX, rightX, eyeCenterY, currentEmotion, blinkProgress);

  // Draw Mouth / Speaking waveform
  if (isSpeaking) {
    const mouthWave = Math.sin(animTime * 15) * 12 + 14;
    ctx.fillStyle = '#ffffff';
    ctx.shadowBlur = 16;
    drawRoundedRect(ctx, w / 2 - 40, eyeCenterY + 85, 80, mouthWave, 6);
  } else if (currentEmotion === 'happy' || currentEmotion === 'rizz') {
    // Subtle smile line
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(w / 2, eyeCenterY + 70, 30, 0.2 * Math.PI, 0.8 * Math.PI);
    ctx.stroke();
  }

  ctx.restore();

  if (faceTexture) {
    faceTexture.needsUpdate = true;
  }
}

function drawEmotionEyes(ctx, lx, rx, cy, emotion, blink) {
  const baseW = 90;
  let baseH = 26 * blink;

  switch (emotion) {
    case 'happy': {
      // Upward happy crescents
      drawHappyEye(ctx, lx, cy, baseW, 28 * blink);
      drawHappyEye(ctx, rx, cy, baseW, 28 * blink);
      break;
    }
    case 'sad': {
      // Slanted downward bars
      ctx.save();
      ctx.translate(lx, cy);
      ctx.rotate(0.25);
      drawRoundedRect(ctx, -baseW / 2, -baseH / 2, baseW, baseH, 8);
      ctx.restore();

      ctx.save();
      ctx.translate(rx, cy);
      ctx.rotate(-0.25);
      drawRoundedRect(ctx, -baseW / 2, -baseH / 2, baseW, baseH, 8);
      ctx.restore();
      break;
    }
    case 'surprised': {
      // Wide open rounded capsules
      const rad = 36 * Math.max(0.2, blink);
      drawCircle(ctx, lx, cy, rad);
      drawCircle(ctx, rx, cy, rad);
      break;
    }
    case 'angry': {
      // Sharp inward angled eyes
      ctx.save();
      ctx.translate(lx, cy);
      ctx.rotate(-0.35);
      drawRoundedRect(ctx, -baseW / 2, -baseH / 2, baseW, baseH, 6);
      ctx.restore();

      ctx.save();
      ctx.translate(rx, cy);
      ctx.rotate(0.35);
      drawRoundedRect(ctx, -baseW / 2, -baseH / 2, baseW, baseH, 6);
      ctx.restore();
      break;
    }
    case 'thinking': {
      // One arched eye, one questioning dot
      drawRoundedRect(ctx, lx - baseW / 2, cy - 20, baseW, baseH, 8);
      const dotRadius = 18 + Math.sin(animTime * 6) * 4;
      drawCircle(ctx, rx, cy, dotRadius);
      break;
    }
    case 'rizz': {
      // Left eye raised smirk, right eye half squint
      drawRoundedRect(ctx, lx - baseW / 2, cy - 25, baseW, baseH * 1.2, 8);
      drawRoundedRect(ctx, rx - baseW / 2, cy + 5, baseW, baseH * 0.5, 4);
      break;
    }
    case 'sleep': {
      // Flat thin lines
      drawRoundedRect(ctx, lx - baseW / 2, cy, baseW, 4, 2);
      drawRoundedRect(ctx, rx - baseW / 2, cy, baseW, 4, 2);
      break;
    }
    case 'excited': {
      // Bouncing glowing pills
      const bounce = Math.sin(animTime * 12) * 8;
      drawRoundedRect(ctx, lx - baseW / 2, cy - 20 + bounce, baseW, 36 * blink, 12);
      drawRoundedRect(ctx, rx - baseW / 2, cy - 20 - bounce, baseW, 36 * blink, 12);
      break;
    }
    case 'reconnecting': {
      // Pulsing loading cycle
      const angle = animTime * 4;
      for (let i = 0; i < 6; i++) {
        const a = angle + (i * Math.PI) / 3;
        const dotX = 256 + Math.cos(a) * 60;
        const dotY = cy + Math.sin(a) * 60;
        drawCircle(ctx, dotX, dotY, 8 + i * 2);
      }
      return;
    }
    default: {
      // Standard neutral pill eyes
      drawRoundedRect(ctx, lx - baseW / 2, cy - baseH / 2, baseW, baseH, 8);
      drawRoundedRect(ctx, rx - baseW / 2, cy - baseH / 2, baseW, baseH, 8);
      break;
    }
  }
}

function drawHappyEye(ctx, cx, cy, w, h) {
  ctx.beginPath();
  ctx.arc(cx, cy + 10, w / 2, Math.PI * 1.15, Math.PI * 1.85, false);
  ctx.lineWidth = Math.max(6, h * 0.7);
  ctx.lineCap = 'round';
  ctx.strokeStyle = ctx.fillStyle;
  ctx.stroke();
}

function drawCircle(ctx, x, y, r) {
  ctx.beginPath();
  ctx.arc(x, y, Math.max(1, r), 0, Math.PI * 2);
  ctx.fill();
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
    camera.position.set(1.25, 0.85, 4.05);
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
  updateFaceDisplay();

  if (controls) controls.update();
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
    projectSensorAnchors();
  }
}

window.init3D = init3D;
window.reset3DCamera = reset3DCamera;
window.set3DStatus = set3DStatus;
window.set3DVisible = (visible) => { viewportVisible = visible; if (visible) onWindowResize(); sync3DAnimation(); };
