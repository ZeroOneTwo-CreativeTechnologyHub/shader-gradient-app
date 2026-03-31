import { useState, useEffect, useRef, useCallback } from "react";
import * as THREE from "three";

/* ─── 3D Model Loaders (standalone app only — install three-stdlib) ─── */
let GLTFLoader, OBJLoader, STLLoader;
let loadersChecked = false;
async function ensureLoaders() {
  if (loadersChecked) return;
  loadersChecked = true;
  try {
    const stdlib = await import("three-stdlib");
    GLTFLoader = stdlib.GLTFLoader;
    OBJLoader = stdlib.OBJLoader;
    STLLoader = stdlib.STLLoader;
  } catch (e) {
    // Loaders unavailable in preview — model loading disabled
  }
}

/* ─── Noise GLSL ─── */
const noiseGLSL = `
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x,289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0);const vec4 D=vec4(0,0.5,1,2);
  vec3 i=floor(v+dot(v,C.yyy)),x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz),l=1.0-g;
  vec3 i1=min(g,l.zxy),i2=max(g,l.zxy);
  vec3 x1=x0-i1+C.xxx,x2=x0-i2+C.yyy,x3=x0-D.yyy;
  i=mod(i,289.0);
  vec4 p=permute(permute(permute(
    i.z+vec4(0,i1.z,i2.z,1))+i.y+vec4(0,i1.y,i2.y,1))+i.x+vec4(0,i1.x,i2.x,1));
  float n_=1.0/7.0;vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z),y_=floor(j-7.0*x_);
  vec4 xx=x_*ns.x+ns.yyyy,yy=y_*ns.x+ns.yyyy,h=1.0-abs(xx)-abs(yy);
  vec4 b0=vec4(xx.xy,yy.xy),b1=vec4(xx.zw,yy.zw);
  vec4 s0=floor(b0)*2.0+1.0,s1=floor(b1)*2.0+1.0;
  vec4 sh=-step(h,vec4(0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy,a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x),p1=vec3(a0.zw,h.y),p2=vec3(a1.xy,h.z),p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);
  m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
vec3 rotateY(vec3 v,float a){float c=cos(a),s=sin(a);return vec3(v.x*c+v.z*s,v.y,-v.x*s+v.z*c);}
`;

/* ─── Displacement per shape ─── */
const displacementCode = {
  plane:`float d=snoise(normal*uDensity+t)*uStrength;transformed+=normal*d;float a=sin(uv.y*uFrequency+t)*uAmplitude;transformed=rotateY(transformed,a);`,
  water:`float w1=sin(transformed.x*2.0+t*1.2)*cos(transformed.y*1.5+t*0.8)*0.3;float w2=sin(transformed.x*4.0-t*0.9+1.0)*cos(transformed.y*3.0+t*0.6)*0.15;float w3=snoise(vec3(transformed.xy*uDensity*0.8,t*0.5))*0.2;transformed.z+=(w1+w2+w3)*uStrength;transformed.x+=snoise(vec3(transformed.y*0.5,t*0.3,0.0))*uAmplitude*0.2;`,
  sphere:`float d=snoise(normal*uDensity+t)*uStrength;transformed+=normal*d;float a=sin(uv.y*uFrequency+t)*uAmplitude;transformed=rotateY(transformed,a);`,
  torus:`float d=snoise(normal*uDensity*1.5+t)*uStrength*0.5;transformed+=normal*d;float tw=sin(uv.x*6.2832*2.0+t)*uAmplitude*0.3;transformed=rotateY(transformed,tw);`,
  torusKnot:`float d=snoise(normal*uDensity*2.0+t)*uStrength*0.3;transformed+=normal*d;float p=sin(uv.x*6.2832*3.0+t*1.5)*uAmplitude*0.15;transformed+=normal*p;`,
  icosahedron:`float d=snoise(normal*uDensity+t)*uStrength*0.6;transformed+=normal*d;float a=sin(uv.y*uFrequency*0.5+t)*uAmplitude*0.5;transformed=rotateY(transformed,a);`,
  cylinder:`float d=snoise(vec3(normal.xy*uDensity,t))*uStrength*0.5;transformed+=normal*d;float b=sin(uv.y*uFrequency*1.5+t)*uAmplitude*0.3;transformed.x+=b*normal.x;transformed.z+=b*normal.z;`,
  blob:`float n1=snoise(normal*uDensity+t)*uStrength;float n2=snoise(normal*uDensity*2.0+t*1.3+5.0)*uStrength*0.5;float n3=snoise(normal*uDensity*4.0+t*0.7+10.0)*uStrength*0.25;transformed+=normal*(n1+n2+n3);float a=sin(uv.y*uFrequency+t)*uAmplitude;transformed=rotateY(transformed,a);`,
  custom:`float n1=snoise(normal*uDensity+t)*uStrength;float n2=snoise(normal*uDensity*2.0+t*1.3+5.0)*uStrength*0.4;transformed+=normal*(n1+n2);float a=sin(uv.y*uFrequency+t)*uAmplitude*0.5;transformed=rotateY(transformed,a);`,
};

const shapeTypes = ["plane","water","sphere","torus","torusKnot","icosahedron","cylinder","blob","custom"];
const shapeLabels = {plane:"Plane",water:"Water",sphere:"Sphere",torus:"Torus",torusKnot:"Knot",icosahedron:"Icosa",cylinder:"Cylinder",blob:"Blob",custom:"Model"};

/* ─── Audio Engine ─── */
class AudioEngine {
  constructor(){
    this.ctx=null;this.analyser=null;this.source=null;this.stream=null;
    this.fftSize=2048;this.data=null;this.active=false;
    this.bass=0;this.mid=0;this.treble=0;this.volume=0;
    this.smoothBass=0;this.smoothMid=0;this.smoothTreble=0;
    this.kick=0;this.kickDecay=0;this.prevBass=0;
    this.kickThreshold=0.3;this.kickDecayRate=0.92;this.smoothing=0.8;this.gain=1.0;
  }
  async start(){
    try{
      this.stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
      this.ctx=new(window.AudioContext||window.webkitAudioContext)();
      this.analyser=this.ctx.createAnalyser();this.analyser.fftSize=this.fftSize;this.analyser.smoothingTimeConstant=0.6;
      this.source=this.ctx.createMediaStreamSource(this.stream);this.source.connect(this.analyser);
      this.data=new Uint8Array(this.analyser.frequencyBinCount);this.active=true;return true;
    }catch(e){console.error("Audio:",e);return false;}
  }
  stop(){
    if(this.source)this.source.disconnect();if(this.stream)this.stream.getTracks().forEach(t=>t.stop());
    if(this.ctx)this.ctx.close();this.active=false;this.bass=this.mid=this.treble=this.kick=0;
    this.smoothBass=this.smoothMid=this.smoothTreble=0;
  }
  update(){
    if(!this.active||!this.analyser)return;
    this.analyser.getByteFrequencyData(this.data);
    const n=this.data.length,sr=this.ctx.sampleRate,bHz=sr/this.fftSize;
    const bE=Math.min(Math.floor(250/bHz),n),mE=Math.min(Math.floor(4000/bHz),n),tE=Math.min(Math.floor(16000/bHz),n);
    let bS=0,mS=0,tS=0,bC=0,mC=0,tC=0;
    for(let i=1;i<n;i++){const v=this.data[i]/255;if(i<bE){bS+=v;bC++;}else if(i<mE){mS+=v;mC++;}else if(i<tE){tS+=v;tC++;}}
    this.bass=(bC?bS/bC:0)*this.gain;this.mid=(mC?mS/mC:0)*this.gain;this.treble=(tC?tS/tC:0)*this.gain;
    this.volume=(this.bass+this.mid+this.treble)/3;
    const s=this.smoothing;
    this.smoothBass=this.smoothBass*s+this.bass*(1-s);this.smoothMid=this.smoothMid*s+this.mid*(1-s);this.smoothTreble=this.smoothTreble*s+this.treble*(1-s);
    const bd=this.bass-this.prevBass;if(bd>this.kickThreshold)this.kickDecay=1.0;
    this.kickDecay*=this.kickDecayRate;this.kick=this.kickDecay;this.prevBass=this.bass;
  }
}

/* ─── Model Loading ─── */
function centerAndScaleGeo(geo, targetSize = 3) {
  geo.computeBoundingBox();
  const box = geo.boundingBox;
  const center = new THREE.Vector3();
  box.getCenter(center);
  geo.translate(-center.x, -center.y, -center.z);
  const size = new THREE.Vector3();
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z);
  if (maxDim > 0) {
    const scale = targetSize / maxDim;
    geo.scale(scale, scale, scale);
  }
  geo.computeBoundingBox();
  geo.computeVertexNormals();
  return geo;
}

function extractGeometry(object) {
  const geometries = [];
  object.traverse((child) => {
    if (child.isMesh && child.geometry) {
      const geo = child.geometry.clone();
      // Apply any transforms from the mesh hierarchy
      child.updateWorldMatrix(true, false);
      geo.applyMatrix4(child.matrixWorld);
      // Ensure it's indexed or non-indexed consistently
      if (geo.index) {
        geometries.push(geo.toNonIndexed());
      } else {
        geometries.push(geo);
      }
    }
  });
  if (geometries.length === 0) return null;
  if (geometries.length === 1) return geometries[0];
  // Merge all geometries — manual merge for compatibility
  const merged = mergeGeos(geometries);
  return merged;
}

function mergeGeos(geos) {
  let totalVerts = 0;
  for (const g of geos) totalVerts += g.attributes.position.count;
  const pos = new Float32Array(totalVerts * 3);
  const norm = new Float32Array(totalVerts * 3);
  const uv = new Float32Array(totalVerts * 2);
  let vOff = 0, uOff = 0;
  for (const g of geos) {
    const c = g.attributes.position.count;
    const p = g.attributes.position.array;
    const n = g.attributes.normal ? g.attributes.normal.array : null;
    const u = g.attributes.uv ? g.attributes.uv.array : null;
    for (let i = 0; i < c * 3; i++) {
      pos[vOff + i] = p[i];
      norm[vOff + i] = n ? n[i] : 0;
    }
    for (let i = 0; i < c * 2; i++) {
      uv[uOff + i] = u ? u[i] : 0;
    }
    vOff += c * 3;
    uOff += c * 2;
  }
  const merged = new THREE.BufferGeometry();
  merged.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  merged.setAttribute('normal', new THREE.BufferAttribute(norm, 3));
  merged.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  return merged;
}

async function loadModelFile(file) {
  await ensureLoaders();
  const ext = file.name.split('.').pop().toLowerCase();
  const url = URL.createObjectURL(file);

  try {
    if ((ext === 'glb' || ext === 'gltf') && GLTFLoader) {
      const loader = new GLTFLoader();
      const gltf = await new Promise((res, rej) => loader.load(url, res, undefined, rej));
      const geo = extractGeometry(gltf.scene);
      URL.revokeObjectURL(url);
      return geo ? centerAndScaleGeo(geo) : null;
    }
    if (ext === 'obj' && OBJLoader) {
      const loader = new OBJLoader();
      const obj = await new Promise((res, rej) => loader.load(url, res, undefined, rej));
      const geo = extractGeometry(obj);
      URL.revokeObjectURL(url);
      return geo ? centerAndScaleGeo(geo) : null;
    }
    if (ext === 'stl' && STLLoader) {
      const loader = new STLLoader();
      const geo = await new Promise((res, rej) => loader.load(url, res, undefined, rej));
      URL.revokeObjectURL(url);
      return geo ? centerAndScaleGeo(geo) : null;
    }
    URL.revokeObjectURL(url);
    return null;
  } catch (e) {
    console.error("Model load error:", e);
    URL.revokeObjectURL(url);
    return null;
  }
}

/* ─── Env map, vertex colors, material, grain ─── */
function createEnvMap(renderer,preset){
  const sc=new THREE.Scene();
  const ps={city:{sky:[.42,.55,.75],hz:[.75,.65,.55],gnd:[.22,.2,.18],sun:[1,.9,.7],sd:[.5,.3,.8]},dawn:{sky:[.35,.35,.55],hz:[.85,.55,.35],gnd:[.15,.12,.12],sun:[1,.6,.3],sd:[0,.1,1]},lobby:{sky:[.3,.3,.35],hz:[.5,.48,.45],gnd:[.25,.22,.2],sun:[.9,.85,.8],sd:[.3,.6,.5]}};
  const p=ps[preset]||ps.city;const g=new THREE.SphereGeometry(50,64,32);
  const m=new THREE.ShaderMaterial({side:THREE.BackSide,uniforms:{sk:{value:new THREE.Vector3(...p.sky)},hz:{value:new THREE.Vector3(...p.hz)},gd:{value:new THREE.Vector3(...p.gnd)},sc:{value:new THREE.Vector3(...p.sun)},sd:{value:new THREE.Vector3(...p.sd).normalize()}},
    vertexShader:`varying vec3 vD;void main(){vD=normalize(position);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1);}`,
    fragmentShader:`uniform vec3 sk,hz,gd,sc,sd;varying vec3 vD;void main(){vec3 d=normalize(vD);float y=d.y;vec3 c;if(y>0.0)c=mix(hz,sk,pow(y,0.6));else c=mix(hz*0.8,gd,pow(-y,0.4));float s=max(dot(d,sd),0.0);c+=sc*pow(s,32.0)*0.8+sc*pow(s,4.0)*0.15;float bn=sin(d.x*12.0)*sin(d.z*8.0);float bm=smoothstep(-0.05,0.15,y)*smoothstep(0.4,0.0,y);c+=vec3(0.12,0.08,0.04)*max(bn,0.0)*bm;gl_FragColor=vec4(c,1);}`});
  sc.add(new THREE.Mesh(g,m));
  const crt=new THREE.WebGLCubeRenderTarget(256,{format:THREE.RGBFormat,generateMipmaps:true,minFilter:THREE.LinearMipmapLinearFilter});
  new THREE.CubeCamera(0.1,100,crt).update(renderer,sc);
  const pm=new THREE.PMREMGenerator(renderer);pm.compileCubemapShader();
  const e=pm.fromCubemap(crt.texture).texture;pm.dispose();crt.dispose();g.dispose();m.dispose();return e;
}
function applyVC(geo,c1h,c2h,c3h,density){
  const c1=new THREE.Color(c1h),c2=new THREE.Color(c2h),c3=new THREE.Color(c3h);
  const pos=geo.attributes.position,ct=pos.count,cols=new Float32Array(ct*3);
  for(let i=0;i<ct;i++){
    const px=pos.getX(i),py=pos.getY(i),pz=pos.getZ(i);
    const n1=Math.sin(px*density*1.7+py*0.9+pz*0.5+0.3)*0.5+0.5;
    const n2=Math.cos(py*density*1.4-px*0.7+pz*0.3+1.9)*0.5+0.5;
    const n3=Math.sin((px*0.8+py*1.1+pz*0.6)*density+2.7)*0.5+0.5;
    const w1=n1*n1,w2=n2*(1-w1*0.4),w3=n3*(1-n1*0.3)*(1-n2*0.3),t=w1+w2+w3+.001;
    cols[i*3]=(c1.r*w1+c2.r*w2+c3.r*w3)/t;cols[i*3+1]=(c1.g*w1+c2.g*w2+c3.g*w3)/t;cols[i*3+2]=(c1.b*w1+c2.b*w2+c3.b*w3)/t;
  }
  geo.setAttribute('color',new THREE.BufferAttribute(cols,3));
}
function createMat(envMap,shapeType,cu){
  const m=new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:0.1,metalness:0,envMapIntensity:1,envMap});
  m.onBeforeCompile=sh=>{
    sh.uniforms.uTime=cu.uTime;sh.uniforms.uSpeed=cu.uSpeed;sh.uniforms.uDensity=cu.uDensity;
    sh.uniforms.uStrength=cu.uStrength;sh.uniforms.uFrequency=cu.uFrequency;sh.uniforms.uAmplitude=cu.uAmplitude;
    sh.vertexShader=sh.vertexShader.replace('void main() {',`uniform float uTime,uSpeed,uDensity,uStrength,uFrequency,uAmplitude;\n${noiseGLSL}\nvoid main() {`);
    sh.vertexShader=sh.vertexShader.replace('#include <begin_vertex>',`#include <begin_vertex>\nfloat t=uTime*uSpeed;\n${displacementCode[shapeType]||displacementCode.custom}`);
  };
  m.needsUpdate=true;return m;
}
function makeGeo(type){
  switch(type){
    case'sphere':return new THREE.SphereGeometry(1.5,200,200);
    case'water':return new THREE.PlaneGeometry(8,8,400,400);
    case'torus':return new THREE.TorusGeometry(1.2,0.5,128,200);
    case'torusKnot':return new THREE.TorusKnotGeometry(1,0.35,256,64);
    case'icosahedron':return new THREE.IcosahedronGeometry(1.6,40);
    case'cylinder':return new THREE.CylinderGeometry(1,1,3,128,128,true);
    case'blob':return new THREE.SphereGeometry(1.3,200,200);
    default:return new THREE.PlaneGeometry(6,6,300,300);
  }
}
function createGrain(){
  return new THREE.ShaderMaterial({transparent:true,depthTest:false,depthWrite:false,
    uniforms:{uTime:{value:0},uGrain:{value:.5},uBlend:{value:.5}},
    vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position,1);}`,
    fragmentShader:`uniform float uTime,uGrain,uBlend;varying vec2 vUv;
    float h(vec2 p){vec3 p3=fract(vec3(p.xyx)*0.1031);p3+=dot(p3,p3.yzx+33.33);return fract((p3.x+p3.y)*p3.z);}
    void main(){float g=h(gl_FragCoord.xy+fract(uTime*7.13))*2.0-1.0;float g2=h(gl_FragCoord.xy*1.3+fract(uTime*3.77+.5))*2.0-1.0;
    float gr=mix(g,g2,0.4);float a=abs(gr)*uGrain*uBlend;vec3 c=gr>0.0?vec3(1):vec3(0);gl_FragColor=vec4(c,a);}`});
}

/* ─── UI ─── */
const Sl=({label,value,min,max,step:s,onChange:f})=>(
  <div style={{marginBottom:5}}>
    <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"#777",marginBottom:1}}>
      <span>{label}</span><span style={{fontFamily:"monospace",fontSize:9}}>{typeof value==='number'?value.toFixed(s<0.1?2:1):value}</span>
    </div>
    <input type="range" min={min} max={max} step={s} value={value} onChange={e=>f(+e.target.value)}
      style={{width:"100%",accentColor:"#555",height:2,cursor:"pointer"}}/>
  </div>
);
const CI=({label,value,onChange:f})=>(
  <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:4}}>
    <input type="color" value={value} onChange={e=>f(e.target.value)}
      style={{width:22,height:22,border:"1px solid #333",borderRadius:3,cursor:"pointer",padding:0,background:"none"}}/>
    <span style={{fontSize:10,color:"#777"}}>{label}</span>
  </div>
);
const Sec=({title,children,open:io=true})=>{
  const[o,sO]=useState(io);
  return(<div style={{marginBottom:2}}>
    <div onClick={()=>sO(!o)} style={{cursor:"pointer",fontSize:9,fontWeight:700,textTransform:"uppercase",
      letterSpacing:1.5,color:"#555",padding:"5px 0",borderBottom:"1px solid #1a1a1a",
      display:"flex",justifyContent:"space-between",userSelect:"none"}}>
      <span>{title}</span><span style={{color:"#333"}}>{o?"−":"+"}</span></div>
    {o&&<div style={{padding:"6px 0"}}>{children}</div>}
  </div>);
};
const Meter=({label,value,color})=>(
  <div style={{marginBottom:3}}>
    <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#666",marginBottom:1}}>
      <span>{label}</span><span style={{fontFamily:"monospace"}}>{(value*100).toFixed(0)}%</span></div>
    <div style={{height:4,background:"#1a1a1a",borderRadius:2,overflow:"hidden"}}>
      <div style={{height:"100%",width:`${Math.min(value*100,100)}%`,background:color,borderRadius:2,transition:"width 0.05s"}}/></div>
  </div>
);
const mapTargets=[{key:"none",label:"None"},{key:"strength",label:"Strength"},{key:"density",label:"Density"},{key:"frequency",label:"Frequency"},{key:"amplitude",label:"Amplitude"},{key:"speed",label:"Speed"},{key:"rotX",label:"Rotation X"},{key:"rotY",label:"Rotation Y"},{key:"rotZ",label:"Rotation Z"},{key:"camDist",label:"Cam Distance"},{key:"brightness",label:"Brightness"},{key:"roughness",label:"Roughness"},{key:"reflection",label:"Env Reflect"},{key:"grain",label:"Grain"}];
const MapRow=({label,color,target,amount,onTarget,onAmount})=>(
  <div style={{marginBottom:6,padding:"4px 6px",background:"#0d0d0d",borderRadius:4,border:"1px solid #1a1a1a"}}>
    <div style={{fontSize:9,color,fontWeight:700,marginBottom:3}}>{label}</div>
    <div style={{display:"flex",gap:4,alignItems:"center"}}>
      <select value={target} onChange={e=>onTarget(e.target.value)}
        style={{flex:1,fontSize:9,background:"#111",color:"#999",border:"1px solid #2a2a2a",borderRadius:3,padding:"3px 4px",cursor:"pointer"}}>
        {mapTargets.map(t=><option key={t.key} value={t.key}>{t.label}</option>)}</select>
      <input type="range" min={0} max={3} step={0.05} value={amount} onChange={e=>onAmount(+e.target.value)}
        style={{width:60,accentColor:color,height:2,cursor:"pointer"}}/>
      <span style={{fontSize:8,color:"#555",fontFamily:"monospace",width:28,textAlign:"right"}}>{amount.toFixed(1)}x</span>
    </div>
  </div>
);

/* ─── MAIN ─── */
export default function App(){
  const mountRef=useRef(null);const sceneRef=useRef({});const paramsRef=useRef(null);
  const audioRef=useRef(new AudioEngine());
  const customGeoRef=useRef(null);
  const[showUI,setShowUI]=useState(true);const[outputMode,setOutputMode]=useState(false);
  const[audioActive,setAudioActive]=useState(false);
  const[audioLevels,setAudioLevels]=useState({bass:0,mid:0,treble:0,kick:0});
  const[modelName,setModelName]=useState(null);
  const[modelLoading,setModelLoading]=useState(false);
  const[dragOver,setDragOver]=useState(false);

  const defaults={
    type:"plane",color1:"#ff5005",color2:"#dbba95",color3:"#d0bce1",
    speed:0.4,strength:4.0,density:1.3,frequency:5.5,amplitude:1.0,
    rotX:0,rotY:10,rotZ:50,posX:-1.4,posY:0,posZ:0,
    camDist:2.4,camAzimuth:180,camPolar:90,zoom:1.0,fov:45,
    brightness:1.2,grain:0.5,grainBlend:0.5,reflection:0.1,
    roughness:0.1,metalness:0.0,envPreset:"city",lightType:0,wireframe:false,bgColor:"#000000",
    modelScale:1.0,modelSubdivide:false,
    audioGain:1.5,audioSmoothing:0.8,kickThreshold:0.3,kickDecay:0.92,
    bassTarget:"strength",bassAmount:1.0,midTarget:"frequency",midAmount:0.5,
    trebleTarget:"grain",trebleAmount:0.8,kickTarget:"rotZ",kickAmount:1.5,
  };
  const[p,setP]=useState(defaults);
  paramsRef.current=p;
  const sp=useCallback((k,v)=>setP(prev=>({...prev,[k]:v})),[]);
  const lastColorsRef=useRef("");const lastEnvRef=useRef("");const lastTypeRef=useRef("");

  const toggleAudio=useCallback(async()=>{
    const ae=audioRef.current;
    if(ae.active){ae.stop();setAudioActive(false);}
    else{const ok=await ae.start();setAudioActive(ok);}
  },[]);

  // Handle model file
  const handleModelFile = useCallback(async (file) => {
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['glb','gltf','obj','stl'].includes(ext)) {
      alert('Supported formats: .glb, .gltf, .obj, .stl');
      return;
    }
    setModelLoading(true);
    const geo = await loadModelFile(file);
    setModelLoading(false);
    if (geo) {
      // Ensure UVs exist (some models lack them)
      if (!geo.attributes.uv) {
        const count = geo.attributes.position.count;
        const uvs = new Float32Array(count * 2);
        for (let i = 0; i < count; i++) {
          const pos = geo.attributes.position;
          uvs[i * 2] = (pos.getX(i) + 1.5) / 3;
          uvs[i * 2 + 1] = (pos.getY(i) + 1.5) / 3;
        }
        geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
      }
      customGeoRef.current = geo;
      setModelName(file.name);
      sp("type", "custom");
    } else {
      alert('Failed to load model. Check console for details.');
    }
  }, [sp]);

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) handleModelFile(file);
  }, [handleModelFile]);

  useEffect(()=>{
    const h=e=>{if(e.key==="Escape"&&outputMode){setOutputMode(false);setShowUI(true);}};
    window.addEventListener("keydown",h);return()=>{window.removeEventListener("keydown",h);audioRef.current.stop();};
  },[outputMode]);

  useEffect(()=>{
    if(!audioActive)return;
    const iv=setInterval(()=>{
      const ae=audioRef.current;
      setAudioLevels({bass:ae.smoothBass,mid:ae.smoothMid,treble:ae.smoothTreble,kick:ae.kick});
    },50);
    return()=>clearInterval(iv);
  },[audioActive]);

  useEffect(()=>{
    const el=mountRef.current;if(!el)return;
    const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.setClearColor(0,1);
    renderer.outputEncoding=THREE.sRGBEncoding;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1;
    el.appendChild(renderer.domElement);
    const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(45,1,0.1,100);
    const cu={uTime:{value:0},uSpeed:{value:.4},uStrength:{value:4},uDensity:{value:1.3},uFrequency:{value:5.5},uAmplitude:{value:1}};
    let envMap=createEnvMap(renderer,"city");scene.environment=envMap;lastEnvRef.current="city";
    const aL=new THREE.AmbientLight(0xffffff,0.5);scene.add(aL);
    const dL=new THREE.DirectionalLight(0xfff5e8,0.6);dL.position.set(5,5,5);scene.add(dL);
    const dL2=new THREE.DirectionalLight(0xc8d8ff,0.2);dL2.position.set(-3,2,-2);scene.add(dL2);
    let geo=makeGeo("plane");applyVC(geo,"#ff5005","#dbba95","#d0bce1",1.3);
    let mat=createMat(envMap,"plane",cu);let mesh=new THREE.Mesh(geo,mat);scene.add(mesh);lastTypeRef.current="plane";
    const grainMat=createGrain();const gS=new THREE.Scene();const gC=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
    gS.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),grainMat));
    sceneRef.current={renderer,scene,camera,mesh,mat,geo,cu,envMap,grainMat,gS,gC,aL,dL,dL2};

    const resize=()=>{const w=el.clientWidth,h=el.clientHeight;if(!w||!h)return;camera.aspect=w/h;camera.updateProjectionMatrix();renderer.setSize(w,h);};
    resize();window.addEventListener("resize",resize);

    const clock=new THREE.Clock();let raf;
    const loop=()=>{
      raf=requestAnimationFrame(loop);
      const cp=paramsRef.current;const s=sceneRef.current;const elapsed=clock.getElapsedTime();
      const ae=audioRef.current;ae.gain=cp.audioGain;ae.smoothing=cp.audioSmoothing;ae.kickThreshold=cp.kickThreshold;ae.kickDecayRate=cp.kickDecay;ae.update();
      const am={};
      const aM=(b,tgt,amt)=>{if(tgt==="none"||!amt)return;am[tgt]=(am[tgt]||0)+b*amt;};
      aM(ae.smoothBass,cp.bassTarget,cp.bassAmount);aM(ae.smoothMid,cp.midTarget,cp.midAmount);
      aM(ae.smoothTreble,cp.trebleTarget,cp.trebleAmount);aM(ae.kick,cp.kickTarget,cp.kickAmount);
      const gP=(k,b)=>b+(am[k]||0);

      cu.uTime.value=elapsed;cu.uSpeed.value=gP("speed",cp.speed);cu.uStrength.value=gP("strength",cp.strength);
      cu.uDensity.value=gP("density",cp.density);cu.uFrequency.value=gP("frequency",cp.frequency);cu.uAmplitude.value=gP("amplitude",cp.amplitude);
      s.mat.roughness=Math.max(0,Math.min(1,gP("roughness",cp.roughness)));s.mat.metalness=Math.max(0,Math.min(1,cp.metalness));
      s.mat.envMapIntensity=gP("reflection",cp.reflection)*10;s.mat.wireframe=cp.wireframe;
      grainMat.uniforms.uTime.value=elapsed;grainMat.uniforms.uGrain.value=Math.max(0,gP("grain",cp.grain));grainMat.uniforms.uBlend.value=cp.grainBlend;
      renderer.setClearColor(new THREE.Color(cp.bgColor),1);renderer.toneMappingExposure=Math.max(.1,gP("brightness",cp.brightness));camera.fov=cp.fov;
      s.aL.intensity=cp.lightType===0?.5:.2;s.dL.intensity=cp.lightType===0?.6:.1;s.dL2.intensity=cp.lightType===0?.2:.05;

      if(cp.envPreset!==lastEnvRef.current){if(s.envMap)s.envMap.dispose();s.envMap=createEnvMap(renderer,cp.envPreset);scene.environment=s.envMap;s.mat.envMap=s.envMap;s.mat.needsUpdate=true;lastEnvRef.current=cp.envPreset;}

      if(cp.type!==lastTypeRef.current){
        scene.remove(s.mesh);s.geo.dispose();s.mat.dispose();
        if(cp.type==="custom"&&customGeoRef.current){
          s.geo=customGeoRef.current.clone();
          s.geo.scale(cp.modelScale,cp.modelScale,cp.modelScale);
        }else{
          s.geo=makeGeo(cp.type);
        }
        applyVC(s.geo,cp.color1,cp.color2,cp.color3,cp.density);
        s.mat=createMat(s.envMap,cp.type,cu);s.mat.roughness=cp.roughness;s.mat.metalness=cp.metalness;
        s.mat.envMapIntensity=cp.reflection*10;s.mat.wireframe=cp.wireframe;
        s.mesh=new THREE.Mesh(s.geo,s.mat);scene.add(s.mesh);lastTypeRef.current=cp.type;lastColorsRef.current="";
      }

      const ck=cp.color1+cp.color2+cp.color3+cp.density.toFixed(2);
      if(ck!==lastColorsRef.current){applyVC(s.geo,cp.color1,cp.color2,cp.color3,cp.density);lastColorsRef.current=ck;}

      s.mesh.rotation.set(gP("rotX",cp.rotX)*Math.PI/180,gP("rotY",cp.rotY)*Math.PI/180,gP("rotZ",cp.rotZ)*Math.PI/180);
      s.mesh.position.set(cp.posX,cp.posY,cp.posZ);
      const az=cp.camAzimuth*Math.PI/180,po=cp.camPolar*Math.PI/180,dist=gP("camDist",cp.camDist);
      camera.position.set(dist*Math.sin(po)*Math.sin(az),dist*Math.cos(po),dist*Math.sin(po)*Math.cos(az));
      camera.lookAt(0,0,0);camera.zoom=cp.zoom;camera.updateProjectionMatrix();
      renderer.autoClear=true;renderer.render(scene,camera);
      if(cp.grain>0.001){renderer.autoClear=false;renderer.render(s.gS,s.gC);renderer.autoClear=true;}
    };
    loop();
    return()=>{cancelAnimationFrame(raf);window.removeEventListener("resize",resize);renderer.dispose();if(el.contains(renderer.domElement))el.removeChild(renderer.domElement);};
  },[]);

  const presets=[
    {name:"Default",color1:"#ff5005",color2:"#dbba95",color3:"#d0bce1",speed:.4,strength:4,density:1.3,frequency:5.5,amplitude:1,type:"plane",rotX:0,rotY:10,rotZ:50,posX:-1.4,camDist:2.4,camAzimuth:180,camPolar:90,fov:45,brightness:1.2,grain:.5,grainBlend:.5,reflection:.1,roughness:.1,metalness:0,envPreset:"city",lightType:0,bgColor:"#000000"},
    {name:"Ocean",color1:"#003388",color2:"#00bbee",color3:"#001133",speed:.2,strength:2.5,density:.8,frequency:3,amplitude:.6,type:"water",rotX:55,rotY:0,rotZ:0,posX:0,camDist:3,camAzimuth:180,camPolar:65,fov:45,brightness:1.3,grain:.2,grainBlend:.3,reflection:.3,roughness:.05,metalness:0,envPreset:"dawn",lightType:0,bgColor:"#000a11"},
    {name:"Molten Orb",color1:"#ff5522",color2:"#ff0066",color3:"#220011",speed:.25,strength:3,density:1,frequency:4,amplitude:1.2,type:"sphere",rotX:0,rotY:0,rotZ:0,posX:0,camDist:3.8,camAzimuth:200,camPolar:80,fov:45,brightness:1.1,grain:.15,grainBlend:.3,reflection:.4,roughness:.05,metalness:.1,envPreset:"city",lightType:0,bgColor:"#080004"},
    {name:"Neon Torus",color1:"#ff00ff",color2:"#00ffff",color3:"#110022",speed:.5,strength:3,density:1.5,frequency:5,amplitude:.6,type:"torus",rotX:25,rotY:30,rotZ:0,posX:0,camDist:3.5,camAzimuth:180,camPolar:75,fov:45,brightness:1,grain:0,grainBlend:0,reflection:.5,roughness:.02,metalness:.2,envPreset:"lobby",lightType:0,bgColor:"#000"},
    {name:"Cosmic Knot",color1:"#9933ff",color2:"#ff6600",color3:"#003366",speed:.3,strength:2,density:1.2,frequency:4,amplitude:.5,type:"torusKnot",rotX:15,rotY:20,rotZ:10,posX:0,camDist:3.5,camAzimuth:190,camPolar:80,fov:45,brightness:1.2,grain:.2,grainBlend:.4,reflection:.35,roughness:.08,metalness:.05,envPreset:"city",lightType:0,bgColor:"#050008"},
    {name:"Warm Blob",color1:"#ff8844",color2:"#ffcc66",color3:"#cc3355",speed:.35,strength:5,density:1.5,frequency:6,amplitude:1.5,type:"blob",rotX:0,rotY:0,rotZ:0,posX:0,camDist:3.5,camAzimuth:200,camPolar:80,fov:45,brightness:1.1,grain:.3,grainBlend:.4,reflection:.2,roughness:.1,metalness:0,envPreset:"dawn",lightType:0,bgColor:"#0a0400"},
  ];

  const fileInputRef = useRef(null);

  return(
    <div style={{width:"100%",height:"100vh",background:"#000",position:"relative",overflow:"hidden",fontFamily:"-apple-system,BlinkMacSystemFont,sans-serif"}}
      onDragOver={e=>{e.preventDefault();setDragOver(true);}}
      onDragLeave={()=>setDragOver(false)}
      onDrop={handleDrop}>

      <div ref={mountRef} style={{position:"absolute",inset:0,zIndex:0}}/>

      {/* Drag overlay */}
      {dragOver&&<div style={{position:"absolute",inset:0,zIndex:200,background:"rgba(0,180,255,0.15)",border:"3px dashed rgba(0,180,255,0.6)",borderRadius:12,display:"flex",alignItems:"center",justifyContent:"center",pointerEvents:"none"}}>
        <div style={{fontSize:18,color:"#0bf",fontWeight:700,textShadow:"0 2px 10px rgba(0,0,0,0.5)"}}>Drop 3D model here</div>
      </div>}

      {outputMode&&<div onClick={()=>{setOutputMode(false);setShowUI(true);}} style={{position:"absolute",top:0,left:0,right:0,zIndex:100,background:"rgba(220,30,30,0.85)",padding:"5px 0",textAlign:"center",cursor:"pointer",fontSize:10,color:"#fff",fontWeight:700,letterSpacing:1}}>OUTPUT MODE — Click or Esc to exit</div>}
      {!showUI&&!outputMode&&<button onClick={()=>setShowUI(true)} style={{position:"absolute",top:10,right:10,zIndex:50,background:"rgba(0,0,0,0.5)",border:"1px solid #333",color:"#888",borderRadius:6,padding:"5px 12px",cursor:"pointer",fontSize:10,backdropFilter:"blur(8px)"}}>▸ Controls</button>}

      {showUI&&!outputMode&&(
        <div style={{position:"absolute",top:8,right:8,bottom:8,width:250,zIndex:50,background:"rgba(8,8,8,0.93)",border:"1px solid #1a1a1a",borderRadius:10,backdropFilter:"blur(20px)",overflowY:"auto",padding:"10px 12px",boxSizing:"border-box",scrollbarWidth:"thin",scrollbarColor:"#333 transparent"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
            <span style={{fontSize:12,fontWeight:700,color:"#ddd",letterSpacing:-.3}}>Shader Gradient</span>
            <button onClick={()=>setShowUI(false)} style={{background:"none",border:"none",color:"#444",cursor:"pointer",fontSize:15,padding:0}}>×</button>
          </div>
          <button onClick={()=>{setOutputMode(true);setShowUI(false);}} style={{width:"100%",padding:"7px",background:"linear-gradient(135deg,#c22,#a11)",color:"#fff",border:"none",borderRadius:5,cursor:"pointer",fontSize:10,fontWeight:700,letterSpacing:.5,marginBottom:10,textTransform:"uppercase"}}>
            Enter Output Mode (NDI / Syphon)
          </button>

          <Sec title="Audio Reactive">
            <button onClick={toggleAudio} style={{width:"100%",padding:"7px",marginBottom:8,background:audioActive?"linear-gradient(135deg,#1a6,#183)":"linear-gradient(135deg,#333,#222)",color:audioActive?"#fff":"#999",border:"none",borderRadius:5,cursor:"pointer",fontSize:10,fontWeight:700,letterSpacing:.5,textTransform:"uppercase"}}>
              {audioActive?"● Audio Active — Click to Stop":"Enable Microphone / Line Input"}</button>
            {audioActive&&<>
              <Meter label="Bass" value={audioLevels.bass} color="#ff4444"/>
              <Meter label="Mids" value={audioLevels.mid} color="#44aaff"/>
              <Meter label="Treble" value={audioLevels.treble} color="#aa66ff"/>
              <Meter label="Kick" value={audioLevels.kick} color="#ff8800"/>
              <div style={{height:1,background:"#1a1a1a",margin:"6px 0"}}/>
              <Sl label="Input Gain" value={p.audioGain} min={0.1} max={5} step={0.1} onChange={v=>sp("audioGain",v)}/>
              <Sl label="Smoothing" value={p.audioSmoothing} min={0.1} max={.98} step={.01} onChange={v=>sp("audioSmoothing",v)}/>
              <Sl label="Kick Threshold" value={p.kickThreshold} min={.05} max={.8} step={.01} onChange={v=>sp("kickThreshold",v)}/>
              <Sl label="Kick Decay" value={p.kickDecay} min={.8} max={.99} step={.01} onChange={v=>sp("kickDecay",v)}/>
              <div style={{height:1,background:"#1a1a1a",margin:"6px 0"}}/>
              <div style={{fontSize:9,color:"#555",marginBottom:4,fontWeight:600,textTransform:"uppercase",letterSpacing:1}}>Map Bands → Parameters</div>
              <MapRow label="Bass" color="#ff4444" target={p.bassTarget} amount={p.bassAmount} onTarget={v=>sp("bassTarget",v)} onAmount={v=>sp("bassAmount",v)}/>
              <MapRow label="Mids" color="#44aaff" target={p.midTarget} amount={p.midAmount} onTarget={v=>sp("midTarget",v)} onAmount={v=>sp("midAmount",v)}/>
              <MapRow label="Treble" color="#aa66ff" target={p.trebleTarget} amount={p.trebleAmount} onTarget={v=>sp("trebleTarget",v)} onAmount={v=>sp("trebleAmount",v)}/>
              <MapRow label="Kick" color="#ff8800" target={p.kickTarget} amount={p.kickAmount} onTarget={v=>sp("kickTarget",v)} onAmount={v=>sp("kickAmount",v)}/>
            </>}
          </Sec>

          <Sec title="Colors">
            <CI label="Color 1" value={p.color1} onChange={v=>sp("color1",v)}/>
            <CI label="Color 2" value={p.color2} onChange={v=>sp("color2",v)}/>
            <CI label="Color 3" value={p.color3} onChange={v=>sp("color3",v)}/>
            <CI label="Background" value={p.bgColor} onChange={v=>sp("bgColor",v)}/>
          </Sec>

          <Sec title="Shape">
            <div style={{display:"flex",flexWrap:"wrap",gap:3,marginBottom:6}}>
              {shapeTypes.filter(t=>t!=="custom"||customGeoRef.current).map(t=><button key={t} onClick={()=>sp("type",t)}
                style={{padding:"4px 8px",fontSize:9,fontWeight:p.type===t?700:400,background:p.type===t?"#333":"#151515",color:p.type===t?"#fff":"#555",border:"1px solid #2a2a2a",borderRadius:3,cursor:"pointer"}}>{shapeLabels[t]}</button>)}
            </div>

            {/* Model loader */}
            <input ref={fileInputRef} type="file" accept=".glb,.gltf,.obj,.stl" style={{display:"none"}}
              onChange={e=>{if(e.target.files[0])handleModelFile(e.target.files[0]);e.target.value="";}}/>
            <div
              onClick={()=>fileInputRef.current?.click()}
              style={{padding:10,marginBottom:8,border:"2px dashed #2a2a2a",borderRadius:6,textAlign:"center",cursor:"pointer",
                background:modelLoading?"#1a1a0a":"#0a0a0a",transition:"all 0.2s"}}>
              {modelLoading ? (
                <div style={{fontSize:10,color:"#ff8800"}}>Loading model...</div>
              ) : modelName ? (
                <div>
                  <div style={{fontSize:10,color:"#0bf",fontWeight:700,marginBottom:2}}>✓ {modelName}</div>
                  <div style={{fontSize:8,color:"#555"}}>Click to replace · Drag & drop anywhere</div>
                </div>
              ) : (
                <div>
                  <div style={{fontSize:10,color:"#888",marginBottom:2}}>Load 3D Model</div>
                  <div style={{fontSize:8,color:"#555"}}>Click or drag & drop · .glb .gltf .obj .stl</div>
                </div>
              )}
            </div>

            {p.type==="custom"&&<>
              <Sl label="Model Scale" value={p.modelScale} min={0.1} max={5} step={0.1} onChange={v=>{
                sp("modelScale",v);
                // Force rebuild
                if(customGeoRef.current){
                  const s=sceneRef.current;
                  s.scene.remove(s.mesh);s.geo.dispose();s.mat.dispose();
                  s.geo=customGeoRef.current.clone();s.geo.scale(v,v,v);
                  applyVC(s.geo,p.color1,p.color2,p.color3,p.density);
                  s.mat=createMat(s.envMap,"custom",s.cu);s.mat.roughness=p.roughness;s.mat.metalness=p.metalness;
                  s.mat.envMapIntensity=p.reflection*10;s.mat.wireframe=p.wireframe;
                  s.mesh=new THREE.Mesh(s.geo,s.mat);s.scene.add(s.mesh);lastColorsRef.current="";
                }
              }}/>
            </>}

            <Sl label="Rotation X" value={p.rotX} min={0} max={360} step={1} onChange={v=>sp("rotX",v)}/>
            <Sl label="Rotation Y" value={p.rotY} min={0} max={360} step={1} onChange={v=>sp("rotY",v)}/>
            <Sl label="Rotation Z" value={p.rotZ} min={0} max={360} step={1} onChange={v=>sp("rotZ",v)}/>
            <div style={{display:"flex",alignItems:"center",gap:6,marginTop:2}}>
              <input type="checkbox" checked={p.wireframe} onChange={e=>sp("wireframe",e.target.checked)} style={{accentColor:"#555"}}/>
              <span style={{fontSize:10,color:"#777"}}>Wireframe</span>
            </div>
          </Sec>

          <Sec title="Noise / Motion">
            <Sl label="Speed" value={p.speed} min={0} max={3} step={.05} onChange={v=>sp("speed",v)}/>
            <Sl label="Strength" value={p.strength} min={0} max={10} step={.1} onChange={v=>sp("strength",v)}/>
            <Sl label="Density" value={p.density} min={.1} max={5} step={.05} onChange={v=>sp("density",v)}/>
            <Sl label="Frequency (Spiral)" value={p.frequency} min={.1} max={14} step={.1} onChange={v=>sp("frequency",v)}/>
            <Sl label="Amplitude" value={p.amplitude} min={0} max={5} step={.05} onChange={v=>sp("amplitude",v)}/>
          </Sec>

          <Sec title="Camera" open={false}>
            <Sl label="Distance" value={p.camDist} min={.5} max={12} step={.1} onChange={v=>sp("camDist",v)}/>
            <Sl label="Azimuth" value={p.camAzimuth} min={0} max={360} step={1} onChange={v=>sp("camAzimuth",v)}/>
            <Sl label="Polar" value={p.camPolar} min={1} max={179} step={1} onChange={v=>sp("camPolar",v)}/>
            <Sl label="Zoom" value={p.zoom} min={.2} max={3} step={.05} onChange={v=>sp("zoom",v)}/>
            <Sl label="FOV" value={p.fov} min={10} max={120} step={1} onChange={v=>sp("fov",v)}/>
          </Sec>

          <Sec title="Position" open={false}>
            <Sl label="X" value={p.posX} min={-5} max={5} step={.1} onChange={v=>sp("posX",v)}/>
            <Sl label="Y" value={p.posY} min={-5} max={5} step={.1} onChange={v=>sp("posY",v)}/>
            <Sl label="Z" value={p.posZ} min={-5} max={5} step={.1} onChange={v=>sp("posZ",v)}/>
          </Sec>

          <Sec title="Material & Lighting" open={false}>
            <div style={{display:"flex",gap:3,marginBottom:6}}>
              {[["3D Light",0],["Env Only",1]].map(([l,v])=><button key={v} onClick={()=>sp("lightType",v)} style={{flex:1,padding:"4px 0",fontSize:9,fontWeight:p.lightType===v?700:400,background:p.lightType===v?"#333":"#151515",color:p.lightType===v?"#fff":"#555",border:"1px solid #2a2a2a",borderRadius:3,cursor:"pointer"}}>{l}</button>)}
            </div>
            <div style={{display:"flex",gap:3,marginBottom:6}}>
              {["city","dawn","lobby"].map(e=><button key={e} onClick={()=>sp("envPreset",e)} style={{flex:1,padding:"4px 0",fontSize:9,fontWeight:p.envPreset===e?700:400,background:p.envPreset===e?"#333":"#151515",color:p.envPreset===e?"#fff":"#555",border:"1px solid #2a2a2a",borderRadius:3,cursor:"pointer",textTransform:"capitalize"}}>{e}</button>)}
            </div>
            <Sl label="Brightness" value={p.brightness} min={.3} max={2.5} step={.05} onChange={v=>sp("brightness",v)}/>
            <Sl label="Env Reflection" value={p.reflection} min={0} max={1} step={.01} onChange={v=>sp("reflection",v)}/>
            <Sl label="Roughness" value={p.roughness} min={0} max={1} step={.01} onChange={v=>sp("roughness",v)}/>
            <Sl label="Metalness" value={p.metalness} min={0} max={1} step={.01} onChange={v=>sp("metalness",v)}/>
            <Sl label="Grain" value={p.grain} min={0} max={1} step={.01} onChange={v=>sp("grain",v)}/>
            <Sl label="Grain Blending" value={p.grainBlend} min={0} max={1} step={.01} onChange={v=>sp("grainBlend",v)}/>
          </Sec>

          <Sec title="Presets" open={false}>
            <div style={{display:"flex",flexDirection:"column",gap:3}}>
              {presets.map(pr=><button key={pr.name} onClick={()=>setP(prev=>({...prev,...pr}))}
                style={{padding:"5px 6px",fontSize:9,background:"#111",color:"#999",border:"1px solid #222",borderRadius:3,cursor:"pointer",textAlign:"left"}}>
                <div style={{display:"flex",alignItems:"center",gap:5}}>
                  <div style={{display:"flex",gap:2}}>{[pr.color1,pr.color2,pr.color3].map((c,i)=><div key={i} style={{width:8,height:8,borderRadius:2,background:c}}/>)}</div>
                  <span>{pr.name}</span><span style={{color:"#444",fontSize:8,marginLeft:"auto"}}>{shapeLabels[pr.type]}</span>
                </div></button>)}
            </div>
          </Sec>

          <div style={{marginTop:6,padding:7,background:"#0a0a0a",borderRadius:5,border:"1px solid #181818"}}>
            <p style={{fontSize:8,color:"#444",lineHeight:1.5,margin:0}}>
              <strong style={{color:"#666"}}>Tip:</strong> Drag any .glb/.gltf/.obj/.stl file onto the window to load it as a shape. The noise displacement + colours will be applied automatically.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}