# Shader Gradient

A real-time audio-reactive shader gradient generator with PBR rendering, custom 3D model support, and NDI/Syphon output for live visuals and projection mapping.

Built with Three.js and React. Runs in any modern browser — deploy to GitHub Pages and use anywhere.

## Live App

👉 **[Launch Shader Gradient](https://zeroonetwo-creativetechnologyhub.github.io/shader-gradient-app/)**

## Features

### Visual Engine
- **PBR rendering** — MeshStandardMaterial with physically-based lighting, roughness, metalness
- **Environment maps** — procedural city, dawn, and lobby presets that produce realistic reflections and that signature colour bleed
- **Noise displacement** — simplex noise vertex deformation with sine-wave twist, injected into the standard material pipeline
- **Film grain** — screen-space grain overlay with amount and blending controls
- **ACES filmic tone mapping** — cinematic colour response

### Shapes
8 built-in geometries, each with tailored displacement:
- **Plane** — the classic shadergradient.co look
- **Water** — layered ocean waves with horizontal drift
- **Sphere** — noise-displaced blob with twist
- **Torus** — ring with wrapping twist distortion
- **Torus Knot** — mathematical knot with pulsing noise
- **Icosahedron** — faceted geodesic sphere with soft deformation
- **Cylinder** — open tube with barrel-wave distortion
- **Blob** — multi-octave noise sphere for organic asymmetric shapes

### Custom 3D Models
- **Drag and drop** any `.glb`, `.gltf`, `.obj`, or `.stl` file onto the window
- Models are auto-centered, scaled, and fitted with vertex colours
- Noise displacement is applied along surface normals
- Adjustable model scale

### Audio Reactive
- **Live microphone / line input** via Web Audio API
- **Frequency analysis** — bass (20–250Hz), mids (250–4kHz), treble (4–16kHz)
- **Kick detection** — onset detection with adjustable threshold and decay
- **Mappable routing** — map any frequency band to any parameter:
  - Strength, density, frequency, amplitude, speed
  - Rotation X/Y/Z, camera distance
  - Brightness, roughness, reflection, grain
- **Tunable** — input gain, smoothing, kick sensitivity

### Output
- **Output Mode** — one click hides all controls for clean fullscreen capture
- **NDI output** via OBS Studio + obs-ndi plugin
- **Syphon output** via OBS Syphon plugin or window capture utilities
- Canvas always fills the full viewport — controls float as an overlay

## Quick Start

### Run locally
```bash
git clone https://github.com/ZeroOneTwo-CreativeTechnologyHub/shader-gradient-app.git
cd shader-gradient-app
npm install
npm run dev
```
Opens at `http://localhost:5173`

### Build static files
```bash
npm run build
```
Output in `dist/` — open `dist/index.html` in any browser.

## NDI / Syphon Output for MadMapper

1. Open the app (GitHub Pages URL or localhost)
2. Click **"Enter Output Mode"** — all UI disappears
3. Open **OBS Studio**
4. Add a **Window Capture** source → select the browser window
5. Install [obs-ndi plugin](https://github.com/obs-ndi/obs-ndi)
6. Go to **Tools → NDI Output Settings** → enable
7. In **MadMapper**, the NDI feed appears as an input source

## Audio Routing

The app uses your microphone or line input. To feed it audio from a DAW, DJ software, or system audio:

### macOS
1. Install [BlackHole](https://existential.audio/blackhole/) (free)
2. Open **Audio MIDI Setup** → create a **Multi-Output Device** (speakers + BlackHole)
3. Set that as your system output
4. In the browser mic prompt, select **BlackHole**

### Windows
1. Install [VB-Cable](https://vb-audio.com/Cable/) (free)
2. Set **VB-Cable Input** as default playback device
3. In the browser mic selection, choose **VB-Cable Output**

## Controls

| Key | Action |
|-----|--------|
| **Esc** | Exit Output Mode |
| **Output Mode button** | Hide all UI for clean capture |
| **Drag & drop** | Load a 3D model (.glb .gltf .obj .stl) |

### Parameter Sections
- **Audio Reactive** — enable mic, live meters, band-to-parameter mapping
- **Colors** — three gradient colours + background
- **Shape** — geometry selection, rotation, wireframe
- **Noise / Motion** — speed, strength, density, frequency, amplitude
- **Camera** — distance, azimuth, polar angle, zoom, FOV
- **Position** — mesh X/Y/Z offset
- **Material & Lighting** — PBR controls, env presets, grain
- **Presets** — instant starting points

## Tech Stack

- [Three.js](https://threejs.org/) — WebGL rendering
- [React](https://react.dev/) — UI
- [Vite](https://vitejs.dev/) — build tool
- [three-stdlib](https://github.com/pmndrs/three-stdlib) — GLTF/OBJ/STL loaders
- Web Audio API — real-time audio analysis

## Credits

Inspired by [shadergradient.co](https://shadergradient.co/) by [ruucm](https://github.com/ruucm/shadergradient). Noise displacement technique based on [Twisted Colorful Spheres](https://tympanus.net/codrops/2021/01/26/twisted-colorful-spheres-with-three-js/) by Mario Carrillo. Simplex noise by Ashima Arts / Stefan Gustavson.

## License

MIT

---

Made by [ZeroOneTwo Creative Technology Hub](https://github.com/ZeroOneTwo-CreativeTechnologyHub)
