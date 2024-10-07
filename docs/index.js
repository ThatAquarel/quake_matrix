import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.124/build/three.module.js";

import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.124/examples/jsm/controls/OrbitControls.js";

let texture_url ="https://raw.githubusercontent.com/ThatAquarel/quake_matrix/refs/heads/main/docs/color_poles_1k.jpg";

let scene = new THREE.Scene();

let camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.01,
  800
);

let renderer = new THREE.WebGLRenderer({ antialias: true });
new OrbitControls(camera, renderer.domElement);

const container = document.getElementById("canvas-container");
container.appendChild(renderer.domElement);

function resize() {
  const width = container.clientWidth;
  const height = container.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

let world_geom = new THREE.SphereGeometry(1024, 64, 64);
let word_mat = new THREE.MeshBasicMaterial({
  color: 0x000000,
  side: THREE.BackSide,
});
let world = new THREE.Mesh(world_geom, word_mat);
scene.add(world);

let moon_geom = new THREE.SphereGeometry(2, 60, 60);
let textureLoader = new THREE.TextureLoader();
let moon_texture = textureLoader.load(texture_url);
let moon_mat = new THREE.MeshPhongMaterial({
  color: 0xffffff,
  map: moon_texture
});

let moon = new THREE.Mesh(moon_geom, moon_mat);
scene.add(moon);

const dir_light = new THREE.DirectionalLight(0xededed, 1);
dir_light.position.set(-128, 16, 64);
scene.add(dir_light);

const point_n = 4096;
const spread = 256;
const positions = new Float32Array(point_n * 3);
for (let i = 0; i < point_n; i++) {
  positions[i * 3] = (Math.random() - 0.5) * spread;
  positions[i * 3 + 1] = (Math.random() - 0.5) * spread;
  positions[i * 3 + 2] = (Math.random() - 0.5) * spread;
}

const buf_geom = new THREE.BufferGeometry();
buf_geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
const buf_material = new THREE.PointsMaterial({
  color: 0xffffff,
  size: 0.0002,
});
const points = new THREE.Points(buf_geom, buf_material);
scene.add(points);

moon.rotation.x = 3.1415 * 0.02;
moon.rotation.y = 3.1415 * 1.54;

function ease_out_cubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

const duration = 5000;
const start = Date.now();

function animate() {
  requestAnimationFrame(animate);
  const dt = Date.now() - start;
  const progress = Math.min(dt / duration, 1);

  const position = ease_out_cubic(1/(progress));
  camera.position.z = position * 5

  moon.rotation.y += 0.0005;
  moon.rotation.x += 0.00002;
  world.rotation.y += 0.00002;
  world.rotation.x += 0.0001;
  points.rotation.y += 0.00002;
  points.rotation.x += 0.0001;

  renderer.render(scene, camera);
}
animate();

resize();
window.addEventListener("resize", resize);

animate();
