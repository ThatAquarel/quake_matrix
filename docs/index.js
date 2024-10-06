import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.124/build/three.module.js";

import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.124/examples/jsm/controls/OrbitControls.js";

// Set up scene, camera, and renderer
var textureURL =
  "https://s3-us-west-2.amazonaws.com/s.cdpn.io/17271/lroc_color_poles_1k.jpg";
var displacementURL =
  "https://s3-us-west-2.amazonaws.com/s.cdpn.io/17271/ldem_3_8bit.jpg";
var worldURL = "https://s3-us-west-2.amazonaws.com/s.cdpn.io/17271/hipp8_s.jpg";

var scene = new THREE.Scene();

var camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);

var renderer = new THREE.WebGLRenderer();

var controls = new OrbitControls(camera, renderer.domElement);
controls.enablePan = false;

const canvasContainer = document.getElementById("canvas-container");

// Append the renderer's canvas to the container
canvasContainer.appendChild(renderer.domElement);

// Set the initial renderer size
function resize() {
  const width = canvasContainer.clientWidth;
  const height = canvasContainer.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

var geometry = new THREE.SphereGeometry(2, 60, 60);

var textureLoader = new THREE.TextureLoader();
var texture = textureLoader.load(textureURL);
var displacementMap = textureLoader.load(displacementURL);
var worldTexture = textureLoader.load(worldURL);

var material = new THREE.MeshPhongMaterial({
  color: 0xffffff,
  map: texture,
  displacementMap: displacementMap,
  displacementScale: 0.06,
  bumpMap: displacementMap,
  bumpScale: 0.04,
  reflectivity: 0,
  shininess: 0,
});

var moon = new THREE.Mesh(geometry, material);

const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(-100, 10, 50);
scene.add(light);

var worldGeometry = new THREE.SphereGeometry(1000, 60, 60);
var worldMaterial = new THREE.MeshBasicMaterial({
//   color: 0x424242,
color: 0x000000,
  map: worldTexture,
  side: THREE.BackSide,
});
var world = new THREE.Mesh(worldGeometry, worldMaterial);
scene.add(world);
scene.add(moon);

// Generate random points
const pointCount = 1000;
const positions = new Float32Array(pointCount * 3);
for (let i = 0; i < pointCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 100; // x
    positions[i * 3 + 1] = (Math.random() - 0.5) * 100; // y
    positions[i * 3 + 2] = (Math.random() - 0.5) * 100; // z
}

// Create a geometry and add the positions
const buf_geom = new THREE.BufferGeometry();
buf_geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

// Create a material for the points
const buf_material = new THREE.PointsMaterial({ color: 0xffffff, size: 0.0001 });

// Create the points object
const points = new THREE.Points(buf_geom, buf_material);
scene.add(points);

camera.position.z = 5;

moon.rotation.x = 3.1415 * 0.02;
moon.rotation.y = 3.1415 * 1.54;

function animate() {
  requestAnimationFrame(animate);
  moon.rotation.y += 0.0005;
  moon.rotation.x += 0.00002;
  world.rotation.y += 0.00002;
  world.rotation.x += 0.0001;
  points.rotation.y += 0.00002;
  points.rotation.x += 0.0001;

  renderer.render(scene, camera);
}
animate();

// Call resize initially and on window resize
resize();
window.addEventListener("resize", resize);

// Start the animation loop
animate();
