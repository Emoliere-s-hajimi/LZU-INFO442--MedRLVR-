// Three.js — animated, parametric "brain" point cloud for the hero.
// Lazy-loaded via BrainTT.ensureThree to keep first-paint small.
//
// Particle counts are reduced on low-DPI screens and small viewports to
// preserve frame rate on mid-range laptops.

(function () {
  const canvas = document.getElementById("brainCanvas");
  if (!canvas) return;

  // Skip if the OS asked for reduced motion — show a static rendering only
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  BrainTT.lazy(canvas, async () => {
    const THREE = await BrainTT.ensureThree();
    if (!THREE) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      50,
      canvas.clientWidth / Math.max(1, canvas.clientHeight),
      0.1,
      100
    );
    camera.position.set(0, 0, 4.6);

    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setClearColor(0x000000, 0);

    // Adaptive particle count
    const big = window.innerWidth >= 1024 && !reduce;
    const N_OUTER = big ? 5500 : 2400;
    const N_INNER = big ? 1800 : 800;

    function brainSDF(p) {
      const lx = p.x - 0.55, rx = p.x + 0.55;
      const h1 = Math.sqrt(lx * lx / 1.1 + p.y * p.y / 1.5 + p.z * p.z / 1.3);
      const h2 = Math.sqrt(rx * rx / 1.1 + p.y * p.y / 1.5 + p.z * p.z / 1.3);
      const temporal = Math.sqrt(p.x * p.x / 1.4 + (p.y + 0.55) * (p.y + 0.55) / 0.6 + p.z * p.z / 1.2);
      return Math.min(h1, h2, temporal) - 1.0;
    }

    function samplePoints(count, shellWidth) {
      const positions = new Float32Array(count * 3);
      const seeds = new Float32Array(count);
      let i = 0, tries = 0;
      while (i < count && tries < count * 80) {
        const x = (Math.random() - 0.5) * 4.0;
        const y = (Math.random() - 0.5) * 3.6;
        const z = (Math.random() - 0.5) * 3.4;
        const d = brainSDF({ x, y, z });
        if (d > -shellWidth && d < shellWidth * 0.4) {
          positions[i * 3]     = x;
          positions[i * 3 + 1] = y;
          positions[i * 3 + 2] = z;
          seeds[i] = Math.random();
          i++;
        }
        tries++;
      }
      return { positions: positions.subarray(0, i * 3), seeds: seeds.subarray(0, i), count: i };
    }

    const outer = samplePoints(N_OUTER, 0.06);
    const inner = samplePoints(N_INNER, 0.22);

    function makeCloud(data, color, size) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
      geom.setAttribute("aSeed", new THREE.BufferAttribute(data.seeds, 1));
      const mat = new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 },
          uColor: { value: new THREE.Color(color) },
          uSize: { value: size },
        },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        vertexShader: `
          attribute float aSeed;
          uniform float uTime;
          uniform float uSize;
          varying float vSeed;
          void main() {
            vSeed = aSeed;
            vec3 p = position;
            float pulse = 0.02 * sin(uTime * 1.2 + aSeed * 8.0);
            p *= 1.0 + pulse;
            vec4 mv = modelViewMatrix * vec4(p, 1.0);
            gl_PointSize = uSize * (300.0 / -mv.z);
            gl_Position = projectionMatrix * mv;
          }
        `,
        fragmentShader: `
          uniform vec3 uColor;
          varying float vSeed;
          void main() {
            vec2 c = gl_PointCoord - vec2(0.5);
            float d = length(c);
            if (d > 0.5) discard;
            float a = smoothstep(0.5, 0.0, d);
            vec3 col = mix(uColor, vec3(1.0, 0.6, 1.0), vSeed * 0.4);
            gl_FragColor = vec4(col, a * 0.9);
          }
        `,
      });
      return new THREE.Points(geom, mat);
    }

    const cloudOuter = makeCloud(outer, 0x22d3ee, 0.55);
    const cloudInner = makeCloud(inner, 0xf472b6, 1.6);
    const group = new THREE.Group();
    group.add(cloudOuter);
    group.add(cloudInner);

    const hullGeo = new THREE.IcosahedronGeometry(1.4, 2);
    const hullMat = new THREE.MeshBasicMaterial({
      color: 0x22d3ee, wireframe: true, transparent: true, opacity: 0.06,
    });
    const hull = new THREE.Mesh(hullGeo, hullMat);
    hull.scale.set(1.6, 1.35, 1.45);
    group.add(hull);

    scene.add(group);

    // Mouse parallax, throttled
    let mouseX = 0, mouseY = 0, targetRX = 0, targetRY = 0;
    let parallaxPending = false;
    window.addEventListener("pointermove", (e) => {
      if (parallaxPending) return;
      parallaxPending = true;
      requestAnimationFrame(() => {
        mouseX = (e.clientX / window.innerWidth) * 2 - 1;
        mouseY = (e.clientY / window.innerHeight) * 2 - 1;
        parallaxPending = false;
      });
    }, { passive: true });

    function resize() {
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const dpr = renderer.getPixelRatio();
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        renderer.setSize(w, h, false);
        camera.aspect = w / Math.max(1, h);
        camera.updateProjectionMatrix();
      }
    }

    // Pause when not visible
    let visible = true;
    const visIo = new IntersectionObserver(
      (entries) => { entries.forEach((e) => (visible = e.isIntersecting)); },
      { threshold: 0.05 }
    );
    visIo.observe(canvas);

    let t0 = performance.now();
    function loop() {
      requestAnimationFrame(loop);
      if (!visible) return;
      resize();
      const t = (performance.now() - t0) * 0.001;
      cloudOuter.material.uniforms.uTime.value = t;
      cloudInner.material.uniforms.uTime.value = t;

      targetRX += (mouseY * 0.3 - targetRX) * 0.04;
      targetRY += (mouseX * 0.3 - targetRY) * 0.04;
      group.rotation.x = targetRX;
      group.rotation.y = t * (reduce ? 0 : 0.18) + targetRY;

      renderer.render(scene, camera);
    }
    loop();
  });
})();
